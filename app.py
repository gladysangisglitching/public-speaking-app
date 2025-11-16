from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from openai import OpenAI
import tempfile
import subprocess
import re

app = Flask(__name__, static_folder='static')
CORS(app)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

FILLER_WORDS = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
                'literally', 'right', 'okay', 'well', 'i mean']

def get_video_duration(video_path):
    """Get video duration using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Could not read video duration")
    return float(result.stdout.strip())

def extract_audio_ffmpeg(video_path, audio_path):
    """Extract audio using ffmpeg."""
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'libmp3lame',
        '-ar', '16000', '-ac', '1', '-b:a', '64k', '-y',
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Could not extract audio from video")

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper."""
    with open(audio_path, 'rb') as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
            language="en",
            temperature=0,
            prompt="This is a public speaking practice video. Include all filler words like um, uh, like."
        )
    return transcript

def count_filler_words(text):
    """Count filler words in transcript."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    filler_counts = {}
    total_count = 0
    
    for filler in FILLER_WORDS:
        if ' ' in filler:
            count = text_lower.count(filler)
        else:
            count = words.count(filler)
        if count > 0:
            filler_counts[filler] = count
            total_count += count
    
    return total_count, filler_counts

def assess_pace(wpm):
    """Assess speaking pace."""
    if 140 <= wpm <= 160:
        return "Excellent pace! You're speaking at an ideal rate."
    elif 120 <= wpm < 140:
        return "Good pace, slightly slow. Consider speaking a bit faster."
    elif 160 < wpm <= 180:
        return "Good pace, slightly fast. Consider slowing down slightly."
    elif wpm < 120:
        return "Too slow. Try to increase your pace."
    else:
        return "Too fast. Slow down for clarity."

# Serve the frontend HTML
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    video_path = None
    audio_path = None
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        file_ext = os.path.splitext(video_file.filename)[1].lower() or '.mp4'

        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_video:
            video_file.save(tmp_video.name)
            video_path = tmp_video.name

        # Get video duration
        duration_seconds = get_video_duration(video_path)

        # Extract audio
        audio_path = video_path.replace(file_ext, '.mp3')
        extract_audio_ffmpeg(video_path, audio_path)

        # Transcribe audio
        transcript = transcribe_audio(audio_path)

        # Analyze transcript
        word_count = len(transcript.split())
        words_per_min = int((word_count / duration_seconds) * 60) if duration_seconds > 0 else 0
        filler_count, filler_breakdown = count_filler_words(transcript)
        filler_rate_per_min = round(filler_count / (duration_seconds / 60), 2) if duration_seconds > 0 else 0
        pace_assessment = assess_pace(words_per_min)

        return jsonify({
            'transcript': transcript,
            'duration': round(duration_seconds, 2),
            'word_count': word_count,
            'words_per_min': words_per_min,
            'filler_count': filler_count,
            'filler_rate_per_min': filler_rate_per_min,
            'filler_words': filler_breakdown,
            'assessment': {'pace': pace_assessment}
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup temp files
        try:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
        except:
            pass

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)