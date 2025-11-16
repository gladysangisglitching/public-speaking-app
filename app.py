from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from openai import OpenAI
import tempfile
import subprocess
import re

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Serve frontend
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

FILLER_WORDS = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
                'literally', 'right', 'okay']

# ----- Utility Functions -----

def extract_audio_ffmpeg(video_path, audio_path):
    """Extract audio using ffmpeg."""
    # Get duration
    duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
    duration = float(duration_result.stdout.strip())

    # Extract audio
    cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'libmp3lame', '-ar', '16000', '-ac', '1', '-b:a', '64k', '-y', audio_path]
    subprocess.run(cmd, capture_output=True, text=True)
    return duration

def transcribe_audio(audio_path):
    """Transcribe audio with Whisper."""
    with open(audio_path, 'rb') as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
            language="en",
            temperature=0,
            prompt="This is a public speaking practice video. Transcribe everything including filler words."
        )
    return transcript

def count_filler_words_detailed(text):
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

# ----- Routes -----

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({'error': 'OpenAI API key not configured'}), 500

        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']

        # Save uploaded video to temp file
        file_ext = os.path.splitext(video_file.filename)[1] or '.mp4'
        if file_ext.lower() not in ['.mp4', '.webm', '.mov', '.avi']:
            file_ext = '.mp4'

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_video:
            video_file.save(tmp_video.name)
            video_path = tmp_video.name

        # Extract audio
        audio_path = video_path.replace(file_ext, '.mp3')
        duration = extract_audio_ffmpeg(video_path, audio_path)

        # Reject if > 3 minutes
        if duration > 180:
            os.unlink(video_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return jsonify({'error': f'Video too long ({duration/60:.1f} min). Please use <3 min.'}), 400

        # Transcribe
        transcript = transcribe_audio(audio_path)

        # Metrics
        words = transcript.split()
        word_count = len(words)
        words_per_min = int((word_count / duration) * 60) if duration > 0 else 0
        filler_count, filler_breakdown = count_filler_words_detailed(transcript)
        filler_rate_per_min = round(filler_count / (duration / 60), 2) if duration > 0 else 0
        pace_assessment = assess_pace(words_per_min)

        # Cleanup
        os.unlink(video_path)
        os.unlink(audio_path)

        return jsonify({
            'transcript': transcript,
            'duration': duration,
            'word_count': word_count,
            'words_per_min': words_per_min,
            'filler_count': filler_count,
            'filler_rate_per_min': filler_rate_per_min,
            'filler_breakdown': filler_breakdown,
            'assessment': {'pace': pace_assessment}
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Server running'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Allow uploads up to 100MB
    app.run(host='0.0.0.0', port=port, debug=False)
