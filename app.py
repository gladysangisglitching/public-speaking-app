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

# Filler words
FILLER_WORDS = [
    'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
    'literally', 'right', 'okay'
]

# --- Helper functions ---

def extract_audio_ffmpeg(video_path, audio_path):
    """Extract audio using ffmpeg"""
    try:
        # Get duration
        duration_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(duration_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        duration = float(result.stdout.strip())

        # Extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'libmp3lame',
            '-ar', '16000', '-ac', '1', '-b:a', '64k',
            '-y', audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)

        return duration
    except Exception as e:
        print(f"Audio extraction error: {e}")
        raise

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper"""
    try:
        with open(audio_path, 'rb') as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language="en",
                temperature=0,
                prompt="This is a public speaking practice video. Transcribe all words, including filler words."
            )
        return transcript
    except Exception as e:
        print(f"Transcription error: {e}")
        raise

def count_filler_words(text):
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    counts = {}
    total = 0
    for filler in FILLER_WORDS:
        if ' ' in filler:
            c = text_lower.count(filler)
        else:
            c = words.count(filler)
        if c > 0:
            counts[filler] = c
            total += c
    return total, counts

def assess_pace(wpm):
    if 140 <= wpm <= 160:
        return "Excellent pace!"
    elif 120 <= wpm < 140:
        return "Slightly slow — speed up a bit."
    elif 160 < wpm <= 180:
        return "Slightly fast — slow down slightly."
    elif wpm < 120:
        return "Too slow — increase your pace."
    else:
        return "Too fast — slow down."

# --- Routes ---

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400

        video_file = request.files['video']

        # File size check (25MB max)
        video_file.seek(0, 2)
        size = video_file.tell()
        video_file.seek(0)
        if size > 25 * 1024 * 1024:
            return jsonify({'error': 'Video too large. Max 25MB (~1–2 min).'}), 400

        # Save temp video
        ext = os.path.splitext(video_file.filename)[1].lower()
        if ext not in ['.mp4', '.mov', '.webm', '.avi']:
            ext = '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_vid:
            video_file.save(tmp_vid.name)
            video_path = tmp_vid.name

        # Audio path
        audio_path = os.path.splitext(video_path)[0] + '.mp3'

        # Extract audio
        try:
            duration = extract_audio_ffmpeg(video_path, audio_path)
        except Exception as e:
            cleanup([video_path, audio_path])
            return jsonify({'error': f'Audio extraction failed: {str(e)}'}), 500

        # Duration check (3 minutes max)
        if duration > 180:
            cleanup([video_path, audio_path])
            return jsonify({'error': f'Video too long ({duration/60:.1f} min). Max is 3 min.'}), 400

        # Transcribe
        transcript = transcribe_audio(audio_path)

        # Metrics
        words = transcript.split()
        word_count = len(words)
        wpm = int((word_count / duration) * 60) if duration > 0 else 0

        filler_count, filler_breakdown = count_filler_words(transcript)
        filler_rate = round(filler_count / (duration / 60), 2)

        result = {
            'transcript': transcript,
            'duration': duration,
            'word_count': word_count,
            'words_per_min': wpm,
            'filler_count': filler_count,
            'filler_rate_per_min': filler_rate,
            'filler_breakdown': filler_breakdown,
            'assessment': {'pace': assess_pace(wpm)}
        }

        # Cleanup
        cleanup([video_path, audio_path])
        return jsonify(result)

    except Exception as e:
        cleanup([video_path, audio_path] if 'video_path' in locals() else [])
        return jsonify({'error': str(e)}), 500

def cleanup(paths):
    """Delete temp files safely"""
    for f in paths:
        if os.path.exists(f):
            try:
                os.unlink(f)
            except:
                pass

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

# --- Run server ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
