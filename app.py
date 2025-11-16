from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from openai import OpenAI
import tempfile
import subprocess
import re


app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Serve HTML file
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

FILLER_WORDS = [
    'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
    'literally', 'right', 'okay'
]


def extract_audio_ffmpeg(video_path, audio_path):
    """Extract audio from video using ffmpeg"""
    try:
        # Get duration
        duration_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        if duration_result.returncode != 0:
            raise Exception(f"Could not read duration: {duration_result.stderr}")
        
        duration = float(duration_result.stdout.strip())

        # Extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'libmp3lame',
            '-ar', '16000', '-ac', '1', '-b:a', '64k',
            '-y', audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Audio extraction failed: {result.stderr}")

        return duration
    except Exception as e:
        print(f"Audio extraction error: {e}")
        raise


def compress_video(input_path, output_path, target_size_mb=15):
    """Compress video size to avoid OOM failures"""
    try:
        duration_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        if duration_result.returncode != 0:
            return input_path

        duration = float(duration_result.stdout.strip())

        # Estimate target bitrate
        target_bitrate = int((target_size_mb * 8 * 1024 / duration) * 0.8)

        compress_cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-b:v', f'{target_bitrate}k',
            '-maxrate', f'{int(target_bitrate * 1.5)}k',
            '-bufsize', f'{int(target_bitrate * 2)}k',
            '-c:a', 'aac', '-b:a', '64k',
            '-preset', 'fast', '-movflags', '+faststart',
            '-y', output_path
        ]
        result = subprocess.run(compress_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return input_path

        return output_path
    except:
        return input_path


def transcribe_audio(audio_path):
    """Transcribe using Whisper API"""
    try:
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en",
                temperature=0,
                prompt="Transcribe all words including filler words."
            )
        return transcript
    except Exception as e:
        print(f"Transcription error: {e}")
        raise


def count_filler_words_detailed(text):
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    filler_counts = {}
    total = 0

    for filler in FILLER_WORDS:
        if ' ' in filler:
            count = text_lower.count(filler)
        else:
            count = words.count(filler)
        if count > 0:
            filler_counts[filler] = count
            total += count

    return total, filler_counts


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


@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
        
        video_file = request.files['video']

        # Size check (50MB)
        video_file.seek(0, 2)
        file_size = video_file.tell()
        video_file.seek(0)

        if file_size > 50 * 1024 * 1024:
            return jsonify({'error': 'File too large. Please use a video under 50MB (~3 min).'}), 400

        # Save temp file
        ext = os.path.splitext(video_file.filename)[1].lower()
        if ext not in ['.mp4', '.webm', '.mov', '.avi']:
            ext = '.mp4'

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_vid:
            video_file.save(temp_vid.name)
            video_path = temp_vid.name

        # Compress if larger than 20MB
        if file_size > 20 * 1024 * 1024:
            compressed_path = video_path.replace(ext, '_compressed.mp4')
            video_path = compress_video(video_path, compressed_path)

        # Extract audio
        audio_path = video_path.replace(ext, '.mp3')
        duration = extract_audio_ffmpeg(video_path, audio_path)

        # Duration check
        if duration > 180:  # 3 minutes
            os.unlink(video_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return jsonify({'error': f'Video too long ({duration/60:.1f} min). Max is 3 minutes.'}), 400

        # Transcribe
        transcript = transcribe_audio(audio_path)

        # Stats
        words = transcript.split()
        wc = len(words)
        wpm = int((wc / duration) * 60)

        filler_count, breakdown = count_filler_words_detailed(transcript)
        filler_rate = round(filler_count / (duration / 60), 2)

        result = {
            'transcript': transcript,
            'duration': duration,
            'word_count': wc,
            'words_per_min': wpm,
            'filler_count': filler_count,
            'filler_rate_per_min': filler_rate,
            'filler_breakdown': breakdown,
            'assessment': {'pace': assess_pace(wpm)}
        }

        # Cleanup
        try:
            os.unlink(video_path)
            os.unlink(audio_path)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
