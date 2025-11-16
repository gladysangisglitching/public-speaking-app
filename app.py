from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from openai import OpenAI
import tempfile
import subprocess
import re
import time

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Increase timeouts for large files
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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
    """Extract audio using ffmpeg with compression for large files."""
    try:
        # Get duration
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
        
        if duration_result.returncode != 0:
            raise Exception(f"Failed to get video duration: {duration_result.stderr}")
        
        duration = float(duration_result.stdout.strip())

        # Extract and compress audio - optimize for Whisper API (25MB limit)
        # Use lower bitrate for longer videos
        bitrate = '32k' if duration > 120 else '64k'
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-ar', '16000',  # 16kHz sample rate (Whisper works well with this)
            '-ac', '1',  # Mono
            '-b:a', bitrate,  # Lower bitrate for compression
            '-y',  # Overwrite
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
            
        # Check output file size
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            # Whisper API has 25MB limit
            if file_size > 24 * 1024 * 1024:
                # Re-encode with even lower bitrate
                temp_audio = audio_path + '.tmp.mp3'
                cmd = [
                    'ffmpeg', '-i', audio_path,
                    '-b:a', '16k',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    temp_audio
                ]
                subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                os.replace(temp_audio, audio_path)
        
        return duration
        
    except subprocess.TimeoutExpired:
        raise Exception("Audio extraction timed out. File may be too large.")
    except Exception as e:
        raise Exception(f"Audio extraction failed: {str(e)}")

def transcribe_audio(audio_path):
    """Transcribe audio with Whisper, handling large files."""
    try:
        file_size = os.path.getsize(audio_path)
        
        # Whisper API has 25MB limit
        if file_size > 24 * 1024 * 1024:
            raise Exception(f"Audio file too large ({file_size / (1024*1024):.1f}MB). Maximum is 25MB.")
        
        with open(audio_path, 'rb') as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language="en",
                temperature=0,
                prompt="This is a public speaking practice video. Transcribe everything including filler words like um, uh, like, you know."
            )
        return transcript
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

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
    video_path = None
    audio_path = None
    
    try:
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({'error': 'OpenAI API key not configured'}), 500

        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        
        if not video_file.filename:
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded video to temp file
        file_ext = os.path.splitext(video_file.filename)[1].lower()
        if not file_ext:
            file_ext = '.mp4'
        
        if file_ext not in ['.mp4', '.webm', '.mov', '.avi', '.mkv']:
            return jsonify({'error': f'Unsupported format: {file_ext}'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_video:
            video_file.save(tmp_video.name)
            video_path = tmp_video.name

        # Extract audio
        audio_path = video_path.replace(file_ext, '.mp3')
        duration = extract_audio_ffmpeg(video_path, audio_path)

        # Check duration limit (5 minutes)
        if duration > 300:
            return jsonify({
                'error': f'Video too long ({duration/60:.1f} min). Please use a video under 5 minutes.'
            }), 400

        # Transcribe
        transcript = transcribe_audio(audio_path)

        # Metrics
        words = transcript.split()
        word_count = len(words)
        words_per_min = int((word_count / duration) * 60) if duration > 0 else 0
        filler_count, filler_breakdown = count_filler_words_detailed(transcript)
        filler_rate_per_min = round(filler_count / (duration / 60), 2) if duration > 0 else 0
        pace_assessment = assess_pace(words_per_min)

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
        error_msg = str(e)
        if 'timed out' in error_msg.lower():
            error_msg = 'Processing timed out. Please try a shorter or smaller video.'
        return jsonify({'error': error_msg}), 500
        
    finally:
        # Cleanup temp files
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Server running'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)