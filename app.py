from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from openai import OpenAI
import tempfile
import subprocess
from collections import Counter
import re


app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Serve HTML file
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Initialize OpenAI client - gets API key from environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

FILLER_WORDS = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 
                'literally', 'right', 'okay']

def extract_audio_ffmpeg(video_path, audio_path):
    """Extract audio using ffmpeg directly"""
    try:
        # Get duration
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        print(f"Video duration: {duration} seconds")
        
        # Extract audio with better quality settings
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'libmp3lame', 
               '-ar', '16000', '-ac', '1', '-b:a', '64k', '-y', audio_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Audio extraction complete")
        
        # Verify audio file was created
        if not os.path.exists(audio_path):
            raise Exception("Audio file was not created")
        
        audio_size = os.path.getsize(audio_path)
        print(f"Audio file size: {audio_size / (1024*1024):.2f} MB")
        
        return duration
    except Exception as e:
        print(f"Audio extraction error: {e}")
        raise

def compress_video(input_path, output_path, target_size_mb=15):
    """Compress video to target size while maintaining quality"""
    try:
        # Get current video info
        probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=duration,bit_rate',
                     '-of', 'default=noprint_wrappers=1', input_path]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        # Get duration
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        
        # Calculate target bitrate (80% of target to leave room for audio)
        target_bitrate = int((target_size_mb * 8 * 1024 / duration) * 0.8)  # in kbps
        
        print(f"Compressing video to ~{target_size_mb}MB (target bitrate: {target_bitrate}kbps)")
        
        # Compress video
        compress_cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',  # Use H.264 codec
            '-b:v', f'{target_bitrate}k',  # Target video bitrate
            '-maxrate', f'{target_bitrate * 1.5}k',  # Max bitrate
            '-bufsize', f'{target_bitrate * 2}k',  # Buffer size
            '-c:a', 'aac',  # Audio codec
            '-b:a', '64k',  # Audio bitrate
            '-preset', 'fast',  # Encoding speed
            '-movflags', '+faststart',  # Enable streaming
            '-y',  # Overwrite output
            output_path
        ]
        
        subprocess.run(compress_cmd, check=True, capture_output=True)
        
        compressed_size = os.path.getsize(output_path)
        print(f"Compressed video size: {compressed_size / (1024*1024):.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"Compression error: {e}")
        # If compression fails, return original
        return input_path

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        # Check file size
        file_size = os.path.getsize(audio_path)
        print(f"Audio file size: {file_size / (1024*1024):.2f} MB")
        
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en",
                temperature=0,
                prompt="This is a public speaking practice video. Please transcribe everything including all filler words and incomplete sentences."
            )
        
        full_text = transcript
        print(f"Transcription length: {len(full_text)} characters")
        print(f"Word count: {len(full_text.split())}")
        return full_text
    except Exception as e:
        print(f"Transcription error: {e}")
        raise

def count_filler_words_detailed(text):
    """Count filler words in transcript with detailed breakdown"""
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
    """Assess speaking pace"""
    if 140 <= wpm <= 160:
        return "Excellent pace! You're speaking at an ideal rate for audience comprehension."
    elif 120 <= wpm < 140:
        return "Good pace, but slightly slow. Consider speaking a bit faster to maintain energy."
    elif 160 < wpm <= 180:
        return "Good pace, but slightly fast. Consider slowing down slightly for clarity."
    elif wpm < 120:
        return "Too slow. Your audience may lose interest. Try to increase your pace."
    else:
        return "Too fast. Slow down to ensure your audience can follow along."

@app.route('/analyze', methods=['POST'])
def analyze_video():
    print("=== ANALYZE ENDPOINT HIT ===")
    
    # Check API key first
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not set!")
        return jsonify({'error': 'OpenAI API key not configured on server'}), 500
    
    try:
        if 'video' not in request.files:
            print("ERROR: No video file in request")
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        print(f"Received video: {video_file.filename}")
        
        # Check file size
        video_file.seek(0, 2)
        file_size = video_file.tell()
        video_file.seek(0)
        
        print(f"Original file size: {file_size / (1024*1024):.2f} MB")
        
        # Increased limit to 50MB since we'll compress
        if file_size > 50 * 1024 * 1024:
            return jsonify({'error': 'Video file too large. Please use a video under 50MB (about 3 minutes).'}), 400
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as video_temp:
            video_file.save(video_temp.name)
            original_video_path = video_temp.name
        
        # Compress video if it's large
        if file_size > 20 * 1024 * 1024:  # Compress if over 20MB
            print("Video is large, compressing...")
            compressed_video_path = original_video_path.replace('.mp4', '_compressed.mp4')
            video_path = compress_video(original_video_path, compressed_video_path, target_size_mb=15)
            # Clean up original if compression succeeded
            if video_path != original_video_path:
                os.unlink(original_video_path)
        else:
            video_path = original_video_path
        
        audio_path = video_path.replace('.mp4', '.mp3')
        
        # Extract audio and get duration
        print("Extracting audio from video...")
        duration = extract_audio_ffmpeg(video_path, audio_path)
        
        # Check duration limit (3 minutes max to avoid timeouts)
        if duration > 180:  # 3 minutes
            os.unlink(video_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return jsonify({'error': f'Video is too long ({duration/60:.1f} minutes). Please use videos under 3 minutes.'}), 400
        
        # Transcribe using Whisper
        print("Transcribing with Whisper API...")
        transcript = transcribe_audio(audio_path)
        print(f"Transcription complete: {len(transcript)} characters")
        
        # Calculate metrics
        words = transcript.split()
        word_count = len(words)
        words_per_min = int((word_count / duration) * 60) if duration > 0 else 0
        
        # Get detailed filler word analysis
        filler_count, filler_breakdown = count_filler_words_detailed(transcript)
        
        # Calculate filler rate per minute
        duration_minutes = duration / 60 if duration > 0 else 1
        filler_rate_per_min = round(filler_count / duration_minutes, 2)
        
        pace_assessment = assess_pace(words_per_min)
        
        # Clean up temp files
        try:
            os.unlink(video_path)
            os.unlink(audio_path)
            print("Cleaned up temp files")
        except:
            pass
        
        result = {
            'transcript': transcript,
            'duration': duration,
            'word_count': word_count,
            'words_per_min': words_per_min,
            'filler_count': filler_count,
            'filler_rate_per_min': filler_rate_per_min,
            'filler_breakdown': filler_breakdown,
            'assessment': {
                'pace': pace_assessment
            }
        }
        
        print("=== ANALYSIS COMPLETE ===")
        return jsonify(result)
    
    except Exception as e:
        print(f"ERROR in analyze_video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Server is running'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting server on port {port}")
    print(f"Static folder: {app.static_folder}")
    print(f"OpenAI API Key set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'NO - MISSING!'}")
    app.run(host='0.0.0.0', port=port, debug=False)