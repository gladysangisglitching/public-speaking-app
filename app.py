from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI
import tempfile
import subprocess
from collections import Counter
import re


app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Add this route to serve your HTML file
@app.route('/')
def index():
    return app.send_static_file('index.html')

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

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        # Check file size
        file_size = os.path.getsize(audio_path)
        print(f"Audio file size: {file_size / (1024*1024):.2f} MB")
        
        with open(audio_path, 'rb') as audio_file:
            # Try with temperature parameter to get more complete transcription
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",  # Switch back to text format
                language="en",
                temperature=0,  # Use deterministic output
                prompt="This is a public speaking practice video. Please transcribe everything including all filler words and incomplete sentences. Make sure you are transcribing till the end of the video as there might be long silences in between"
            )
        
        full_text = transcript
        print(f"Transcription length: {len(full_text)} characters")
        print(f"Word count: {len(full_text.split())}")
        print(f"First 100 chars: {full_text[:100] if len(full_text) > 100 else full_text}")
        print(f"Last 100 chars: {full_text[-100:] if len(full_text) > 100 else full_text}")
        return full_text
    except Exception as e:
        print(f"Transcription error: {e}")
        raise

def count_filler_words_detailed(text):
    """Count filler words in transcript with detailed breakdown"""
    text_lower = text.lower()
    # Use word boundaries to avoid partial matches
    words = re.findall(r'\b\w+\b', text_lower)
    
    filler_counts = {}
    total_count = 0
    
    for filler in FILLER_WORDS:
        # Handle multi-word fillers like "you know"
        if ' ' in filler:
            # Count occurrences of the phrase
            count = text_lower.count(filler)
        else:
            # Count individual word occurrences
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
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as video_temp:
            video_file.save(video_temp.name)
            video_path = video_temp.name
        
        audio_path = video_path.replace('.mp4', '.mp3')
        
        # Extract audio and get duration
        print("Extracting audio from video...")
        duration = extract_audio_ffmpeg(video_path, audio_path)
        
        # Get audio duration separately to verify
        audio_duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
        audio_duration = float(subprocess.check_output(audio_duration_cmd).decode().strip())
        print(f"Extracted audio duration: {audio_duration} seconds")
        
        # Transcribe using Whisper
        print("Transcribing with Whisper API...")
        transcript = transcribe_audio(audio_path)
        print(f"Transcription complete: {len(transcript)} characters")
        
        # Calculate metrics
        words = transcript.split()
        word_count = len(words)
        words_per_min = int((word_count / duration) * 60) if duration > 0 else 0
        
        print(f"Word count: {word_count}")
        print(f"Words per minute: {words_per_min}")
        
        # Get detailed filler word analysis
        filler_count, filler_breakdown = count_filler_words_detailed(transcript)
        
        # Calculate filler rate per minute
        duration_minutes = duration / 60 if duration > 0 else 1
        filler_rate_per_min = round(filler_count / duration_minutes, 2)
        
        pace_assessment = assess_pace(words_per_min)
        
        # DON'T clean up temp files yet - keep for debugging
        print(f"Temp files kept at: {video_path} and {audio_path}")
        print("IMPORTANT: Delete these files manually after debugging!")
        
        return jsonify({
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
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)