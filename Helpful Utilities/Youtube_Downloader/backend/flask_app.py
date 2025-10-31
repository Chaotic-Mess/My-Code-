"""
YouTube Downloader Backend API for PythonAnywhere
Uses yt-dlp to fetch YouTube video download links
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for GitHub Pages frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_video_info(url, audio_only=False, quality='best', audio_format='mp3'):
    """
    Extract video information and download URL using yt-dlp
    
    Args:
        url (str): YouTube video URL
        audio_only (bool): Extract audio only
        quality (str): Video quality (max, 2160, 1440, 1080, 720, 480, 360)
        audio_format (str): Audio format for audio-only downloads (mp3, wav, ogg, opus)
    
    Returns:
        dict: Video information including download URL
    """
    try:
        # Configure yt-dlp options
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        if audio_only:
            # Audio-only extraction
            ydl_opts.update({
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': audio_format,
                }]
            })
        else:
            # Video extraction with quality selection
            if quality == 'max' or quality == 'best':
                format_str = 'bestvideo+bestaudio/best'
            else:
                # Select specific quality
                format_str = f'bestvideo[height<={quality}]+bestaudio/best[height<={quality}]'
            
            ydl_opts['format'] = format_str
        
        # Extract video information
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Get the best format URL
            if 'url' in info:
                download_url = info['url']
            elif 'formats' in info and len(info['formats']) > 0:
                # Get the best format
                download_url = info['formats'][-1]['url']
            else:
                download_url = info.get('webpage_url', url)
            
            return {
                'success': True,
                'download_url': download_url,
                'audio_only': audio_only,
                'info': {
                    'title': info.get('title', 'Unknown'),
                    'duration': format_duration(info.get('duration', 0)),
                    'format': info.get('ext', 'mp4'),
                    'thumbnail': info.get('thumbnail', ''),
                }
            }
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            'success': False,
            'error': f"Failed to process video: {str(e)}"
        }


def format_duration(seconds):
    """Format duration in seconds to HH:MM:SS or MM:SS"""
    if not seconds:
        return "Unknown"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'YouTube Downloader API is running',
        'endpoints': {
            '/api/download': 'POST - Get download link for a YouTube video'
        }
    })


@app.route('/api/download', methods=['POST'])
def download():
    """
    API endpoint to get download link for YouTube video
    
    Request body:
    {
        "url": "https://www.youtube.com/watch?v=...",
        "audio_only": false,
        "quality": "1080",
        "audio_format": "mp3"
    }
    
    Response:
    {
        "success": true,
        "download_url": "https://...",
        "info": {
            "title": "Video Title",
            "duration": "3:45",
            "format": "mp4"
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing YouTube URL in request'
            }), 400
        
        url = data['url']
        audio_only = data.get('audio_only', False)
        quality = data.get('quality', 'best')
        audio_format = data.get('audio_format', 'mp3')
        
        # Validate URL
        if 'youtube.com' not in url and 'youtu.be' not in url:
            return jsonify({
                'success': False,
                'error': 'Invalid YouTube URL'
            }), 400
        
        # Get video information and download URL
        result = get_video_info(url, audio_only, quality, audio_format)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    # For local testing only
    # On PythonAnywhere, use their WSGI configuration
    app.run(debug=True, host='0.0.0.0', port=5000)
