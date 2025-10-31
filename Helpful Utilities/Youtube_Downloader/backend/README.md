# YouTube Downloader Backend

This is the backend API for the YouTube Video Downloader utility. It runs on PythonAnywhere's free tier and uses yt-dlp to process YouTube videos.

## 🚀 Quick Start

Follow the [DEPLOYMENT.md](./DEPLOYMENT.md) guide for complete setup instructions.

## 📁 Files

- **flask_app.py** - Main Flask application with API endpoints
- **requirements.txt** - Python dependencies
- **DEPLOYMENT.md** - Complete deployment guide for PythonAnywhere

## 🔧 API Endpoints

### `GET /`
Health check - Returns API status

**Response:**
```json
{
  "status": "online",
  "message": "YouTube Downloader API is running"
}
```

### `POST /api/download`
Get download link for a YouTube video

**Request Body:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "audio_only": false,
  "quality": "1080",
  "audio_format": "mp3"
}
```

**Parameters:**
- `url` (required): YouTube video URL
- `audio_only` (optional): Extract audio only (default: false)
- `quality` (optional): Video quality - "max", "2160", "1440", "1080", "720", "480", "360" (default: "best")
- `audio_format` (optional): Audio format - "mp3", "wav", "ogg", "opus" (default: "mp3")

**Response:**
```json
{
  "success": true,
  "download_url": "https://...",
  "audio_only": false,
  "info": {
    "title": "Video Title",
    "duration": "3:45",
    "format": "mp4",
    "thumbnail": "https://..."
  }
}
```

### `GET /api/health`
Health check endpoint

## 🧪 Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python flask_app.py
```

The API will be available at `http://localhost:5000`

Test with curl:
```bash
curl -X POST http://localhost:5000/api/download \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "quality": "720"
  }'
```

## 📦 Dependencies

- **Flask** - Web framework
- **flask-cors** - Enable CORS for frontend
- **yt-dlp** - YouTube video processing

## 🔒 Security & Privacy

- No data is logged or stored
- Video URLs are processed and immediately discarded
- No authentication required (consider adding rate limiting for production)
- CORS enabled for GitHub Pages frontend

## ⚠️ PythonAnywhere Free Tier Limits

- **CPU seconds:** 100 seconds/day
- **Disk space:** 512 MB
- **Web apps:** 1 app
- Processing large videos or high-quality formats uses more CPU time

## 🛠️ Troubleshooting

### yt-dlp errors
Update yt-dlp to the latest version:
```bash
pip install --upgrade yt-dlp
```

### CORS errors
Ensure `flask-cors` is installed and `CORS(app)` is called in `flask_app.py`

### Import errors
Check that the virtualenv path is correctly set in PythonAnywhere's Web tab

## 📝 License

MIT License - Free to use, modify, and distribute
