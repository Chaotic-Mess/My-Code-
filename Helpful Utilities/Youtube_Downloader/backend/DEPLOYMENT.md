# Deploying to PythonAnywhere

## Step-by-Step Deployment Guide

### 1. Create a PythonAnywhere Account
1. Go to [PythonAnywhere.com](https://www.pythonanywhere.com)
2. Sign up for a **free Beginner account** (no credit card required)
3. Verify your email

### 2. Upload Your Code

#### Option A: Using Git (Recommended)
1. Open a **Bash console** in PythonAnywhere dashboard
2. Clone your repository:
   ```bash
   git clone https://github.com/Chaotic-Mess/My-Code-.git
   cd My-Code-/Helpful\ Utilities/Youtube_Downloader/backend/
   ```

#### Option B: Upload Files Manually
1. Go to the **Files** tab in PythonAnywhere
2. Create a new directory: `/home/yourusername/youtube-downloader/`
3. Upload `flask_app.py` and `requirements.txt`

### 3. Create a Virtual Environment
Open a Bash console and run:
```bash
cd ~
python3.10 -m venv myenv
source myenv/bin/activate
```

### 4. Install Dependencies
```bash
cd ~/youtube-downloader/  # or your project directory
pip install -r requirements.txt
```

**Note:** On the free tier, you might need to install packages one at a time if you hit memory limits:
```bash
pip install Flask
pip install flask-cors
pip install yt-dlp
```

### 5. Configure Web App

1. Go to the **Web** tab in PythonAnywhere dashboard
2. Click **"Add a new web app"**
3. Choose **"Manual configuration"** (not Flask wizard)
4. Select **Python 3.10**

### 6. Configure WSGI File

1. In the **Web** tab, click on the WSGI configuration file link
2. Replace the contents with:

```python
import sys
import os

# Add your project directory to the sys.path
project_home = '/home/yourusername/youtube-downloader'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Set up environment variables if needed
os.environ['FLASK_APP'] = 'flask_app'

# Import the Flask app
from flask_app import app as application
```

**Important:** Replace `yourusername` with your actual PythonAnywhere username!

### 7. Configure Virtual Environment Path

1. In the **Web** tab, find the **"Virtualenv"** section
2. Enter the path to your virtual environment:
   ```
   /home/yourusername/myenv
   ```
3. Click the checkmark to save

### 8. Reload Your Web App

1. Scroll to the top of the **Web** tab
2. Click the green **"Reload yourusername.pythonanywhere.com"** button

### 9. Update Frontend

1. Open `main.js` in your frontend code
2. Update the API endpoint:
   ```javascript
   const API_ENDPOINT = 'https://yourusername.pythonanywhere.com/api/download';
   ```
3. Replace `yourusername` with your actual PythonAnywhere username
4. Commit and push to GitHub

### 10. Test Your API

Visit your API in a browser:
```
https://yourusername.pythonanywhere.com/
```

You should see:
```json
{
  "status": "online",
  "message": "YouTube Downloader API is running"
}
```

## Testing the Download Endpoint

Use curl or Postman to test:

```bash
curl -X POST https://yourusername.pythonanywhere.com/api/download \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "audio_only": false,
    "quality": "720"
  }'
```

## Troubleshooting

### Issue: "Import Error" or "Module not found"
- Check that your virtualenv is correctly configured in the Web tab
- Verify all packages are installed: `pip list` in your bash console

### Issue: "504 Gateway Timeout"
- Free tier has CPU second limits (100s/day)
- yt-dlp can be CPU intensive for large videos
- Consider caching results or limiting video length

### Issue: CORS errors in browser
- Make sure `flask-cors` is installed
- Verify CORS is enabled in `flask_app.py`

### Issue: "Permission denied"
- Check file permissions: `ls -la` in bash console
- Make sure files are readable: `chmod 644 flask_app.py`

## Free Tier Limitations

- **CPU seconds:** 100 seconds/day (resets daily)
- **Disk space:** 512 MB
- **Web apps:** 1 app
- **Bandwidth:** Reasonable use (no specific limit)
- **Always-on tasks:** Not available on free tier

## Upgrading (Optional)

If you hit free tier limits:
- **Hacker plan ($5/month):** More CPU, disk space, always-on tasks
- Consider caching frequent requests to reduce processing

## Monitoring

Check your API logs:
1. Go to **Web** tab
2. Click on **Error log** or **Server log** links
3. Useful for debugging issues

## Updating Your Code

When you make changes:

```bash
cd ~/youtube-downloader/
git pull origin main
# Then reload web app from Web tab
```

Or use PythonAnywhere's file editor in the **Files** tab.

## Security Notes

- The free tier domain is public: `yourusername.pythonanywhere.com`
- Consider adding rate limiting if abuse becomes an issue
- Free tier doesn't support HTTPS custom domains
- yt-dlp stays updated; periodically run `pip install --upgrade yt-dlp`

## Support

- [PythonAnywhere Forums](https://www.pythonanywhere.com/forums/)
- [PythonAnywhere Help](https://help.pythonanywhere.com/)
- [yt-dlp Documentation](https://github.com/yt-dlp/yt-dlp)
