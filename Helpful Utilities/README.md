# 🛠️ Helpful Utilities

> **Your browser-powered toolkit for productivity, downloads, and automation**  
> No installs. No tracking. No BS. Just clean, modern web tools that actually work.

---

## Quick Access

| Tool | Description | Link |     |
|------|-------------|------|-----|
| **YouTube Downloader** | Download videos & audio in any quality | [Launch Tool](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Youtube_Downloader) | ![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white) | 
| **Directory Downloader** | Grab entire GitHub directories as ZIP | [Launch Tool](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Directory_Downloader) | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white) |
| **Brightspace Scraper** | Download all course materials at once | [Launch Tool](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Brightspace_Scraper) | ![Brightspace](https://img.shields.io/badge/Brightspace-000000?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAIAAAAiOjnJAAAVtUlEQVR4Ae2dzXLbRhKA9QrEE1AvEOkFbL+A5AdYOw+wzF6pilIl7mkp5RArlwi5Lbdq5ZL2tPJah9WeTOoUnWzpFrkipeKsxD+QlPi3ome2AJAQAKIHg/8eEC5VAoIAMT8funt6enqW6ML/m7R/ffj9&logoColor=white)
 |

---

# ***GitHub Directory Downloader***

**Download any GitHub directory as a ZIP file — no cloning required.**

**[Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Directory_Downloader)**

### Features

- **Recursive downloads** — Grabs all subfolders automatically
- **Parallel fetching** — Multiple files at once for speed
- **Private repo support** — Use your GitHub token for private access
- **Size preview** — See file count and total size before downloading
- **100% client-side** — Nothing leaves your browser

### How to Use

1. Paste a GitHub directory URL (e.g., `https://github.com/owner/repo/tree/main/folder`)
2. (Optional) Add a GitHub token for private repos
3. Click **"Fetch Directory"**
4. Review the file count and size
5. Hit **"Download as ZIP"** — done!

### Under the Hood

Pure JavaScript magic using the GitHub REST API and JSZip. Everything runs locally in your browser — no servers, no tracking, no bullshit.

### Pro Tips

- Works with any branch (main, dev, feature branches, etc.)
- Get a GitHub token [here](https://github.com/settings/tokens) for private repos
- Your token is used only for authentication and never leaves your browser
- Progress logs appear in real-time as files download

---

# ***Brightspace Scraper***

**One-click bookmarklet to download all your course materials as a single ZIP.**

**[Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Brightspace_Scraper)**

### Features

- **Grabs everything** — PDFs, DOCX, PPTX, XLSX, videos, and more
- **One-click operation** — Just add the bookmarklet and click
- **Session-based** — Uses your existing login, no credentials needed
- **Single ZIP output** — All files organized and ready to go
- **Cross-browser** — Works on Chrome, Edge, Firefox, and Brave
- **Zero tracking** — 100% client-side, nothing leaves your machine

### How to Use

1. Visit the tool page and click **"Copy Bookmarklet"**
2. Drag the bookmark to your bookmarks bar
3. Open your Brightspace course **Content** page
4. Click the bookmarklet
5. Select file types you want (PDFs, videos, etc.)
6. Hit **"Create ZIP"** — all files downloaded in seconds!

### Under the Hood

Runs as a bookmarklet directly in your Brightspace page. Scans the DOM for downloadable files, uses your authenticated session to fetch them, and bundles everything with JSZip — all locally in your browser. Your credentials and files never touch any external servers.

### Pro Tips

- Works best on the main Content page of your course
- Can also be used as a Tampermonkey userscript for permanent integration
- Respects your current session — no need to re-login
- If you see mixed content warnings, open the course in a new tab

---

# ***YouTube Video Downloader***

**Download YouTube videos and audio in HD, 4K, or as audio-only — fast, free, and private.**

**[Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Youtube_Downloader)**

### Features

- **Multiple quality options** — From 360p to 4K (2160p)
- **Audio extraction** — MP3, WAV, Opus, or Ogg formats
- **Format choice** — Download as MP4 or WebM
- **Lightning fast** — Powered by yt-dlp backend
- **Works with everything** — Regular videos, Shorts, age-restricted content
- **Privacy-first** — URLs processed by personal backend, nothing logged

### How to Use

1. Paste any YouTube video URL
2. Choose your options:
   - **Audio Only** checkbox for music/podcasts
   - **Quality** selector (360p to 4K)
   - **Audio Format** for audio-only downloads
3. Click **"Get Download Link"**
4. Download your file — done!

### Under the Hood

Uses a custom PythonAnywhere backend running [yt-dlp](https://github.com/yt-dlp/yt-dlp), the most powerful YouTube downloader available. Your browser sends the URL to the backend, which processes it and returns a direct download link. No data is logged or stored.

### Pro Tips

- Audio-only mode is perfect for music or podcasts
- If a quality isn't available, you'll get the closest match
- Some videos may be geo-restricted or unavailable
- Download speeds depend on your connection and video size

---

## License

MIT License — do whatever you want with these tools, just keep the credits intact.

---

## About

Created by **[chaotic-mess](https://github.com/chaotic-mess)** as a collection of practical web utilities that actually solve real problems.

No ads. No paywalls. No data collection. Just tools that work.

**[View on GitHub](https://github.com/chaotic-mess/My-Code-/tree/main/Helpful%20Utilities)** | **[Report Issues](https://github.com/chaotic-mess/My-Code-/issues)**

---

<div align="center">

</div>
---


