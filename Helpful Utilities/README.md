# 🛠️ Helpful Utilities

> **Your browser-powered toolkit for productivity, downloads, and automation**  
> No installs. No tracking. No BS. Just clean, modern web tools that actually work.

---

## 🎯 Quick Access

| Tool | Description | Link |
|------|-------------|------|
| 🎥 **YouTube Downloader** | Download videos & audio in any quality | [Launch Tool](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Youtube_Downloader) |
| 📦 **Directory Downloader** | Grab entire GitHub directories as ZIP | [Launch Tool](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Directory_Downloader) |
| 📚 **Brightspace Scraper** | Download all course materials at once | [Launch Tool](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Brightspace_Scraper) |

---

## 📦 GitHub Directory Downloader

**Download any GitHub directory as a ZIP file — no cloning required.**

**[🚀 Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Directory_Downloader)**

### ✨ Features

- 🌲 **Recursive downloads** — Grabs all subfolders automatically
- ⚡ **Parallel fetching** — Multiple files at once for speed
- 🔒 **Private repo support** — Use your GitHub token for private access
- 📊 **Size preview** — See file count and total size before downloading
- 🎯 **100% client-side** — Nothing leaves your browser

### 🎮 How to Use

1. Paste a GitHub directory URL (e.g., `https://github.com/owner/repo/tree/main/folder`)
2. (Optional) Add a GitHub token for private repos
3. Click **"Fetch Directory"**
4. Review the file count and size
5. Hit **"Download as ZIP"** — done!

### 🔧 Under the Hood

Pure JavaScript magic using the GitHub REST API and JSZip. Everything runs locally in your browser — no servers, no tracking, no bullshit.

### 💡 Pro Tips

- Works with any branch (main, dev, feature branches, etc.)
- Get a GitHub token [here](https://github.com/settings/tokens) for private repos
- Your token is used only for authentication and never leaves your browser
- Progress logs appear in real-time as files download

---

## 📚 Brightspace Scraper

**One-click bookmarklet to download all your course materials as a single ZIP.**

**[🚀 Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Brightspace_Scraper)**

### ✨ Features

- 📄 **Grabs everything** — PDFs, DOCX, PPTX, XLSX, videos, and more
- 🎯 **One-click operation** — Just add the bookmarklet and click
- 🔐 **Session-based** — Uses your existing login, no credentials needed
- 📦 **Single ZIP output** — All files organized and ready to go
- 🌐 **Cross-browser** — Works on Chrome, Edge, Firefox, and Brave
- 🚫 **Zero tracking** — 100% client-side, nothing leaves your machine

### 🎮 How to Use

1. Visit the tool page and click **"Copy Bookmarklet"**
2. Drag the bookmark to your bookmarks bar
3. Open your Brightspace course **Content** page
4. Click the bookmarklet
5. Select file types you want (PDFs, videos, etc.)
6. Hit **"Create ZIP"** — all files downloaded in seconds!

### 🔧 Under the Hood

Runs as a bookmarklet directly in your Brightspace page. Scans the DOM for downloadable files, uses your authenticated session to fetch them, and bundles everything with JSZip — all locally in your browser. Your credentials and files never touch any external servers.

### 💡 Pro Tips

- Works best on the main Content page of your course
- Can also be used as a Tampermonkey userscript for permanent integration
- Respects your current session — no need to re-login
- If you see mixed content warnings, open the course in a new tab

---

## 🎥 YouTube Video Downloader

**Download YouTube videos and audio in HD, 4K, or as audio-only — fast, free, and private.**

**[🚀 Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Youtube_Downloader)**

### ✨ Features

- 📺 **Multiple quality options** — From 360p to 4K (2160p)
- 🎵 **Audio extraction** — MP3, WAV, Opus, or Ogg formats
- 🎬 **Format choice** — Download as MP4 or WebM
- ⚡ **Lightning fast** — Powered by yt-dlp backend
- 🎯 **Works with everything** — Regular videos, Shorts, age-restricted content
- 🔒 **Privacy-first** — URLs processed by personal backend, nothing logged

### 🎮 How to Use

1. Paste any YouTube video URL
2. Choose your options:
   - **Audio Only** checkbox for music/podcasts
   - **Quality** selector (360p to 4K)
   - **Audio Format** for audio-only downloads
3. Click **"Get Download Link"**
4. Download your file — done!

### 🔧 Under the Hood

Uses a custom PythonAnywhere backend running [yt-dlp](https://github.com/yt-dlp/yt-dlp), the most powerful YouTube downloader available. Your browser sends the URL to the backend, which processes it and returns a direct download link. No data is logged or stored.

### 💡 Pro Tips

- Audio-only mode is perfect for music or podcasts
- If a quality isn't available, you'll get the closest match
- Some videos may be geo-restricted or unavailable
- Download speeds depend on your connection and video size

---

## 🎨 Design Philosophy

All tools share a unified, modern design language:

- **🎨 Color-coded headers** — Red for YouTube, Green for GitHub, Blue for Brightspace
- **🌑 Dark theme** — Easy on the eyes, modern aesthetic
- **📐 Wide layouts** — Maximum content visibility
- **⚡ Fast & responsive** — Optimized for speed and usability
- **🔒 Privacy-first** — Everything runs client-side when possible

---

## 🤝 Contributing

Got ideas? Found a bug? Want to add a new tool?

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

MIT License — do whatever you want with these tools, just keep the credits intact.

---

## 💬 About

Created by **[chaotic-mess](https://github.com/chaotic-mess)** as a collection of practical web utilities that actually solve real problems.

No ads. No paywalls. No data collection. Just tools that work.

**[View on GitHub](https://github.com/chaotic-mess/My-Code-/tree/main/Helpful%20Utilities)** | **[Report Issues](https://github.com/chaotic-mess/My-Code-/issues)**

---

<div align="center">

**Made with ❤️ and way too much caffeine ☕**

</div>

**[Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Youtube_Downloader)**

### Usage

1. Open the page linked above.
2. Paste any YouTube video URL into the input field.
3. Choose your preferred options:
   - **Audio Only** - Extract just the audio track (MP3, WAV, Opus, or Ogg)
   - **Quality** - Select video quality (360p to 4K)
   - **Audio Format** - Choose audio format when downloading audio only
4. Click **"Get Download Link"**.
5. Once processed, click the download button to save your video or audio file.

### Features

- Download videos in **HD, Full HD, 2K, or 4K** quality.
- Extract **audio-only** in MP3, WAV, Opus, or Ogg formats.
- Choose between **MP4 and WebM** video formats.
- Works with standard YouTube videos, **Shorts**, and age-restricted content.
- **Completely free** — no API keys, no subscriptions, no hidden costs.
- **Privacy-focused** — your URLs are processed directly by cobalt.tools API, nothing is logged or tracked.
- Works entirely in your browser using the **cobalt.tools API** (free and open-source).

### How it works

This downloader sends your YouTube URL to [cobalt.tools](https://cobalt.tools), a free and open-source media download API.
The API processes the video and returns a direct download link.

**Privacy:** Your video URL is sent directly from your browser to cobalt's API servers. I don't store, log, or track anything.
cobalt.tools also respects privacy and doesn't retain your data.

The entire interface runs client-side in your browser — no backend servers to maintain, no databases, no tracking pixels.

### Notes

- Some videos may be unavailable due to geographic restrictions, copyright claims, or YouTube's policies.
- If a specific quality isn't available, cobalt will return the closest available option.
- Download speeds depend on your internet connection and the video size.
- This tool is powered by the incredible [cobalt.tools](https://cobalt.tools) service — consider supporting them!

---

## YouTube Scraper

A smart **bookmarklet-based companion** for YouTube that instantly shows video metadata, stream options, and lets you copy or save everything — including thumbnails, transcripts, or shortcuts — directly from any video page.

**[Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Youtube_Scraper)**

### Usage

1. Open the link above.  
2. Drag the **"YouTube Scraper"** button to your bookmarks bar.  
3. Go to any YouTube video page (e.g. `https://www.youtube.com/watch?v=abc123`).  
4. Click the bookmarklet while the video is open.  
5. A floating panel appears showing:
   - Title, Channel, Views, Likes
   - Description (first few lines)
   - Downloadable formats (video/audio)
   - Copy-to-clipboard buttons for title, link, etc.
6. Choose what you want (e.g. video stream, thumbnail, transcript, or metadata) and hit **Download**.

### Features

- Detects available **video/audio streams** (resolution, format, codec).  
- Lets you **copy video info** or URLs directly to your clipboard.  
- Can save:
  - Video/Audio streams (opens direct link)
  - Thumbnail (JPG)
  - Transcript (TXT, if captions exist)
  - Metadata (JSON)
  - Shortcut file (.webloc)
- Built-in **fallbacks** — if a download type isn’t available, the tool automatically offers the next best alternative (e.g., saves a shortcut).  
- Stylish dark overlay that matches the **Helpful Utilities** visual theme.  
- No servers, no API keys, no backend — everything happens client-side in your browser.

### How it works

The YouTube Scraper runs directly inside the current page when you click the bookmarklet.  
It extracts metadata from the player’s internal JSON (`ytInitialPlayerResponse`) and available `streamingData`.  
Detected video and audio URLs are shown in a dropdown, and each can be opened, copied, or saved.

If no streams are available (e.g., YouTube has restricted access), the scraper gracefully falls back to:
- Thumbnail or transcript fetching  
- Metadata export (`.json`)  
- Shortcut file download (`.webloc`)  

Clipboard and download functions are handled entirely in-browser using the [Clipboard API](https://developer.mozilla.org/en-US/docs/Web/API/Clipboard_API) and the [Blob API](https://developer.mozilla.org/en-US/docs/Web/API/Blob).  
Everything runs locally — **no data leaves your machine**.

### Notes

- Works best on regular `youtube.com/watch?v=` URLs (not Shorts or embedded players).  
- The bookmarklet auto-checks that you’re on a valid YouTube page before running.  
- Safe to use: no tracking, cookies, or third-party uploads.

---


