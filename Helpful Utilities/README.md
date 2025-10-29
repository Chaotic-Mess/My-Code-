# Helpful Utilities

## GitHub Directory Downloader

A web tool to download all files (including subdirectories) from a GitHub directory link.

**[Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Directory_Downloader)**

### Usage

1. Enter a GitHub directory link (e.g. `https://github.com/owner/repo/tree/main/some/path`)
2. (Optional) Enter a GitHub token if you want to access private repos.
3. Click "Fetch Info".
4. The tool will show the total number of files and their combined size.
5. Confirm download to get all files as a `.zip`.

### Features

- Recursively downloads files in subdirectories.
- Shows file count and total size before downloading.
- Downloads files as a zip archive.
- Works with public and private repositories (for private access, provide a token).

### How it works

This page runs entirely in your browser. It uses the GitHub API to get file information and fetches files using JSZip. No weird virus shit.


---

## Brightspace Scraper

A one-click **bookmarklet-based downloader** for Brightspace course content.  
Lets you grab all PDFs (and other files) from a course page as a single `.zip` — directly in your browser, no installs, no extensions.

**[Use it here](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Brightspace_Scraper)**

### Usage

1. Open the page linked above.  
2. Click **"Copy Bookmarklet"** or drag the **"Download Brightspace Files"** button to your bookmarks bar.  
3. Log in to your Brightspace course (e.g. `https://bright.uvic.ca/d2l/home/426051`).  
4. While on the course’s **Content** page, click your new bookmarklet.  
5. A small overlay appears showing all detected downloadable files (PDF, DOCX, PPTX, etc.).  
6. Click **"Create ZIP"** — the tool fetches the files and gives you one clean `.zip` download.

### Features

- Detects any downloadable file linked in the page (PDF, DOCX, PPTX, XLSX, ZIP, TXT, CSV).  
- Works entirely client-side (no servers, no data collection).  
- Uses your active Brightspace session for file access — you just have to be logged in.  
- Generates a single `.zip` for all selected files.  
- Compatible with Chrome, Edge, Firefox, and Brave.

### How it works

The scraper runs directly **inside the Brightspace page** using your logged-in session (via a bookmarklet).  
It scans the DOM for file links, fetches them using authenticated requests, and bundles everything using **JSZip** and **FileSaver.js** — all locally in your browser.  
No credentials, cookies, or files ever leave your machine.  

If you prefer, you can also install it as a **Tampermonkey userscript** to add a permanent “Download Files” button to Brightspace.  
Either way, everything happens client-side — quick, safe, and spam-free.
