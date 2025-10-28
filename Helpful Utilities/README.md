# Helpful Utilities

## GitHub Directory Downloader

A web tool to download all files (including subdirectories) from a GitHub directory link.

**[▶️ Use it here (GitHub Pages Deploy)](https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/)**

## Usage

1. Enter a GitHub directory link (e.g. `https://github.com/owner/repo/tree/main/some/path`)
2. (Optional) Enter a GitHub token if you want to access private repos.
3. Click "Fetch Info".
4. The tool will show the total number of files and their combined size.
5. Confirm download to get all files as a `.zip`.

## Features

- Recursively downloads files in subdirectories.
- Shows file count and total size before downloading.
- Downloads files as a zip archive.
- Works with public and private repositories (for private access, provide a token).

## How it works

This page runs entirely in your browser. It uses the GitHub API to get file information and fetches files using JSZip. No weird virus shit.
