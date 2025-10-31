// ------------------ CONFIG ------------------
const WORKER_URL = "https://youtube-cors-proxy.zacmatthiass.workers.dev";
// --------------------------------------------

// Simple logger
function log(msg) {
  const area = document.getElementById("console-area");
  const p = document.createElement("p");
  p.textContent = msg;
  area.appendChild(p);
  area.scrollTop = area.scrollHeight;
}

// Download stream function
async function downloadStream(url, filename, videoTitle) {
  try {
    log(`Starting download: ${filename}`);
    
    // Create a temporary anchor element and trigger download
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.target = "_blank";
    
    // For cross-origin URLs, we need to fetch and create a blob
    try {
      // Try direct download first (works if CORS allows)
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      log(`✅ Download started: ${filename}`);
    } catch (e) {
      // If direct download fails, try fetching as blob
      log("Fetching video data...");
      const response = await fetch(url);
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      
      // Clean up blob URL after a delay
      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
      log(`✅ Download completed: ${filename}`);
    }
  } catch (error) {
    console.error("Download error:", error);
    log(`❌ Download failed: ${error.message}`);
    alert("Download failed. Try copying the URL and opening it in a new tab, then right-click to save.");
  }
}

// Extract video ID from link
function extractVideoId(url) {
  const m = url.match(/(?:v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/);
  return m ? m[1] : null;
}

// Fetch oEmbed metadata (title, thumbnail, etc.)
async function fetchOembed(videoId) {
  const res = await fetch(`https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`);
  if (!res.ok) throw new Error("Failed to fetch oEmbed data.");
  return res.json();
}

// Fetch video info from the unified Worker
async function fetchVideoInfo(videoId) {
  const api = `${WORKER_URL}/?url=https://www.youtube.com/watch?v=${videoId}`;
  const res = await fetch(api);
  if (!res.ok) throw new Error(`Worker returned ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// UI handler
document.getElementById("fetch-info").onclick = async () => {
  const link = document.getElementById("yt-link").value.trim();
  const infoArea = document.getElementById("info-area");
  const streamsArea = document.getElementById("streams-area");
  const consoleArea = document.getElementById("console-area");

  infoArea.innerHTML = "";
  streamsArea.innerHTML = "";
  consoleArea.innerHTML = "";

  const id = extractVideoId(link);
  if (!id) return alert("Invalid YouTube link.");

  log("Fetching metadata…");
  const meta = await fetchOembed(id);
  infoArea.innerHTML = `
    <h2>${meta.title}</h2>
    <p>by ${meta.author_name}</p>
    <img src="${meta.thumbnail_url}" width="100%" style="border-radius:8px">
  `;

  try {
    log("Contacting worker for available streams…");
    const data = await fetchVideoInfo(id);

    const streams = data.formats || [];
    if (!streams.length) {
      streamsArea.innerHTML = "<p>No downloadable streams found.</p>";
      log("No streams detected.");
      return;
    }

    // Populate the UI with available formats
    streams.forEach((f, idx) => {
      const div = document.createElement("div");
      div.className = "stream-item";
      
      const infoSpan = document.createElement("span");
      infoSpan.textContent = `${f.mime.toUpperCase()} — ${f.quality} (${f.size || "?"})`;
      
      const btnGroup = document.createElement("div");
      btnGroup.className = "btn-group";
      
      // Download button
      const downloadBtn = document.createElement("button");
      downloadBtn.textContent = "Download";
      downloadBtn.className = "btn-download";
      downloadBtn.onclick = () => downloadStream(f.url, `${data.title}_${f.quality}_${idx}.${f.ext}`, meta.title);
      
      // Copy URL button
      const copyBtn = document.createElement("button");
      copyBtn.textContent = "Copy URL";
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(f.url).then(() => {
          copyBtn.textContent = "Copied!";
          setTimeout(() => copyBtn.textContent = "Copy URL", 2000);
        });
      };
      
      btnGroup.appendChild(downloadBtn);
      btnGroup.appendChild(copyBtn);
      
      div.appendChild(infoSpan);
      div.appendChild(btnGroup);
      streamsArea.appendChild(div);
    });

    streamsArea.classList.remove("hidden");
    log(`✅ Found ${streams.length} streams.`);
  } catch (e) {
    console.error(e);
    log("❌ Error: " + e.message);
    infoArea.innerHTML += `<p style="color:red">${e.message}</p>`;
  }
};
