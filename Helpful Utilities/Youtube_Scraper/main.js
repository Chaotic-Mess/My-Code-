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

// Download stream function - Routes through Worker proxy to bypass IP restrictions
async function downloadStream(url, filename, videoTitle) {
  try {
    log(`Starting download: ${filename}`);
    
    // Construct the proxy URL - this makes the Worker download the video
    const proxyUrl = `${WORKER_URL}/download?download=${encodeURIComponent(url)}&filename=${encodeURIComponent(filename)}`;
    
    log("Downloading through proxy (this may take a moment)...");
    
    // Fetch through the proxy
    const response = await fetch(proxyUrl);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      throw new Error(errorData.error || `Proxy returned ${response.status}`);
    }
    
    // Get content length for progress indication
    const contentLength = response.headers.get('Content-Length');
    if (contentLength) {
      const sizeMB = (parseInt(contentLength, 10) / 1048576).toFixed(2);
      log(`Downloading ${sizeMB} MB...`);
    }
    
    // Convert response to blob
    const blob = await response.blob();
    log(`Processing download...`);
    
    // Create download link
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    // Clean up blob URL
    setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    log(`✅ Download completed: ${filename}`);
  } catch (error) {
    console.error("Download error:", error);
    log(`❌ Download failed: ${error.message}`);
    
    // Offer to copy the direct URL as fallback
    const copyDirect = confirm(
      "Download through proxy failed.\n\n" +
      "Would you like to copy the direct YouTube URL?\n" +
      "(Warning: It expires quickly and may not work from your IP)"
    );
    
    if (copyDirect) {
      navigator.clipboard.writeText(url).then(() => {
        alert("URL copied! Try pasting it in your browser immediately.");
      });
    }
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
  const data = await res.json();
  
  if (!res.ok) {
    console.error("Worker error response:", data);
    throw new Error(data.error || `Worker returned ${res.status}`);
  }
  
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
    
    // Log the full response for debugging
    console.log("Worker response:", data);
    
    if (data.error) {
      log(`❌ Worker error: ${data.error}`);
      streamsArea.innerHTML = `<p style="color:red">Worker error: ${data.error}</p>`;
      return;
    }

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
