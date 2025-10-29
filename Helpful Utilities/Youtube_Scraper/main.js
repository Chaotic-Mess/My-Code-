// ------------------ CONFIG ------------------
const WORKER_URL = "https://youtube-cors-proxy.zacmatthiass.workers.dev/";
// --------------------------------------------

// Utility: extract video ID from a YouTube link
function extractVideoId(url) {
  const match = url.match(/(?:v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/);
  return match ? match[1] : null;
}

// Get basic video info (title, thumbnail, etc.)
async function fetchVideoInfo(url) {
  const id = extractVideoId(url);
  if (!id) throw new Error("Invalid YouTube URL");

  const oembed = `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${id}&format=json`;
  const res = await fetch(oembed);
  if (!res.ok) throw new Error(`Failed to fetch oEmbed data: ${res.status}`);
  const data = await res.json();

  return {
    title: data.title,
    author_name: data.author_name,
    thumbnail_url: data.thumbnail_url,
    videoId: id
  };
}

// Proxy YouTube HTML through Cloudflare Worker or manual paste fallback
async function fetchYouTubeHTML(videoId) {
  const target = `https://www.youtube.com/watch?v=${videoId}`;

  try {
    const url = `${WORKER_URL}/?url=${encodeURIComponent(target)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Worker returned ${res.status}`);
    console.log(`✅ Loaded YouTube HTML via worker`);
    return await res.text();
  } catch (e) {
    console.warn("Proxy failed:", e);
    return manualFallbackPrompt(target);
  }
}

// Manual paste fallback
function manualFallbackPrompt(target) {
  return new Promise((resolve, reject) => {
    const overlay = document.createElement("div");
    overlay.style = `
      position:fixed;top:0;left:0;width:100%;height:100%;
      background:rgba(0,0,0,0.95);z-index:999999;
      display:flex;align-items:center;justify-content:center;
      flex-direction:column;color:#eee;font-family:system-ui;
    `;
    overlay.innerHTML = `
      <div style="max-width:600px;padding:20px;text-align:center">
        <h2>🧩 Manual Fallback</h2>
        <p>Could not reach YouTube via proxy.<br>
        Please open <b>${target}</b>, press <b>Ctrl+U</b> (View Source),
        copy all the text, and paste it below.</p>
        <textarea id="yt_html_paste" style="width:100%;height:200px;background:#111;color:#eee;border:1px solid #333;border-radius:6px"></textarea>
        <br><br>
        <button id="yt_submit_html" style="background:#4ade80;border:0;padding:8px 16px;border-radius:6px;cursor:pointer">Use Pasted HTML</button>
        <button id="yt_cancel_html" style="background:#333;border:0;padding:8px 16px;border-radius:6px;cursor:pointer;margin-left:8px">Cancel</button>
      </div>
    `;
    document.body.appendChild(overlay);
    overlay.querySelector("#yt_submit_html").onclick = () => {
      const val = overlay.querySelector("#yt_html_paste").value.trim();
      if (!val) return alert("Please paste the HTML first.");
      overlay.remove();
      resolve(val);
    };
    overlay.querySelector("#yt_cancel_html").onclick = () => {
      overlay.remove();
      reject(new Error("User canceled manual HTML entry"));
    };
  });
}

// Extract playable streams from HTML
async function extractStreams(videoId) {
  const html = await fetchYouTubeHTML(videoId);
  const match = html.match(/ytInitialPlayerResponse\s*=\s*(\{.*?\});/s);
  if (!match) throw new Error("Failed to parse player data.");
  const data = JSON.parse(match[1]);

  const streams = [
    ...(data.streamingData?.formats || []),
    ...(data.streamingData?.adaptiveFormats || [])
  ]
    .filter(f => f.url)
    .map(f => ({
      mime: f.mimeType.split(";")[0],
      quality: f.qualityLabel || f.audioQuality || "unknown",
      url: f.url
    }));

  return streams;
}

// Log messages to the console area
function logToConsole(msg) {
  const area = document.getElementById("console-area");
  const line = document.createElement("p");
  line.textContent = msg;
  area.appendChild(line);
  area.scrollTop = area.scrollHeight;
}

// MAIN UI HANDLER
document.getElementById("fetch-info").onclick = async () => {
  const link = document.getElementById("yt-link").value.trim();
  const infoArea = document.getElementById("info-area");
  const streamsArea = document.getElementById("streams-area");
  const consoleArea = document.getElementById("console-area");

  streamsArea.innerHTML = "";
  streamsArea.classList.add("hidden");
  consoleArea.innerHTML = "";

  if (!link) return alert("Please enter a YouTube link.");

  infoArea.textContent = "Fetching video info…";

  try {
    const info = await fetchVideoInfo(link);
    infoArea.innerHTML = `
      <h2>${info.title}</h2>
      <img src="${info.thumbnail_url}" width="100%" style="border-radius:8px">
      <p>${info.author_name}</p>
    `;

    const streams = await extractStreams(info.videoId);
    if (!streams.length) {
      streamsArea.innerHTML = "<p>No downloadable streams found.</p>";
    } else {
      streams.forEach(s => {
        const el = document.createElement("div");
        el.className = "stream-item";
        el.innerHTML = `
          <span>${s.mime} – ${s.quality}</span>
          <div style="display:flex;gap:4px">
            <a href="${s.url}" target="_blank">Open</a>
            <button onclick="navigator.clipboard.writeText('${s.url}').then(()=>alert('Copied!'))">Copy</button>
          </div>
        `;
        streamsArea.appendChild(el);
      });
    }

    streamsArea.classList.remove("hidden");
    consoleArea.textContent = `Found ${streams.length} streams.`;
  } catch (e) {
    console.error(e);
    infoArea.textContent = "Error: " + e.message;
  }
};
