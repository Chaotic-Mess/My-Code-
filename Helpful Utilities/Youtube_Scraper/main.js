// ------------------ CONFIG ------------------
const WORKER_URL = "https://youtube-cors-proxy.zacmatthiass.workers.dev";
// --------------------------------------------

function log(msg) {
  const area = document.getElementById("console-area");
  const p = document.createElement("p");
  p.textContent = msg;
  area.appendChild(p);
  area.scrollTop = area.scrollHeight;
}

// Extract ID from YouTube link
function extractVideoId(url) {
  const m = url.match(/(?:v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/);
  return m ? m[1] : null;
}

// Basic metadata
async function fetchOembed(videoId) {
  const res = await fetch(`https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`);
  if (!res.ok) throw new Error("Could not fetch video metadata.");
  return res.json();
}

// Try ytInitialPlayerResponse from raw HTML
async function tryParseHTML(videoId) {
  const htmlRes = await fetch(`${WORKER_URL}/?url=${encodeURIComponent(`https://www.youtube.com/watch?v=${videoId}`)}`);
  const html = await htmlRes.text();
  const m = html.match(/ytInitialPlayerResponse\s*=\s*(\{.*?\});/s);
  if (!m) return [];
  const data = JSON.parse(m[1]);
  const arr = [
    ...(data.streamingData?.formats || []),
    ...(data.streamingData?.adaptiveFormats || [])
  ].filter(f => f.url);
  return arr.map(f => ({
    url: f.url,
    quality: f.qualityLabel || f.audioQuality || "unknown",
    ext: f.mimeType?.split(";")[0] || "unknown"
  }));
}

// Try yt-dlp API via Worker
async function tryYtDlp(videoId) {
  const api = `${WORKER_URL}/api/info?url=https://www.youtube.com/watch?v=${videoId}`;
  const res = await fetch(api);
  if (!res.ok) throw new Error("yt-dlp API failed");
  const data = await res.json();
  const arr = data.formats || [];
  return arr.filter(f => f.url).map(f => ({
    url: f.url,
    quality: f.format_note || f.abr || "unknown",
    ext: f.ext
  }));
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
  const info = await fetchOembed(id);

  infoArea.innerHTML = `
    <h2>${info.title}</h2>
    <p>by ${info.author_name}</p>
    <img src="${info.thumbnail_url}" width="100%" style="border-radius:8px">
  `;

  let streams = [];
  try {
    log("Attempting direct YouTube parse…");
    streams = await tryParseHTML(id);
    if (streams.length === 0) {
      log("Falling back to yt-dlp API…");
      streams = await tryYtDlp(id);
    }
  } catch (e) {
    log("Primary failed, using yt-dlp API fallback.");
    streams = await tryYtDlp(id);
  }

  if (!streams.length) {
    streamsArea.innerHTML = "<p>No downloadable streams found.</p>";
    log("No streams detected.");
    return;
  }

  streams.forEach(s => {
    const div = document.createElement("div");
    div.className = "stream-item";
    div.innerHTML = `
      <span>${s.ext.toUpperCase()} — ${s.quality}</span>
      <div>
        <a href="${s.url}" target="_blank">Open</a>
        <button onclick="navigator.clipboard.writeText('${s.url}').then(()=>alert('Copied!'))">Copy</button>
      </div>
    `;
    streamsArea.appendChild(div);
  });

  streamsArea.classList.remove("hidden");
  log(`✅ Found ${streams.length} streams.`);
};
