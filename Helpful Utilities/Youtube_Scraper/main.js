async function tryFetchYouTubeHTML(videoId) {
  const target = `https://www.youtube.com/watch?v=${videoId}`;
  const urls = [
    `https://youtube-cors-proxy.zacmatthiass.workers.dev/?url=${encodeURIComponent(target)}`,
    `https://api.allorigins.win/raw?url=${encodeURIComponent(target)}`,
    `https://corsproxy.io/?${encodeURIComponent(target)}`,
    `https://thingproxy.freeboard.io/fetch/${target}`
  ];

  for (const u of urls) {
    try {
      const res = await fetch(u);
      if (res.ok) {
        console.log(`Loaded via ${u}`);
        return await res.text();
      }
      console.warn(`Proxy failed (${u}): ${res.status}`);
    } catch (e) {
      console.warn(`Proxy error (${u}): ${e}`);
    }
  }

  // All proxies failed — manual fallback
  return await new Promise((resolve, reject) => {
    const overlay = document.createElement("div");
    overlay.style = `
      position:fixed;top:0;left:0;width:100%;height:100%;
      background:rgba(0,0,0,0.9);z-index:999999;
      display:flex;align-items:center;justify-content:center;
      flex-direction:column;color:#eee;font-family:system-ui;
    `;
    overlay.innerHTML = `
      <div style="max-width:600px;padding:20px;text-align:center">
        <h2>Manual Fallback</h2>
        <p>All proxies were blocked.<br>
        Please open the YouTube video, press <b>Ctrl+U</b> (View Source), 
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

async function extractStreams(videoId) {
  const html = await tryFetchYouTubeHTML(videoId);
  const match = html.match(/ytInitialPlayerResponse\s*=\s*(\{.*?\});/);
  if (!match) throw new Error("Failed to parse YouTube data (ytInitialPlayerResponse not found)");
  const playerData = JSON.parse(match[1]);

  const streams = [
    ...(playerData.streamingData?.formats || []),
    ...(playerData.streamingData?.adaptiveFormats || [])
  ]
    .filter(f => f.url)
    .map(f => ({
      mime: f.mimeType.split(";")[0],
      quality: f.qualityLabel || f.audioQuality || "unknown",
      url: f.url
    }));

  return streams;
}

document.getElementById("fetch-info").onclick = async () => {
  const link = document.getElementById("yt-link").value.trim();
  const infoArea = document.getElementById("info-area");
  const streamsArea = document.getElementById("streams-area");
  const consoleArea = document.getElementById("console-area");

  streamsArea.innerHTML = "";
  streamsArea.classList.add("hidden");
  consoleArea.textContent = "";

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
            <button style="background:#333;border:0;color:#aaa;border-radius:4px;padding:2px 6px;cursor:pointer"
              onclick="navigator.clipboard.writeText('${s.url}').then(()=>alert('Copied!'))">Copy</button>
          </div>
        `;
        streamsArea.appendChild(el);
      });
    }

    streamsArea.classList.remove("hidden");
    consoleArea.textContent = `Found ${streams.length} streams.`;
  } catch (e) {
    infoArea.textContent = "Error: " + e.message;
  }
};
