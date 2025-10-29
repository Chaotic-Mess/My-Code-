(async () => {
  if (window.__yt_helper_open) return alert("🎥 YouTube Scraper is already active!");
  window.__yt_helper_open = true;

  const safe = s => (s || "").replace(/[<>:"/\\|?*]+/g, "_");
  const copy = text => navigator.clipboard?.writeText(text).then(() => alert("✅ Copied to clipboard")).catch(() => alert("❌ Failed to copy."));

  let info = {};
  try {
    info.title = document.querySelector('meta[name="title"]')?.content ||
                 document.title.replace(" - YouTube", "");
    info.url = location.href;
    info.channel = document.querySelector('ytd-channel-name a')?.innerText || "Unknown";
    info.thumbnail = `https://img.youtube.com/vi/${location.href.match(/v=([^&]+)/)?.[1]}/maxresdefault.jpg`;
    info.description = document.querySelector('#description')?.innerText?.trim() ||
                       document.querySelector('meta[name="description"]')?.content || '';
    info.views = document.querySelector('meta[itemprop="interactionCount"]')?.content ||
                 document.querySelector('.view-count')?.innerText || 'Unknown';
    info.likes = document.querySelector('ytd-segmented-like-dislike-button-renderer')?.innerText?.split('\n')[0] || 'N/A';
  } catch (err) {
    console.warn("Metadata parsing failed", err);
  }

  // Try to extract available streams
  let streams = [];
  try {
    const ytData = window.ytInitialPlayerResponse || window.ytplayer?.config?.args?.player_response && JSON.parse(window.ytplayer.config.args.player_response);
    const formats = ytData?.streamingData?.formats || [];
    const adaptive = ytData?.streamingData?.adaptiveFormats || [];
    streams = [...formats, ...adaptive].map(f => ({
      quality: f.qualityLabel || f.audioQuality || "unknown",
      mime: f.mimeType?.split(";")[0],
      url: f.url
    })).filter(f => f.url);
  } catch (err) {
    console.warn("Stream extraction failed", err);
  }

  // Create overlay UI
  const box = document.createElement("div");
  box.id = "yt-helper";
  box.innerHTML = `
    <h3>🎥 YouTube Scraper</h3>
    <p><b>Title:</b> ${info.title} <button class="copy" data-copy="${info.title}">📋</button></p>
    <p><b>Channel:</b> ${info.channel} <button class="copy" data-copy="${info.channel}">📋</button></p>
    <p><b>Views:</b> ${info.views} | 👍 ${info.likes}</p>
    <p><b>Link:</b> <a href="${info.url}" target="_blank">${info.url}</a> <button class="copy" data-copy="${info.url}">📋</button></p>
    <p><b>Description:</b></p>
    <textarea rows="4">${info.description.slice(0, 400)}</textarea>
    <p><b>Download As:</b></p>
    <select id="yt-helper-format">
      ${streams.length ? streams.map(s => `<option value="${s.url}">${s.mime} – ${s.quality}</option>`).join("") : `
      <option value="thumb">Thumbnail (.jpg)</option>
      <option value="json">Metadata (.json)</option>
      <option value="webloc">Shortcut (.webloc)</option>`}
    </select>
    <button id="yt-helper-download">⬇️ Download</button>
    <button id="yt-helper-close" style="background:#6e7681;margin-left:6px">✖ Close</button>
  `;
  document.body.appendChild(box);

  // Copy buttons
  box.querySelectorAll(".copy").forEach(btn => btn.onclick = () => copy(btn.dataset.copy));

  // Close
  box.querySelector("#yt-helper-close").onclick = () => {
    box.remove();
    window.__yt_helper_open = false;
  };

  // Download handler
  box.querySelector("#yt-helper-download").onclick = async () => {
    const val = box.querySelector("#yt-helper-format").value;

    if (val.startsWith("http")) {
      window.open(val, "_blank");
      return;
    }
    if (val === "thumb") return window.open(info.thumbnail, "_blank");

    if (val === "json") {
      const blob = new Blob([JSON.stringify(info, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = safe(info.title) + ".json";
      a.click();
      return;
    }

    if (val === "webloc") {
      const blob = new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict><key>URL</key><string>${info.url}</string></dict></plist>`], {type:"application/xml"});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = safe(info.title) + ".webloc";
      a.click();
      return;
    }

    alert("⚠️ Could not process this format — try another option or open the stream link manually.");
  };
})();
