(async () => {
  if (window.__yt_helper_open) return alert("YouTube Scraper is already open!");
  window.__yt_helper_open = true;

  const info = {};
  try {
    info.title = document.querySelector('meta[name="title"]')?.content ||
                 document.querySelector('h1.title yt-formatted-string')?.textContent?.trim() ||
                 document.title;
    info.url = location.href;
    info.channel = document.querySelector('#text-container.ytd-channel-name')?.innerText?.trim() ||
                   document.querySelector('link[itemprop="name"]')?.content;
    info.thumbnail = `https://img.youtube.com/vi/${location.href.match(/v=([^&]+)/)?.[1]}/maxresdefault.jpg`;
    info.description = document.querySelector('#description')?.innerText?.trim() ||
                       document.querySelector('meta[name="description"]')?.content || '';
    info.views = document.querySelector('meta[itemprop="interactionCount"]')?.content ||
                 document.querySelector('.view-count')?.innerText;
  } catch (e) { console.error("Metadata parse failed", e); }

  // Create overlay
  const box = document.createElement('div');
  box.id = 'yt-helper';
  box.innerHTML = `
    <h3>YouTube Helper</h3>
    <p><b>Title:</b> ${info.title || 'Unknown'}</p>
    <p><b>Channel:</b> ${info.channel || 'Unknown'}</p>
    <p><b>Views:</b> ${info.views || 'N/A'}</p>
    <p><b>Link:</b> <a href="${info.url}" target="_blank">${info.url}</a></p>
    <p><b>Description:</b></p>
    <textarea rows="4">${info.description.slice(0, 500)}</textarea>
    <p><b>Download As:</b></p>
    <select id="yt-helper-format">
      <option value="mp4">Video (.mp4)</option>
      <option value="mp3">Audio (.mp3)</option>
      <option value="txt">Transcript (.txt)</option>
      <option value="jpg">Thumbnail (.jpg)</option>
      <option value="json">Metadata (.json)</option>
      <option value="webloc">Shortcut (.webloc)</option>
    </select>
    <button id="yt-helper-download">Download</button>
    <button id="yt-helper-close" style="background:#6e7681;margin-left:6px">✖ Close</button>
  `;
  document.body.appendChild(box);

  // Handlers
  box.querySelector("#yt-helper-close").onclick = () => {
    box.remove();
    window.__yt_helper_open = false;
  };

  box.querySelector("#yt-helper-download").onclick = async () => {
    const type = box.querySelector("#yt-helper-format").value;
    const id = info.url.match(/v=([^&]+)/)?.[1];
    if (!id) return alert("Could not extract video ID!");

    switch (type) {
      case "jpg":
        window.open(info.thumbnail, "_blank");
        break;
      case "webloc": {
        const blob = new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict><key>URL</key><string>${info.url}</string></dict></plist>`], {type:"application/xml"});
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${info.title.replace(/[^\w\d-_]/g,'_')}.webloc`;
        a.click();
        break;
      }
      case "json": {
        const blob = new Blob([JSON.stringify(info, null, 2)], {type:"application/json"});
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${info.title.replace(/[^\w\d-_]/g,'_')}.json`;
        a.click();
        break;
      }
      case "txt": {
        // Try to fetch transcript via YouTube captions API
        try {
          const ytInitialData = window.ytInitialPlayerResponse;
          const captionTracks = ytInitialData?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
          if (captionTracks?.length) {
            const url = captionTracks[0].baseUrl;
            const text = await fetch(url).then(r => r.text());
            const blob = new Blob([text], {type:"text/plain"});
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = `${info.title.replace(/[^\w\d-_]/g,'_')}_transcript.txt`;
            a.click();
          } else throw new Error("No transcript found");
        } catch (e) {
          alert("Transcript unavailable — saving shortcut instead.");
          box.querySelector("#yt-helper-format").value = "webloc";
        }
        break;
      }
      default:
        alert(`Direct ${type.toUpperCase()} download not supported client-side.\nA shortcut will be saved instead.`);
        box.querySelector("#yt-helper-format").value = "webloc";
    }
  };
})();
