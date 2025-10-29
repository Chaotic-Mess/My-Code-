async function fetchVideoInfo(link) {
  const idMatch = link.match(/v=([^&]+)/);
  if (!idMatch) throw new Error("Invalid YouTube link");
  const videoId = idMatch[1];

  const infoUrl = `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`;

  const info = await fetch(infoUrl).then(r => {
    if (!r.ok) throw new Error("Video not found");
    return r.json();
  });

  return { videoId, ...info };
}

async function extractStreams(videoId) {
  const html = await fetch(`https://cors.isomorphic-git.org/https://www.youtube.com/watch?v=${videoId}`).then(r => r.text());
  const match = html.match(/ytInitialPlayerResponse\s*=\s*(\{.*?\});/);
  if (!match) throw new Error("Failed to parse YouTube data");
  const playerData = JSON.parse(match[1]);
  const streams = [...(playerData.streamingData?.formats || []), ...(playerData.streamingData?.adaptiveFormats || [])]
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
          <a href="${s.url}" target="_blank">Download</a>
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
