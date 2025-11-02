// --- Parse GitHub link safely using the URL API ---
function parseGithubLink(link) {
  try {
    const url = new URL(link);
    const parts = url.pathname.split("/").filter(Boolean);
    // Expect: [owner, repo, "tree", branch, ...pathParts]
    if (parts.length < 4 || parts[2] !== "tree") return null;
    const owner = parts[0];
    const repo = parts[1];
    const branch = parts[3];
    const path = parts.slice(4).join("/");
    return { owner, repo, branch, path };
  } catch {
    return null;
  }
}

// --- Utility: log messages to UI console ---
function logToConsole(msg, type = "log") {
  const consoleArea = document.getElementById("console-area");
  const line = document.createElement("p");
  line.textContent = msg;
  if (type === "error") line.style.color = "#ff5555";
  consoleArea.appendChild(line);
  consoleArea.scrollTop = consoleArea.scrollHeight;
}

// --- Fetch directory contents recursively ---
async function fetchDirectoryContents(owner, repo, path, branch, token, depth = 0) {
  const apiUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;
  const headers = token ? { Authorization: `token ${token}` } : {};
  const indent = " ".repeat(depth * 2);

  // Retry logic for transient errors
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const resp = await fetch(apiUrl, { headers });
      if (!resp.ok) {
        if (resp.status === 404) throw new Error(`404 Not Found`);
        if (resp.status === 403) throw new Error(`403 Forbidden (rate limit or private repo)`);
        throw new Error(`HTTP ${resp.status}`);
      }

      const items = await resp.json();
      let files = [];
      for (const item of items) {
        if (item.type === "file") {
          files.push({ path: item.path, download_url: item.download_url });
        } else if (item.type === "dir") {
          logToConsole(`${indent}📁 Entering ${item.path}`);
          try {
            const sub = await fetchDirectoryContents(owner, repo, item.path, branch, token, depth + 1);
            files = files.concat(sub);
          } catch (err) {
            logToConsole(`${indent}⚠️ Skipped ${item.path}: ${err.message}`, "error");
          }
        }
      }
      return files;
    } catch (err) {
      if (attempt < 3) {
        logToConsole(`${indent}Retrying ${path} (attempt ${attempt})...`);
        await new Promise(r => setTimeout(r, 1000 * attempt));
      } else {
        throw err;
      }
    }
  }
}

// --- Download all files as ZIP ---
async function downloadAsZip(files, token, zipName) {
  const consoleArea = document.getElementById("console-area");
  consoleArea.innerHTML = "";
  logToConsole(`Downloading ${files.length} files...`);

  // Load JSZip if needed
  if (!window.JSZip) {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
    document.body.appendChild(script);
    await new Promise(r => (script.onload = r));
  }

  const zip = new JSZip();
  const headers = token ? { Authorization: `token ${token}` } : {};
  const concurrency = 6;
  let index = 0;

  async function worker() {
    while (index < files.length) {
      const i = index++;
      const f = files[i];
      try {
        logToConsole(`> ${f.path}`);
        const resp = await fetch(f.download_url, { headers });
        if (!resp.ok) throw new Error(resp.statusText);
        const blob = await resp.blob();
        zip.file(f.path, blob);
      } catch (e) {
        logToConsole(`! Error downloading ${f.path}: ${e.message}`, "error");
      }
    }
  }

  await Promise.all(Array(concurrency).fill(0).map(worker));
  logToConsole(`Generating ZIP...`);

  const blob = await zip.generateAsync({ type: "blob" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = zipName + ".zip";
  a.click();

  logToConsole(`✅ Download complete!`);
}

// --- UI setup ---
document.getElementById("fetch-info").onclick = async function () {
  const link = document.getElementById("github-link").value.trim();
  const token = document.getElementById("github-token").value.trim() || undefined;
  const infoArea = document.getElementById("info-area");
  const confirmArea = document.getElementById("confirm-area");
  const consoleArea = document.getElementById("console-area");
  infoArea.textContent = "";
  confirmArea.style.display = "none";
  consoleArea.innerHTML = "";

  const info = parseGithubLink(link);
  if (!info) {
    infoArea.textContent = "❌ Invalid GitHub directory link format.";
    return;
  }

  const apiTestUrl = `https://api.github.com/repos/${info.owner}/${info.repo}/contents/${info.path}?ref=${info.branch}`;
  logToConsole(`🔗 Testing API endpoint: ${apiTestUrl}`);

  infoArea.textContent = "Fetching file list...";
  let files = [];
  try {
    files = await fetchDirectoryContents(info.owner, info.repo, info.path, info.branch, token);
  } catch (err) {
    infoArea.textContent = `Error fetching files: ${err.message}`;
    logToConsole(`Error fetching files: ${err.message}`, "error");
    return;
  }

  if (!files.length) {
    infoArea.innerHTML = `<b>Directory:</b> ${info.path}<br><b>No files found.</b>`;
    return;
  }

  infoArea.innerHTML = `
    <b>Directory:</b> ${info.path}<br>
    <b>Files found:</b> ${files.length}<br>
    <b>Ready to download?</b>
  `;
  confirmArea.style.display = "block";
  confirmArea.scrollIntoView({ behavior: "smooth" });

  confirmArea.querySelector("#download-zip").onclick = () =>
    downloadAsZip(files, token, info.path.replace(/[^a-zA-Z0-9-_]/g, "_"));

  confirmArea.querySelector("#cancel-download").onclick = () => {
    confirmArea.style.display = "none";
    consoleArea.innerHTML = "<p>❌ Download canceled.</p>";
  };
};
