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


async function fetchDirectoryContents(owner, repo, path, branch, token) {
  let files = [];
  const apiUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;
  const headers = token ? {Authorization: `token ${token}`} : {};
  const resp = await fetch(apiUrl, {headers});
  if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status}`);
  const items = await resp.json();
  for (const item of items) {
    if (item.type === 'file') files.push({path: item.path, download_url: item.download_url});
    else if (item.type === 'dir') {
      const sub = await fetchDirectoryContents(owner, repo, item.path, branch, token);
      files = files.concat(sub);
    }
  }
  return files;
}

function logToConsole(msg) {
  const consoleArea = document.getElementById('console-area');
  const line = document.createElement('p');
  line.textContent = msg;
  consoleArea.appendChild(line);
  consoleArea.scrollTop = consoleArea.scrollHeight;
}

async function downloadAsZip(files, token, zipName) {
  const consoleArea = document.getElementById('console-area');
  consoleArea.innerHTML = '';
  logToConsole(`Downloading from [https://github.com/.../${zipName}]`);

  if (!window.JSZip) {
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
    document.body.appendChild(script);
    await new Promise(r => {script.onload = r;});
  }

  const zip = new JSZip();
  const headers = token ? {Authorization: `token ${token}`} : {};

  // Parallel downloads (limit concurrency)
  const concurrency = 6;
  let index = 0;
  async function worker() {
    while (index < files.length) {
      const i = index++;
      const f = files[i];
      logToConsole(`> ${f.path}`);
      try {
        const resp = await fetch(f.download_url, {headers});
        const blob = await resp.blob();
        zip.file(f.path, blob);
      } catch (e) {
        logToConsole(`! Error downloading ${f.path}`);
      }
    }
  }
  await Promise.all(Array(concurrency).fill(0).map(worker));

  logToConsole(`Generating zip file...`);
  const blob = await zip.generateAsync({type:"blob"});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = zipName + ".zip";
  a.click();
  logToConsole(`✅ Download complete!`);
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  consoleArea.appendChild(cursor);
}

document.getElementById('fetch-info').onclick = async function() {
  const link = document.getElementById('github-link').value.trim();
  const token = document.getElementById('github-token').value.trim() || undefined;
  const info = parseGithubLink(link);
  const infoArea = document.getElementById('info-area');
  const confirmArea = document.getElementById('confirm-area');
  const consoleArea = document.getElementById('console-area');
  infoArea.textContent = '';
  confirmArea.style.display = 'none';
  consoleArea.innerHTML = '';

  if (!info) {
    infoArea.textContent = 'Invalid GitHub directory link.';
    return;
  }

  infoArea.textContent = 'Fetching file list...';
  let files;
  try {
    files = await fetchDirectoryContents(info.owner, info.repo, info.path, info.branch, token);
  } catch (err) {
    infoArea.textContent = 'Error fetching files: ' + err;
    return;
  }

  infoArea.innerHTML = `
    <b>Directory:</b> ${info.path}<br>
    <b>Files found:</b> ${files.length}<br>
    <b>Ready to download?</b>
  `;
  confirmArea.style.display = 'block';
  confirmArea.scrollIntoView({behavior:'smooth'});
  confirmArea.querySelector('#download-zip').onclick = () =>
    downloadAsZip(files, token, info.path.replace(/[^a-zA-Z0-9-_]/g,'_'));
  confirmArea.querySelector('#cancel-download').onclick = () => {
    confirmArea.style.display = 'none';
    consoleArea.innerHTML = '<p>❌ Download canceled.</p>';
  };
};
