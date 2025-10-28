function parseGithubLink(link) {
  const m = link.match(/github\.com\/([^\/]+)\/([^\/]+)\/tree\/([^\/]+)\/(.+)/);
  if (!m) return null;
  return {
    owner: m[1],
    repo: m[2],
    branch: m[3],
    path: m[4]
  };
}

async function fetchDirectoryContents(owner, repo, path, branch, token) {
  let files = [];
  const apiUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;
  const headers = token ? { Authorization: `token ${token}` } : {};
  const resp = await fetch(apiUrl, { headers });
  if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status}`);
  const items = await resp.json();
  for (const item of items) {
    if (item.type === 'file') {
      files.push({ path: item.path, size: item.size, download_url: item.download_url });
    } else if (item.type === 'dir') {
      const subFiles = await fetchDirectoryContents(owner, repo, item.path, branch, token);
      files = files.concat(subFiles);
    }
  }
  return files;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(2) + ' MB';
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB';
}

async function downloadAsZip(files, token, zipName) {
  const status = document.getElementById('status-area');
  status.className = 'show loading';
  status.innerHTML = '<span class="loader"></span>Preparing zip...';

  if (files.length === 0) {
    status.className = 'show error';
    status.textContent = 'No files to download.';
    return;
  }

  if (!window.JSZip) {
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
    document.body.appendChild(script);
    await new Promise(r => { script.onload = r; });
  }

  const zip = new JSZip();
  for (let i = 0; i < files.length; i++) {
    status.innerHTML = `<span class="loader"></span>Fetching file ${i + 1}/${files.length}: ${files[i].path}`;
    try {
      const headers = token ? { Authorization: `token ${token}` } : {};
      const fileResp = await fetch(files[i].download_url, { headers });
      const blob = await fileResp.blob();
      zip.file(files[i].path, blob);
    } catch (err) {
      status.className = 'show error';
      status.textContent = `Error downloading file: ${files[i].path}`;
      return;
    }
  }

  status.innerHTML = '<span class="loader"></span>Generating zip file...';
  const zipBlob = await zip.generateAsync({ type: "blob" });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(zipBlob);
  a.download = zipName + '.zip';
  a.click();

  status.className = 'show success';
  status.textContent = 'Download complete';
}

document.getElementById('fetch-info').onclick = async function () {
  const link = document.getElementById('github-link').value.trim();
  const token = document.getElementById('github-token').value.trim() || undefined;
  const info = parseGithubLink(link);
  const infoArea = document.getElementById('info-area');
  const confirmArea = document.getElementById('confirm-area');
  const status = document.getElementById('status-area');

  infoArea.className = '';
  infoArea.innerHTML = '';
  confirmArea.className = '';
  status.className = '';

  if (!info) {
    status.className = 'show error';
    status.textContent = 'Invalid GitHub directory link';
    return;
  }

  status.className = 'show loading';
  status.innerHTML = '<span class="loader"></span>Fetching file list...';

  let files;
  try {
    files = await fetchDirectoryContents(info.owner, info.repo, info.path, info.branch, token);
  } catch (err) {
    status.className = 'show error';
    status.textContent = 'Error fetching files: ' + err.message;
    return;
  }

  const totalSize = files.reduce((a, b) => a + b.size, 0);
  infoArea.className = 'show';
  infoArea.innerHTML = `
        <div class="info-item">
          <span class="info-label">Directory</span>
          <span class="info-value">${info.path}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Files Found</span>
          <span class="info-value">${files.length}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Total Size</span>
          <span class="info-value">${formatSize(totalSize)}</span>
        </div>
      `;

  confirmArea.className = 'show';
  status.className = '';
  confirmArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  confirmArea.querySelector('#download-zip').onclick = () => {
    downloadAsZip(files, token, info.path.replace(/[^a-zA-Z0-9-_]/g, '_') || info.repo);
  };

  confirmArea.querySelector('#cancel-download').onclick = () => {
    confirmArea.className = '';
    infoArea.className = '';
    status.className = 'show error';
    status.textContent = 'Download canceled';
    setTimeout(() => {
      status.className = '';
    }, 2000);
  };
};