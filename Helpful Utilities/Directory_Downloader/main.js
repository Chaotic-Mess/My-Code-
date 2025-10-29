// Helper: parse GitHub directory link
function parseGithubLink(link) {
  // Example: https://github.com/owner/repo/tree/main/path/to/subdir
  const m = link.match(/github\.com\/([^\/]+)\/([^\/]+)\/tree\/([^\/]+)\/(.+)/);
  if (!m) return null;
  return {
    owner: m[1],
    repo: m[2],
    branch: m[3],
    path: m[4]
  };
}

// GitHub API fetch: recursively get all files in directory
async function fetchDirectoryContents(owner, repo, path, branch, token) {
  let files = [];
  const apiUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;
  const headers = token ? {Authorization: `token ${token}`} : {};
  const resp = await fetch(apiUrl, {headers});
  if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status}`);
  const items = await resp.json();
  for (const item of items) {
    if (item.type === 'file') {
      files.push({path: item.path, size: item.size, download_url: item.download_url});
    } else if (item.type === 'dir') {
      const subFiles = await fetchDirectoryContents(owner, repo, item.path, branch, token);
      files = files.concat(subFiles);
    }
  }
  return files;
}

// Format size
function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
  if (bytes < 1024*1024*1024) return (bytes/1024/1024).toFixed(2) + ' MB';
  return (bytes/1024/1024/1024).toFixed(2) + ' GB';
}

// Download files as zip (uses JSZip)
async function downloadAsZip(files, token, zipName) {
  const status = document.getElementById('status-area');
  status.textContent = 'Preparing zip...';
  if (files.length === 0) {
    status.textContent = 'No files to download.';
    return;
  }
  // Dynamically load JSZip
  if (!window.JSZip) {
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
    document.body.appendChild(script);
    await new Promise(r => {script.onload = r;});
  }
  const zip = new JSZip();
  for (let i=0; i<files.length; i++) {
    status.textContent = `Fetching file ${i+1}/${files.length}: ${files[i].path}`;
    try {
      const headers = token ? {Authorization: `token ${token}`} : {};
      const fileResp = await fetch(files[i].download_url, {headers});
      const blob = await fileResp.blob();
      zip.file(files[i].path, blob);
    } catch (err) {
      status.textContent = `Error downloading file: ${files[i].path}`;
      return;
    }
  }
  status.textContent = 'Generating zip file...';
  const zipBlob = await zip.generateAsync({type:"blob"});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(zipBlob);
  a.download = zipName + '.zip';
  a.click();
  status.textContent = 'Download complete!';
}

// UI logic
document.getElementById('fetch-info').onclick = async function() {
  const link = document.getElementById('github-link').value.trim();
  const token = document.getElementById('github-token').value.trim() || undefined;
  const info = parseGithubLink(link);
  const infoArea = document.getElementById('info-area');
  const confirmArea = document.getElementById('confirm-area');
  const status = document.getElementById('status-area');
  infoArea.textContent = '';
  confirmArea.style.display = 'none';
  status.textContent = '';
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
  const totalSize = files.reduce((a,b) => a+b.size, 0);
  infoArea.innerHTML = `
    <b>Directory:</b> ${info.path}<br>
    <b>Files found:</b> ${files.length}<br>
    <b>Total size:</b> ${formatSize(totalSize)}<br>
    <b>Are you sure you want to download?</b>
  `;
  confirmArea.style.display = 'block';
  confirmArea.scrollIntoView({behavior:'smooth'});
  confirmArea.querySelector('#download-zip').onclick = () => {
    downloadAsZip(files, token, info.path.replace(/[^a-zA-Z0-9-_]/g,'_') || info.repo);
  };
  confirmArea.querySelector('#cancel-download').onclick = () => {
    confirmArea.style.display = 'none';
    status.textContent = 'Download canceled.';
  };
}
