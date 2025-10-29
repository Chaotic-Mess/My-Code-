/* script.js — for the GitHub Pages site.
   It produces a bookmarklet that runs the scrape-and-zip function on the Brightspace page.
*/

const bookmarkletSource = `
(async function(){
  // Avoid double run
  if(window.__brightDownloaderRunning) return alert('Brightspace downloader already running on this page.');
  window.__brightDownloaderRunning = true;

  // lazy-load JSZip
  function loadScript(url){return new Promise((res,rej)=>{
    const s=document.createElement('script'); s.src=url; s.onload=res; s.onerror=rej; document.head.appendChild(s);
  })}

  try{
    // load JSZip and FileSaver (for better save)
    if(typeof JSZip === 'undefined') await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js');
    if(typeof saveAs === 'undefined') await loadScript('https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js');
  }catch(e){
    console.error(e); alert('Failed to load JSZip or FileSaver libraries.');
    window.__brightDownloaderRunning = false;
    return;
  }

  // helper: find candidate links (PDFs, docs, common file endpoints)
  function gatherLinks(){
    const anchors = Array.from(document.querySelectorAll('a[href]'));
    const candidates = [];
    const seen = new Set();
    const extRegex = /\\.(pdf|docx?|pptx?|xlsx?|zip|txt|csv)(?:[?#].*)?$/i;
    for(const a of anchors){
      const href = a.href;
      if(!href) continue;
      // common Brightspace file endpoints often include '/d2l/common/viewFile.d2l' or '/d2l/le/content' etc.
      if(extRegex.test(href) || /\\/d2l\\/common\\/viewFile\\.d2l|\\/d2l\\/le\\/content\\//i.test(href) ){
        if(!seen.has(href)){
          candidates.push({href, text: (a.innerText||a.title||a.getAttribute('aria-label')||href).trim()});
          seen.add(href);
        }
      }
    }
    return candidates;
  }

  const links = gatherLinks();
  if(links.length===0){
    alert('No downloadable file links found on this page. Try opening the course Content page and run again.');
    window.__brightDownloaderRunning = false;
    return;
  }

  // show a small UI overlay to confirm
  function makeOverlay(){
    const overlay = document.createElement('div');
    overlay.style = 'position:fixed;left:14px;bottom:14px;z-index:2147483647;background:#fff;padding:12px;border-radius:10px;box-shadow:0 8px 30px rgba(0,0,0,0.15);max-width:520px;font-family:system-ui,Arial';
    overlay.innerHTML = \`
      <div style="font-weight:600;margin-bottom:6px">Brightspace — download files</div>
      <div style="font-size:13px;margin-bottom:8px">Found <b>\${links.length}</b> files. Select and click "Create ZIP". Fetches using your current session (must be logged in).</div>
      <div id="bd-list" style="max-height:180px;overflow:auto;border:1px solid #eef2ff;padding:6px;border-radius:6px;margin-bottom:8px"></div>
      <div style="display:flex;gap:8px;justify-content:flex-end">
        <button id="bd-cancel" style="padding:6px 10px;border-radius:6px;border:1px solid #ddd;background:#fafafa">Cancel</button>
        <button id="bd-zip" style="padding:6px 10px;border-radius:6px;background:#0b67ff;color:white;border:0">Create ZIP</button>
      </div>
    \`;
    document.body.appendChild(overlay);
    return overlay;
  }

  const overlay = makeOverlay();
  const list = overlay.querySelector('#bd-list');

  links.forEach((l, idx)=>{
    const row = document.createElement('div');
    row.style = 'display:flex;align-items:center;gap:8px;padding:6px;border-bottom:1px dashed #f0f6ff';
    row.innerHTML = '<input type="checkbox" checked data-idx="'+idx+'"><div style="font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;width:360px">'+(l.text||l.href)+'</div><div style="font-size:12px;color:#6b7280">('+ (new URL(l.href)).pathname.split('/').pop() +')</div>';
    list.appendChild(row);
  });

  overlay.querySelector('#bd-cancel').onclick = ()=>{
    overlay.remove();
    window.__brightDownloaderRunning = false;
  };

  overlay.querySelector('#bd-zip').onclick = async ()=>{
    const checkedIdx = Array.from(list.querySelectorAll('input[type=checkbox]:checked')).map(cb=>Number(cb.dataset.idx));
    if(checkedIdx.length===0){ alert('No files selected'); return; }
    try{
      const zip = new JSZip();
      const status = document.createElement('div'); status.style='margin-top:8px;font-size:13px';
      overlay.appendChild(status);
      for(const i of checkedIdx){
        const file = links[i];
        status.textContent = 'Fetching: ' + (file.href.split('/').pop() || file.href);
        // fetch as same-origin (should carry cookies because script runs on domain)
        try{
          const resp = await fetch(file.href, {credentials:'same-origin'});
          if(!resp.ok) { console.warn('Failed fetch', file.href, resp.status); continue; }
          const blob = await resp.blob();
          // generate safe filename
          let name = decodeURIComponent((new URL(file.href)).pathname.split('/').pop() || 'file_'+i);
          // fallback: try to get filename from content-disposition
          const cd = resp.headers.get('content-disposition');
          if(cd){
            const m = cd.match(/filename\\*=UTF-8''([^;\\n]+)/i) || cd.match(/filename="?([^";\\n]+)"?/i);
            if(m && m[1]) name = decodeURIComponent(m[1]);
          }
          zip.file(name, blob);
        }catch(err){
          console.error('Error fetching', file.href, err);
        }
      }
      status.textContent = 'Generating ZIP…';
      const zipBlob = await zip.generateAsync({type:'blob'}, meta => {
        status.textContent = 'Zipping: ' + Math.round(meta.percent) + '%';
      });
      status.textContent = 'Done — preparing download';
      saveAs(zipBlob, 'brightspace-files.zip');
      setTimeout(()=>{overlay.remove(); window.__brightDownloaderRunning = false;}, 500);
    }catch(err){
      console.error(err);
      alert('Error while creating ZIP: ' + err.message);
      window.__brightDownloaderRunning = false;
    }
  };

})();
`.trim();

function makeBookmarklet() {
  // minify a bit and encode
  const min = bookmarkletSource.replace(/\n\s+/g,' ').replace(/\s{2,}/g,' ');
  return 'javascript:' + encodeURIComponent(min);
}

document.addEventListener('DOMContentLoaded', ()=>{
  const copyBtn = document.getElementById('copyBtn');
  const dragBtn = document.getElementById('dragBtn');
  const bm = makeBookmarklet();
  // set drag link href directly to javascript:... (works when dragged to bookmarks bar)
  dragBtn.href = bm;

  copyBtn.addEventListener('click', async ()=>{
    try{
      await navigator.clipboard.writeText(bm);
      copyBtn.textContent = 'Bookmarklet copied!';
      setTimeout(()=>copyBtn.textContent='Copy Bookmarklet',2200);
    }catch(e){
      alert('Could not copy. You can drag the "Drag:" link to your bookmarks bar, or copy the code manually from the page source.');
    }
  });
});
