/* ===========================================================
   Brightspace_Scraper v4.0 (Simple & Ruthless)
   by chaotic-mess | https://chaotic-mess.github.io/My-Code-/
   One job: grab real files (PDF/MP4/DOCX/PPTX/XLSX/ZIP/TXT).
   - Sources (priority): ToC API → Lessons DOM → Smart-Curriculum iframe
   - QuickLinks resolved to /content/enforced/... when possible
   - Follow redirects, respect Content-Disposition filenames
   - Optional fallback to open+close real tabs only when fetch fails
   =========================================================== */
(async () => {
  if (window.__bs_scraper_active) { alert("Brightspace Scraper already running."); return; }
  window.__bs_scraper_active = true;

  /* ---------- Utils ---------- */
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const abs = u => { try { return new URL(u, location.href).href; } catch { return null; } };
  const san = s => (s || "").replace(/[<>:"/\\|?*]+/g, "_").trim();
  const extOf = u => { const m = u && u.match(/\.[a-z0-9]{2,5}(?:$|\?)/i); return m ? m[0].toLowerCase() : ""; };
  const looksFile = u =>
    /\.(pdf|mp4|m4v|mov|docx?|pptx?|xlsx?|zip|txt|csv|rtf|md|epub)(?:[?#].*)?$/i.test(u) ||
    /\/content\/enforced\//i.test(u) ||
    /\/d2l\/common\/viewFile\.d2l/i.test(u);

  const nowTag = () => {
    const d = new Date();
    return `(${String(d.getFullYear()).slice(2)}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')})`;
  };

  const withTimeout = (p, ms, tag='op') => Promise.race([
    p, new Promise((_,rej)=>setTimeout(()=>rej(new Error(`${tag} timeout ${ms}ms`)), ms))
  ]);

  // Fetch that follows redirects and returns blob + a good filename if we can infer it
  async function fetchFile(url, opts) {
    const res = await withTimeout(fetch(url, { credentials: 'include', redirect: 'follow', ...opts }), 15000, 'fetch');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    let filename = null;

    // Try Content-Disposition
    const cd = res.headers.get('content-disposition') || '';
    const m1 = cd.match(/filename\*?=(?:UTF-8''|")?([^;"\r\n]+)/i);
    if (m1) try { filename = decodeURIComponent(m1[1].replace(/"/g,'')); } catch {}

    // Fallback from URL
    if (!filename) {
      try {
        const u = new URL(res.url);
        filename = u.pathname.split('/').pop();
      } catch { filename = 'file' + (extOf(url) || '.bin'); }
    }
    // last fallback
    if (!/\.[a-z0-9]{2,5}$/i.test(filename)) filename += (extOf(url) || '.bin');

    const blob = await res.blob();
    return { blob, filename };
  }

  // Converts the QuickLink URL to a direct /content/enforced path if possible
  function resolveQuickLinkToEnforced(u) {
    // coursefile quicklink
    const isQL = /\/d2l\/common\/dialogs\/quickLink\/quickLink\.d2l/i.test(u);
    if (!isQL) return null;
    const url = new URL(u, location.href);
    const type = url.searchParams.get('type') || url.searchParams.get('Type');
    const fileId = url.searchParams.get('fileId');
    if (type && type.toLowerCase() === 'coursefile' && fileId) {
      // If fileId already looks enforced (rare), just return abs
      if (/^\/?content\/enforced\//i.test(fileId)) return abs(fileId);
      // Otherwise try courseId-based enforced path (most orgs store like this)
      // We can’t always reconstruct the full section folder name, but many
      // deployments allow direct fileId under the enforced root.
      return `/content/enforced/${decodeURIComponent(fileId)}`;
    }
    return null;
  }

  /* ---------- Course Detection ---------- */
  const cm = location.pathname.match(/\/d2l\/le\/(?:content|lessons|home)\/(\d+)/);
  const courseId = cm && cm[1];
  const isLessons = /\/d2l\/le\/lessons\//.test(location.pathname);
  if (!courseId) { alert("Open a course home, Content, or Lessons page first."); window.__bs_scraper_active=false; return; }

  /* ---------- UI ---------- */
  const ui = document.createElement('div');
  ui.style.cssText = `
    position:fixed;right:16px;bottom:16px;z-index:999999;font:14px/1.4 system-ui,Segoe UI,Roboto,Arial;
    color:#eee;background:#1f1f1f;border-radius:10px;box-shadow:0 8px 20px rgba(0,0,0,.35);width:360px;overflow:hidden;
  `;
  ui.innerHTML = `
    <div id="bs-drag" style="cursor:move;background:repeating-linear-gradient(135deg,#ff9f43 0,#ff9f43 10px,#2d2d2d 10px,#2d2d2d 20px);height:18px"></div>
    <div style="padding:12px 14px">
      <b>Brightspace Scraper v4.0</b>
      <div id="bs-course" style="margin:6px 0;color:#9ca3af">Detecting course…</div>
      <div id="bs-status" style="margin:8px 0;color:#ccc">Initializing…</div>
      <div id="bs-exts" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px"></div>
      <label style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
        <input type="checkbox" id="bs-deepscan" checked> Deep-scan HTML topics (quickLinks inside)
      </label>
      <label style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
        <input type="checkbox" id="bs-tabs"> Allow opening real tabs if a file won’t fetch (auto-closes)
      </label>
      <div style="height:6px;background:#333;border-radius:6px;overflow:hidden;margin-bottom:6px">
        <div id="bs-bar" style="height:6px;width:0;background:#4ade80"></div>
      </div>
      <div id="bs-count" style="margin:6px 0;color:#bbb;font-size:12px"></div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px">
        <button id="bs-start" style="background:#0b67ff;color:#fff;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Scan & Download</button>
        <button id="bs-show"  style="background:#333;color:#ccc;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Show Skipped</button>
        <button id="bs-close" style="background:#333;color:#ccc;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(ui);
  const S = t => ui.querySelector('#bs-status').textContent = t;
  const C = ui.querySelector('#bs-count');
  const B = ui.querySelector('#bs-bar');

  // Drag
  (function makeDraggable() {
    const bar = ui.querySelector('#bs-drag');
    let sx=0, sy=0, ox=0, oy=0;
    const onDown = e => { sx=e.clientX; sy=e.clientY; const r=ui.getBoundingClientRect(); ox=r.left; oy=r.top;
      document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp); };
    const onMove = e => { const dx=e.clientX-sx, dy=e.clientY-sy; ui.style.left=(ox+dx)+'px'; ui.style.top=(oy+dy)+'px'; ui.style.right='auto'; ui.style.bottom='auto'; };
    const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
    bar.addEventListener('mousedown', onDown);
  })();

  ui.querySelector('#bs-close').onclick = () => { ui.remove(); window.__bs_scraper_active=false; };
  const exDiv = ui.querySelector('#bs-exts');
  [".pdf",".mp4",".docx",".pptx",".xlsx",".zip",".txt"].forEach(x=>{
    const L=document.createElement('label'); L.innerHTML=`<input type="checkbox" data-ext="${x}" checked> ${x}`; exDiv.appendChild(L);
  });

  /* ---------- Get ToC ---------- */
  let toc=null, courseName='';
  try {
    const r = await fetch(`/d2l/api/le/1.68/${courseId}/content/toc?loadDescription=true`, { credentials:'include' });
    if (r.ok) { toc = await r.json(); courseName = san(toc.Title||''); }
  } catch {}
  ui.querySelector('#bs-course').textContent = courseName || `Course ID: ${courseId}`;

  /* ---------- DOM fallback (Lessons/Smart-Curriculum/regular pages) ---------- */
  function scrapeDOM(rootDoc=document) {
    const links=[];
    rootDoc.querySelectorAll('a[href],source[src],iframe[src],embed[src]').forEach(el=>{
      const href = abs(el.getAttribute('href')||el.getAttribute('src'));
      if(!href) return;
      // direct file
      if(looksFile(href)) links.push({Title: el.textContent.trim() || el.getAttribute('title') || 'file', Url: href});
      // quickLink
      else if (/\/quickLink\/quickLink\.d2l/i.test(href)) {
        const eff = resolveQuickLinkToEnforced(href);
        if (eff) links.push({ Title: el.textContent.trim() || 'Course file', Url: eff });
        else links.push({ Title: el.textContent.trim() || 'QuickLink', Url: href });
      }
    });
    return { Modules:[{ Title:"Page", Topics:links }] };
  }

  async function scrapeSmartCurriculum() {
    const ifr = document.querySelector('iframe[src*="smart-curriculum"]');
    if (!ifr) return null;
    try {
      const d = ifr.contentDocument || ifr.contentWindow.document;
      return scrapeDOM(d);
    } catch { return null; }
  }

  // Choose data source
  let tree;
  if (toc && toc.Modules?.length && !isLessons) {
    S('Using Brightspace ToC API.');
    tree = toc;
  } else {
    S('No ToC or Lessons detected — scanning page…');
    tree = await scrapeSmartCurriculum() || scrapeDOM();
  }

  /* ---------- Flatten topics ---------- */
  const topics=[];
  (function walk(m){ (m.Topics||[]).forEach(t=>topics.push(t)); (m.Modules||[]).forEach(walk); })(tree);
  S(`Found ${topics.length} topics.`);

  /* ---------- Deep scan for anchored quickLinks inside HTML pages ---------- */
  async function deepScan(url, seen=new Set()) {
    if (!url || seen.has(url)) return [];
    seen.add(url);
    try {
      const r = await withTimeout(fetch(url, { credentials:'include' }), 12000, 'deepScan');
      if (!r.ok) return [];
      const html = await r.text();
      const doc = new DOMParser().parseFromString(html, 'text/html');
      const found = new Set();

      doc.querySelectorAll('a[href],source[src],iframe[src],embed[src]').forEach(n=>{
        const u = abs(n.getAttribute('href')||n.getAttribute('src')||'');
        if (!u) return;
        if (looksFile(u)) found.add(u);
        else if (/quickLink\/quickLink\.d2l/i.test(u)) {
          const eff = resolveQuickLinkToEnforced(u);
          if (eff) found.add(eff); else found.add(u);
        }
      });
      return [...found];
    } catch { return []; }
  }

  /* ---------- Download orchestration ---------- */
  await new Promise(r => {
    const btn = ui.querySelector('#bs-start');
    btn.disabled = false;
    btn.onclick = r;
  });

  const wantExts = [...exDiv.querySelectorAll('input:checked')].map(i=>i.dataset.ext);
  const filterByExt = (u) => {
    const e = extOf(u);
    return !wantExts.length || wantExts.includes(e);
  };
  const doDeep = ui.querySelector('#bs-deepscan').checked;
  const allowTabs = ui.querySelector('#bs-tabs').checked;

  const zip = new JSZip();
  const skipped = [];
  ui.querySelector('#bs-show').onclick = () => {
    if (!skipped.length) return alert('No skipped items yet.');
    alert(skipped.map(s=>`${s.Title || 'item'} -> ${s.Url || '(no url)'}`).join('\n'));
  };

  // Optional confirmation when tabs are enabled
  if (allowTabs) {
    const ok = confirm("Detected instances may require opening new tabs which will automatically close. This is used to bypass dynamic loading. Continue?");
    if (!ok) { S('Cancelled.'); return; }
  }

  // Build a light queue
  const queue = [];
  function queuePush(task){ queue.push(task); }
  async function runQueue(concurrency=6){
    let i=0, active=0;
    return new Promise(res=>{
      const next = () => {
        if (i>=queue.length && active===0) return res();
        while (active<concurrency && i<queue.length){
          const fn = queue[i++]; active++;
          Promise.resolve().then(fn).catch(()=>{}).finally(()=>{ active--; next(); });
        }
      };
      next();
    });
  }

  // Add .url shortcut
  function addURLShortcut(dir, title, link){
    zip.file(`${dir}${san(title||'link')}.url`, `[InternetShortcut]\nURL=${link}\n`);
  }

  let totalPlanned = 0;
  const incBar = (()=>{ let done=0; return () => {
    done++; B.style.width = ((done/(totalPlanned||1))*100).toFixed(1)+'%';
    C.textContent = `${done}/${totalPlanned} downloaded`;
  };})();

  // Build tasks
  function scheduleDownload(dir, title, url){
    // Resolve quickLink immediately if possible
    const ql = resolveQuickLinkToEnforced(url);
    const target = ql || url;

    if (!filterByExt(target) && !/quickLink/i.test(url)) return; // skip if ext filtered and not a ql that might resolve later

    totalPlanned++;
    queuePush(async () => {
      // If still a quickLink page, try deep-scan first to discover the real file
      let candidates = [target];
      if (doDeep && !looksFile(target)) {
        const extra = await deepScan(target);
        if (extra.length) candidates = extra;
      }

      // Try candidates
      let saved = false, lastErr=null;
      for (const link of candidates) {
        try {
          if (!looksFile(link)) continue;
          const { blob, filename } = await fetchFile(link);
          zip.file(dir + san(title || filename), blob);
          incBar(); saved = true; break;
        } catch (e) { lastErr = e; }
      }

      // If not saved, maybe this file refuses fetch; open tab (optional)
      if (!saved && allowTabs) {
        try {
          const w = window.open(target, '_blank', 'noopener,noreferrer');
          await sleep(1200);
          if (w) w.close();
        } catch {}
      }

      if (!saved) {
        addURLShortcut(dir, title || 'link', target);
        skipped.push({ Title:title, Url:target, Error: (lastErr && lastErr.message) || 'unfetchable' });
        incBar();
      }
    });
  }

  // Shape modules for simple foldering
  function planModule(mods, pre=''){
    for (const m of (mods||[])) {
      const dir = pre + san(m.Title || 'Module') + '/';
      for (const t of (m.Topics||[])) {
        if (!t.Url) continue;
        scheduleDownload(dir, (t.Title || 'file') + (extOf(t.Url) || ''), t.Url);
      }
      planModule(m.Modules, dir);
    }
  }
  planModule(tree.Modules);

  // Run it
  S('Downloading…'); C.textContent = `0/${totalPlanned} downloaded`;
  await runQueue(6);

  S('Building ZIP…');
  const name = `Brightspace_${courseName || 'Course'}_${nowTag()}.zip`;
  const blob = await zip.generateAsync({type:'blob'});
  const a = document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=name; a.click();

  S(`Done. Saved ${totalPlanned - skipped.length}, skipped ${skipped.length}.`);
  setTimeout(()=>{ ui.remove(); window.__bs_scraper_active=false; }, 7000);
})();
