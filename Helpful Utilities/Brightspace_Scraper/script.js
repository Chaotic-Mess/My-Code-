/* ===========================================================
   Brightspace_Scraper v3.9.6 "TabCrawler + Organized + Orange"
   by chaotic-mess | https://chaotic-mess.github.io/My-Code-/
   - Orange hazard-stripe draggable title bar (classic look)
   - Organized ZIP: preserves module/submodule hierarchy (ToC)
     + Smart bucketing for Lecture/Pre-class/Video when detectable
   - Hybrid discovery: API ToC, Lessons DOM, Smart-Curriculum,
     QuickLinks, deep recursive HTML scan, popup/tab priming
   - Robust queue with retries/timeouts; .url shortcuts fallback
   =========================================================== */
(async () => {
  if (window.__bs_scraper_active) { alert("Brightspace Scraper already running."); return; }
  window.__bs_scraper_active = true;

  /* ---------- utils ---------- */
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const abs = u => { try { return new URL(u, location.href).href; } catch { return null; } };
  const san = s => (s || "").replace(/[<>:"/\\|?*]+/g, "_").trim();
  const extOf = u => { const m = u && u.match(/\.[a-z0-9]{2,6}(?:$|\?)/i); return m ? m[0].toLowerCase() : ""; };
  const looksFile = u => !!u && (/\.(pdf|mp4|m4v|mov|mp3|docx?|pptx?|xlsx?|zip|txt|csv|rtf|md|epub|png|jpe?g)(?:[?#].*)?$/i.test(u)
    || /\/content\/enforced\//i.test(u)
    || /\/d2l\/common\/viewFile\.d2l/i.test(u)
    || /type=coursefile/i.test(u));
  const fetchJson = async (url) => { const r = await fetch(url, { credentials: "same-origin" }); if (!r.ok) return null; try { return await r.json(); } catch { return null; } };

  /* ---------- course ---------- */
  const pm = location.pathname.match(/\/d2l\/le\/(?:content|lessons|home)\/(\d+)/);
  const courseId = pm && pm[1];
  const isLessons = /\/d2l\/le\/lessons\//.test(location.pathname);
  if (!courseId) { alert("Open a course home, Content, or Lessons page first."); window.__bs_scraper_active = false; return; }

  /* ---------- UI (orange hazard top bar, draggable) ---------- */
  const ui = document.createElement("div");
  ui.id = "bs-scraper-ui";
  ui.style = `
    position:fixed;right:16px;bottom:16px;z-index:999999;
    background:#1e1e1e;border-radius:10px;box-shadow:0 12px 30px rgba(0,0,0,.45);
    color:#e8e8e8;width:380px;max-height:90vh;overflow:hidden;font:14px/1.4 system-ui,Segoe UI,Roboto,Arial;
  `;
  ui.innerHTML = `
    <div id="bs-drag" style="
      cursor:move;height:22px;display:flex;align-items:center;justify-content:space-between;padding:0 8px;
      background:
        repeating-linear-gradient(135deg, #ff9b29 0 10px, #1e1e1e 10px 20px);
      color:#111;font-weight:700;letter-spacing:.2px;
      border-top-left-radius:10px;border-top-right-radius:10px;">
      <span>Brightspace Scraper v3.9.6</span>
      <span style="display:flex;gap:8px;align-items:center">
        <button id="bs-min" title="Minimize" style="border:0;background:#00000066;color:#fff;padding:2px 6px;border-radius:4px;cursor:pointer">–</button>
        <button id="bs-close" title="Close" style="border:0;background:#00000066;color:#fff;padding:2px 6px;border-radius:4px;cursor:pointer">✕</button>
      </span>
    </div>
    <div id="bs-body" style="padding:12px 14px;overflow:auto;max-height:calc(90vh - 22px)">
      <div id="bs-course" style="margin:2px 0 8px 0;color:#9ca3af">Detecting course…</div>
      <div id="bs-status" style="margin:8px 0;color:#d1d5db">Idle</div>
      <div id="bs-exts" style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0"></div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin:6px 0 8px 0">
        <label style="display:flex;align-items:center;gap:6px">
          <input type="checkbox" id="bs-deepscan" checked> Deep Scan HTML topics
        </label>
        <label style="display:flex;align-items:center;gap:6px">
          <input type="checkbox" id="bs-allowpopups"> Allow Popups
        </label>
      </div>
      <div style="height:8px;background:#2a2a2a;border-radius:8px;overflow:hidden">
        <div id="bs-bar" style="height:8px;width:0;background:#46e07a"></div>
      </div>
      <div id="bs-count" style="margin:6px 0 10px 0;color:#9fb1a3;font-size:12px">0/0 downloaded</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px">
        <button id="bs-start" style="background:#0b67ff;color:#fff;border:0;border-radius:6px;padding:8px 12px;cursor:pointer">Scan & Download</button>
        <button id="bs-show" style="background:#333;color:#ddd;border:0;border-radius:6px;padding:8px 12px;cursor:pointer">Show Skipped</button>
      </div>
      <div id="bs-log" style="margin-top:10px;color:#a7a7a7;font-size:12px;max-height:140px;overflow:auto;border-top:1px solid #2a2a2a;padding-top:8px"></div>
    </div>
  `;
  document.body.appendChild(ui);
  const S = t => ui.querySelector("#bs-status").textContent = t;
  const B = ui.querySelector("#bs-bar");
  const C = ui.querySelector("#bs-count");
  const LOG = m => { const el = ui.querySelector("#bs-log"); const p = document.createElement("div"); p.textContent = m; el.appendChild(p); el.scrollTop = el.scrollHeight; console.log("[BScraper]", m); };

  // drag
  (() => {
    const drag = ui.querySelector("#bs-drag");
    let sx=0, sy=0, ox=0, oy=0, dragging=false;
    const onDown=e=>{dragging=true; sx=e.clientX; sy=e.clientY; const r=ui.getBoundingClientRect(); ox=r.left; oy=r.top; document.addEventListener("mousemove",onMove); document.addEventListener("mouseup",onUp);};
    const onMove=e=>{ if(!dragging) return; const dx=e.clientX-sx, dy=e.clientY-sy; ui.style.left=(ox+dx)+"px"; ui.style.top=(oy+dy)+"px"; ui.style.right="auto"; ui.style.bottom="auto"; };
    const onUp=()=>{dragging=false; document.removeEventListener("mousemove",onMove); document.removeEventListener("mouseup",onUp);};
    drag.addEventListener("mousedown",onDown);
    // default bottom-right
    ui.style.right="16px"; ui.style.bottom="16px";
  })();
  ui.querySelector("#bs-close").onclick = () => { ui.remove(); window.__bs_scraper_active = false; };
  ui.querySelector("#bs-min").onclick = () => {
    const body = ui.querySelector("#bs-body");
    body.style.display = body.style.display === "none" ? "block" : "none";
  };

  /* ---------- libs ---------- */
  const load = src => new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) return res();
    const s=document.createElement("script"); s.src=src; s.onload=res; s.onerror=rej; document.head.appendChild(s);
  });
  await load("https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js");
  await load("https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js");
  const zip = new JSZip();

  /* ---------- toggles ---------- */
  const types = [".pdf",".mp4",".mp3",".docx",".pptx",".xlsx",".zip",".txt"];
  const exDiv = ui.querySelector("#bs-exts");
  types.forEach(x => { const L=document.createElement("label"); L.style="font-size:12px;color:#d1d5db"; L.innerHTML=`<input type="checkbox" data-ext="${x}" checked> ${x}`; exDiv.appendChild(L); });

  /* ---------- ToC fetch ---------- */
  S("Fetching ToC…");
  let toc = await fetchJson(`/d2l/api/le/1.68/${courseId}/content/toc`);
  let courseName = toc?.Title ? san(toc.Title) : "";
  ui.querySelector("#bs-course").textContent = courseName || `Course ID: ${courseId}`;
  if (toc) LOG("ToC loaded via API.");
  else LOG("ToC API not available; will use DOM fallbacks.");

  /* ---------- Smart-Curriculum ---------- */
  async function extractSmartCurriculum() {
    const iframe = document.querySelector('iframe[src*="smart-curriculum"]');
    if (!iframe) return null;
    try {
      const doc = iframe.contentDocument || iframe.contentWindow.document;
      await sleep(250);
      const links=[];
      doc.querySelectorAll("a[href],source[src],iframe[src],embed[src]").forEach(n=>{
        const u=abs(n.getAttribute("href")||n.getAttribute("src")||""); if(!u) return;
        if (looksFile(u) || /quickLink/i.test(u)) links.push({ Title:n.textContent.trim()||n.title||"file", Url:u });
      });
      if (links.length) { LOG("Smart-Curriculum links extracted."); return { Modules:[{ Title:"SmartCurriculum", Topics:links }]}; }
    } catch { LOG("Smart-Curriculum blocked."); }
    return null;
  }

  /* ---------- DOM fallback ---------- */
  function parseQuickLinkUrl(url) {
    try { const u=new URL(url,location.href); const id=u.searchParams.get("fileId")||u.searchParams.get("id"); return id?decodeURIComponent(id):null; }
    catch { return null; }
  }
  function scrapeDomLinksFromPage() {
    const header = document.querySelector(".module-header,.d2l-heading-2")?.textContent?.trim() || "Page Links";
    const links=[];
    document.querySelectorAll("a[href],source[src],iframe[src],embed[src]").forEach(el=>{
      const raw=el.getAttribute("href")||el.getAttribute("src"); const href=abs(raw); if(!href) return;
      const title=(el.textContent||el.title||href).trim();
      if (looksFile(href)) links.push({ Title:title, Url:href });
      else if (/quickLink.*fileId=/i.test(href)) links.push({ Title:title||"QuickLink", Url:href });
      else if (/\/d2l\/le\/lessons|viewContent|\/content\//i.test(href)) links.push({ Title:title, Url:href }); // allow deep-scan
    });
    return { Modules:[{ Title:header, Topics:links }] };
  }

  if (!toc?.Modules?.length || isLessons) {
    S("No ToC or Lessons layout detected — scanning DOM…");
    toc = (await extractSmartCurriculum()) || scrapeDomLinksFromPage();
  }

  /* ---------- flatten topics WITH HIERARCHY (module path) ---------- */
  const flat = [];
  (function walk(mods, chain=[]) {
    for (const m of mods||[]) {
      const next = [...chain, m.Title || "Module"];
      for (const t of (m.Topics||[])) {
        flat.push({ Title: t.Title || t.TopicTitle || "file", Url: t.Url || t.UrlRaw || t.TopicUrl || "", ModulePath: next });
      }
      if (m.Modules && m.Modules.length) walk(m.Modules, next);
    }
  })(toc.Modules || []);

  /* ---------- smart bucketing (Lecture / Pre-class / Video) ---------- */
  function bucketFromTitle(title="") {
    const s = title.toLowerCase();
    if (/(pre[-\s]?class)/i.test(s)) return "Pre-class";
    if (/lecture\s*notes?/i.test(s)) return "Lecture notes";
    if (/video|echo360|media/i.test(s)) return "Video";
    return null;
  }
  const dateFromTitle = (t="") => {
    // try M101_Oct27_2025.pdf or similar
    const m = t.match(/(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[_\- ]?(\d{1,2})[_\- ]?(\d{4})/i);
    if (!m) return null;
    const months = {jan:0,feb:1,mar:2,apr:3,may:4,jun:5,jul:6,aug:7,sep:8,oct:9,nov:10,dec:11};
    const YY = String(m[3]).slice(2);
    const MM = String(months[m[1].slice(0,3).toLowerCase()]+1).padStart(2,"0");
    const DD = String(m[2]).padStart(2,"0");
    return `${YY}-${MM}-${DD}`;
  };

  /* ---------- deep scan + quicklink follow ---------- */
  async function fetchWithTimeout(url, opts={}, timeout=15000, retries=1) {
    for (let a=0;a<=retries;a++){
      try {
        const ctl = new AbortController(); const id=setTimeout(()=>ctl.abort(), timeout);
        const r = await fetch(url, { ...opts, credentials:"same-origin", signal:ctl.signal }); clearTimeout(id); return r;
      } catch (e) { if (a===retries) throw e; await sleep(300*(a+1)); }
    }
  }
  async function deepScan(url, depth=0, visited=new Set()) {
    if (!url || visited.has(url) || depth>4) return [];
    visited.add(url);
    try {
      const r = await fetchWithTimeout(url, {}, 12000, 1);
      if (!r.ok) return [];
      const html = await r.text();
      try { zip.file(`html_pages/page_${depth}_${Date.now()}.html`, html); } catch {}
      const d = new DOMParser().parseFromString(html,"text/html");
      const out = new Set();
      d.querySelectorAll("a[href],source[src],iframe[src],embed[src]").forEach(n=>{
        const u = abs(n.getAttribute("href") || n.getAttribute("src") || ""); if(!u) return;
        if (looksFile(u) || /quickLink/i.test(u)) out.add(u);
        else if (/\/d2l\/le\/lessons|viewContent|\/content\//i.test(u)) out.add(u);
      });
      const results=[];
      for (const u of out) {
        if (looksFile(u)) results.push(u);
        else if (/^https?:\/\/[^\/]+/i.test(u) && new URL(u).origin !== location.origin) continue;
        else results.push(...await deepScan(u, depth+1, visited));
      }
      return results;
    } catch { return []; }
  }

  /* ---------- build target list (organized) ---------- */
  const allowBuckets = ["Pre-class","Lecture notes","Video"];
  function buildPathFor(t) {
    const path = [...(t.ModulePath||[])];
    const b = bucketFromTitle(t.Title);
    if (b) path.push(b);
    return path.map(san).join("/");
  }

  /* ---------- UI actions ---------- */
  const skipped = [];
  ui.querySelector("#bs-show").onclick = () => {
    if (!skipped.length) return alert("No skipped items yet.");
    const txt = skipped.map(x => `${x.title} → ${x.url}`).join("\n");
    try { navigator.clipboard.writeText(txt); } catch {}
    alert(txt);
  };

  ui.querySelector("#bs-start").onclick = async () => {
    try {
      ui.querySelector("#bs-start").disabled = true;
      const deepscan = ui.querySelector("#bs-deepscan").checked;
      const allowPopups = ui.querySelector("#bs-allowpopups").checked;

      // initial set of topic candidates (keep module paths)
      let candidates = flat.filter(t => t.Url);
      LOG(`Initial topics: ${candidates.length}`);

      // deep-augment (optional)
      let extra = [];
      if (deepscan) {
        S("Deep scanning HTML topics (best-effort)...");
        for (const t of candidates) {
          if (/\.html?/i.test(t.Url) || /viewContent|\/content\//i.test(t.Url) || /quickLink/i.test(t.Url)) {
            const found = await deepScan(t.Url);
            for (const u of found) extra.push({ Title: t.Title, Url:u, ModulePath: t.ModulePath });
          }
        }
        LOG(`Deep scan new links: ${extra.length}`);
        candidates = candidates.concat(extra);
      }

      // optional prompt if quicklinks present
      const quicks = candidates.filter(x => /quickLink/i.test(x.Url));
      if (quicks.length && !allowPopups) {
        const go = confirm(
          "Detected instances that may require new tabs. " +
          "If you proceed with 'Allow Popups', you will see tabs briefly open/close to bypass Brightspace dynamic loading. Continue?"
        );
        if (go) ui.querySelector("#bs-allowpopups").checked = true;
      }

      // if popups allowed, open unique quicklink pages to prime
      let quickPages = [...new Set(quicks.map(q => q.Url))].filter(u => {
        try { return new URL(u, location.href).origin === location.origin; } catch { return false; }
      });
      if (ui.querySelector("#bs-allowpopups").checked && quickPages.length) {
        if (confirm(`Open ${quickPages.length} helper tabs to prime quickLinks?`)) {
          S(`Opening ${quickPages.length} helper tabs…`);
          for (const u of quickPages) { const w = window.open(u, "_blank"); if (w) await sleep(250); }
          await sleep(2500);
          // best-effort close (cannot force-close cross-origin tabs not opened via script in some browsers)
          // We rely on the user’s popup policy; generally works well on Brightspace (same-origin).
          // No forced close here to avoid cross-origin issues; browser will collect.
        }
      }

      // build final tasks list and filter by extension
      const allowed = [...exDiv.querySelectorAll("input:checked")].map(i=>i.dataset.ext.toLowerCase());
      const tasks = [];
      const seen = new Set();

      function pushTask(title, url, modulePath) {
        const key = `${url}::${modulePath.join("/")}`;
        if (seen.has(key)) return;
        seen.add(key);
        const e = extOf(url).toLowerCase();
        if (!e || !allowed.length || allowed.includes(e)) tasks.push({ title, url, modulePath });
      }

      for (const t of candidates) {
        const path = t.ModulePath || ["Module"];
        // If quickLink: keep as is (downloader will try to resolve)
        pushTask(t.Title, t.Url, path);
      }

      if (!tasks.length) { S("No downloadable targets found."); ui.querySelector("#bs-start").disabled=false; return; }

      // progress UI
      let done = 0, failed = 0;
      const total = tasks.length;
      C.textContent = `0/${total} downloaded`;
      B.style.width = "0%";
      S(`Downloading ${total} items…`);

      // worker (retry+timeout)
      async function fetchWithFallback(url) {
        try {
          const r = await fetchWithTimeout(url, {}, 15000, 1);
          if (!r.ok) return { type:"shortcut" };
          const ct = r.headers.get("content-type")||"";
          if (ct.includes("text/html") && !looksFile(url)) {
            // parse, find first direct file
            const body = await r.text();
            const d = new DOMParser().parseFromString(body,"text/html");
            const subs = [];
            d.querySelectorAll("a[href],source[src],iframe[src],embed[src]").forEach(n=>{
              const u=abs(n.getAttribute("href")||n.getAttribute("src")||""); if(u) subs.push(u);
            });
            const dl = subs.find(u => looksFile(u));
            if (dl) {
              const rr = await fetchWithTimeout(dl, {}, 15000, 1);
              if (rr?.ok) return { type:"blob", url: dl, blob: await rr.blob() };
            }
            return { type:"html", html: body };
          } else {
            return { type:"blob", url, blob: await r.blob() };
          }
        } catch { return { type:"shortcut" }; }
      }

      // orderly naming + folders
      function goodFileBase(title) {
        const dTag = dateFromTitle(title);
        return dTag ? `${title} (${dTag})` : title;
      }
      function zipPath(modulePath, title, urlOrExt) {
        const bucket = bucketFromTitle(title);
        const mp = [...modulePath];
        if (bucket && !mp.includes(bucket)) mp.push(bucket);
        const baseDir = mp.map(san).join("/");
        const baseName = san(goodFileBase(title));
        const ext = typeof urlOrExt === "string" ? (extOf(urlOrExt) || ".bin") : ".bin";
        return `${baseDir}/${baseName}${ext}`;
      }

      // run downloads with small concurrency
      const concurrency = 6;
      let idx = 0;
      async function worker() {
        while (true) {
          const i = idx++;
          if (i >= tasks.length) break;
          const t = tasks[i];
          try {
            // quicklink: try as-is; fallback .url
            const res = await fetchWithFallback(t.url);
            if (res.type === "blob") {
              const path = zipPath(t.modulePath, t.title, res.url);
              zip.file(path, res.blob);
              done++;
            } else if (res.type === "html") {
              const path = zipPath(t.modulePath, `${t.title} (page)`, ".html");
              zip.file(path, res.html);
              failed++;
              skipped.push({ title: t.title, url: t.url });
            } else {
              // shortcut
              const path = zipPath(t.modulePath, t.title, ".url");
              zip.file(path, `[InternetShortcut]\nURL=${t.url}\n`);
              failed++;
              skipped.push({ title: t.title, url: t.url });
            }
          } catch {
            const path = zipPath(t.modulePath, t.title, ".url");
            zip.file(path, `[InternetShortcut]\nURL=${t.url}\n`);
            failed++;
            skipped.push({ title: t.title, url: t.url });
          }
          const prog = ((done+failed)/total)*100;
          B.style.width = `${prog.toFixed(1)}%`;
          C.textContent = `${done}/${total} downloaded (failed ${failed})`;
          await sleep(40);
        }
      }
      await Promise.all(Array(concurrency).fill(0).map(worker));

      // zip it
      S("Building ZIP…");
      const now=new Date(); const tag=`(${String(now.getFullYear()).slice(2)}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')})`;
      const name = `Brightspace_${courseName || courseId}_${tag}.zip`;
      const blob = await zip.generateAsync({ type:"blob" });
      saveAs(blob, name);
      S(`Done: ${done} files, ${failed} skipped.`);
    } catch (e) {
      S("Error: "+(e.message||e));
    } finally {
      ui.querySelector("#bs-start").disabled = false;
      window.__bs_scraper_active = false;
    }
  };
})();
