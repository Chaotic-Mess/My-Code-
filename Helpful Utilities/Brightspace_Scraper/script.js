/* ===========================================================
   Brightspace_Scraper V4.1  — "TabCrawler 2"
   by chaotic-mess (Zac) + ChatGPT | https://github.com/Chaotic-Mess/My-Code
   Old + new layouts: ToC, Lessons, Smart-Curriculum, QuickLinks.
   Opens real tabs only when necessary, then closes them automatically.
   =========================================================== */
(async () => {
  if (window.__bs_scraper_active) { alert("Brightspace Scraper already running."); return; }
  window.__bs_scraper_active = true;

  /* ---------------- tiny utils ---------------- */
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const abs = (u) => { try { return new URL(u, location.href).href; } catch { return null; } };
  const san = (s) => (s || "").replace(/[<>:"/\\|?*]+/g, "_").trim();
  const extOf = (u) => { const m = u && u.match(/\.[a-z0-9]{2,6}(?:$|\?)/i); return m ? m[0].toLowerCase() : ""; };
  const todayTag = () => { const d = new Date(); return `(${String(d.getFullYear()).slice(2)}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')})`; };

  const looksFile = (u) =>
    !!u && (/\.(pdf|mp4|m4v|mov|mp3|docx?|pptx?|xlsx?|zip|7z|rar|txt|csv|rtf|md|png|jpe?g|gif|svg)(?:[?#].*)?$/i.test(u)
      || /\/content\/enforced\//i.test(u)
      || /\/d2l\/common\/viewFile\.d2l/i.test(u)
      || /type=coursefile/i.test(u));

  const isHtmlLike = (u) =>
    /\.html?(?:[?#].*)?$/i.test(u) ||
    /\/viewContent\/\d+\/View/i.test(u) ||
    /\/content\/\d+(?:\/)?$/i.test(u) ||
    /\/d2l\/le\/lessons\//i.test(u) ||
    /\/d2l\/common\/dialogs\/quickLink\/quickLink\.d2l/i.test(u);

  const isQuickLink = (u) =>
    /\/d2l\/common\/dialogs\/quickLink\/quickLink\.d2l/i.test(u);

  const getParam = (url, key) => { try { return new URL(url, location.href).searchParams.get(key); } catch { return null; } };

  function log(msg) { console.log("[BScraper]", msg); try {
    const el = document.querySelector("#bs-log"); if (!el) return;
    const d = document.createElement("div"); d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    el.appendChild(d); el.scrollTop = el.scrollHeight;
  } catch {} }

  /* ---------------- course detection ---------------- */
  const m = location.pathname.match(/\/d2l\/le\/(?:content|lessons|home)\/(\d+)/);
  let courseId = m && m[1];
  const isLessons = /\/d2l\/le\/lessons\//.test(location.pathname);
  if (!courseId) {
    const a = [...document.querySelectorAll('a[href*="ou="]')].map(x => getParam(x.href, "ou")).find(Boolean);
    if (a) courseId = a;
  }
  if (!courseId) { alert("Open a course home, Content, or Lessons page first."); window.__bs_scraper_active = false; return; }

  /* ---------------- UI ---------------- */
  const ui = document.createElement("div");
  ui.style = `
    position:fixed;right:16px;bottom:16px;z-index:999999;
    background:#1e1e1e;color:#eee;padding:0;border-radius:12px;
    font:14px/1.4 system-ui,Segoe UI,Roboto,Arial;box-shadow:0 8px 20px rgba(0,0,0,.35);
    width:360px;max-height:90vh;overflow:hidden;border:1px solid #333;
  `;
  ui.innerHTML = `
    <div id="drag" style="
      background: repeating-linear-gradient(45deg,#ff7a00,#ff7a00 8px,#303030 8px,#303030 16px);
      padding:8px 12px;cursor:move;color:#000;display:flex;justify-content:space-between;align-items:center;">
      <b style="color:#111">Brightspace Scraper v3.9.6</b>
      <button id="bs-close" title="Close" style="border:0;background:#111;color:#eee;padding:4px 8px;border-radius:6px">✕</button>
    </div>
    <div style="padding:12px">
      <div id="bs-course" style="margin:6px 0;color:#9ca3af">Detecting course…</div>
      <div id="bs-status" style="margin:8px 0;color:#ccc">Idle</div>
      <div id="bs-exts" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px"></div>
      <label style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
        <input type="checkbox" id="bs-deepscan" checked> Deep Scan HTML topics
      </label>
      <label style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
        <input type="checkbox" id="bs-allowpopups"> Allow Popups / Open Tabs when needed
      </label>
      <div style="height:6px;background:#333;border-radius:6px;overflow:hidden;margin-bottom:6px">
        <div id="bs-bar" style="height:6px;width:0;background:#4ade80"></div>
      </div>
      <div id="bs-count" style="margin:6px 0;color:#bbb;font-size:12px">0/0 downloaded</div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px">
        <button id="bs-start" style="background:#0b67ff;color:#fff;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Scan & Download</button>
        <button id="bs-show" style="background:#333;color:#ccc;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Show Skipped</button>
        <button id="bs-canceltabs" style="background:#333;color:#ccc;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Close Helper Tabs</button>
      </div>
      <div id="bs-log" style="margin-top:8px;color:#9b9b9b;font-size:12px;max-height:140px;overflow:auto;border-top:1px solid #2a2a2a;padding-top:8px"></div>
    </div>
  `;
  document.body.appendChild(ui);
  const S = (t) => ui.querySelector("#bs-status").textContent = t;
  const B = ui.querySelector("#bs-bar");
  const C = ui.querySelector("#bs-count");
  ui.querySelector("#bs-close").onclick = () => { ui.remove(); window.__bs_scraper_active = false; };

  // drag
  (function makeDraggable(){
    const bar = ui.querySelector("#drag");
    let sx=0, sy=0, ox=0, oy=0, d=false;
    bar.addEventListener("mousedown",e=>{d=true;sx=e.clientX;sy=e.clientY;const r=ui.getBoundingClientRect();ox=r.right-e.clientX;oy=r.bottom-e.clientY;});
    window.addEventListener("mousemove",e=>{if(!d)return; ui.style.right=(ox+window.innerWidth-e.clientX-16-ox)+"px"; ui.style.bottom=(oy+window.innerHeight-e.clientY-16-oy)+"px";});
    window.addEventListener("mouseup",()=>d=false);
  })();

  /* ---------------- libs ---------------- */
  const load = (src) => new Promise((res,rej)=>{ if(document.querySelector(`script[src="${src}"]`)) return res();
    const s=document.createElement("script"); s.src=src; s.onload=res; s.onerror=rej; document.head.appendChild(s); });
  await load("https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js");
  await load("https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js");
  const zip = new JSZip();

  /* ---------------- type toggles ---------------- */
  const types = [".pdf",".mp4",".m4v",".mp3",".docx",".pptx",".xlsx",".zip",".txt"];
  const exDiv = ui.querySelector("#bs-exts");
  types.forEach(x => {
    const L = document.createElement("label"); L.style="font-size:12px;color:#d1d5db";
    L.innerHTML = `<input type="checkbox" data-ext="${x}" checked> ${x}`;
    exDiv.appendChild(L);
  });
  const allowedSet = () => new Set([...exDiv.querySelectorAll("input:checked")].map(c => c.dataset.ext.toLowerCase()));

  /* ---------------- ToC fetch + fallbacks ---------------- */
  S("Fetching Table of Contents…");
  let toc = null, courseName = "";
  async function fetchTOC() {
    for (const u of [
      `/d2l/api/le/1.68/${courseId}/content/toc?loadDescription=true`,
      `/d2l/api/le/1.68/${courseId}/content/toc`
    ]) { try { const r=await fetch(u,{credentials:"same-origin"}); if(r.ok) return r.json(); } catch{} }
    return null;
  }
  try { toc = await fetchTOC(); if (toc) courseName = san(toc.Title || ""); log("ToC fetched"); } catch { log("ToC failed"); }
  ui.querySelector("#bs-course").textContent = courseName || `Course ID: ${courseId}`;

  function scrapeDomLinks(doc = document) {
    const links = [];
    doc.querySelectorAll("a[href], source[src], iframe[src], embed[src], object[data], d2l-html-block[html]")
      .forEach(el => {
        if (el.tagName.toLowerCase() === "d2l-html-block") {
          const html = el.getAttribute("html") || "";
          if (!html) return;
          try {
            const d = new DOMParser().parseFromString(html, "text/html");
            d.querySelectorAll("a[href], source[src], iframe[src], embed[src], object[data]").forEach(n=>{
              const href = abs(n.getAttribute("href")||n.getAttribute("src")||n.getAttribute("data")); if (!href) return;
              links.push({ Title: (n.textContent||n.title||href).trim().slice(0,120)||"link", Url: href });
            });
          } catch {}
        } else {
          const href = abs(el.getAttribute("href")||el.getAttribute("src")||el.getAttribute("data")); if (!href) return;
          links.push({ Title: (el.textContent||el.title||href).trim().slice(0,120)||"link", Url: href });
        }
      });
    return { Modules: [{ Title: "Visible Page", Topics: links }] };
  }
  async function scrapeSmartCurriculum() {
    const iframe = document.querySelector('iframe[src*="smart-curriculum"]'); if (!iframe) return null;
    try { const doc = iframe.contentDocument || iframe.contentWindow?.document; if (!doc) return null; return scrapeDomLinks(doc); } catch { return null; }
  }
  if (!toc || !toc.Modules?.length || isLessons) {
    S("No ToC / Lessons detected — scanning DOM…");
    toc = await scrapeSmartCurriculum() || scrapeDomLinks(document);
  }

  /* ---------------- QuickLink resolution ---------------- */
  const fileIdToDirect = (fileId) => { const p = decodeURIComponent(fileId || "").replace(/^\/+/, ""); return abs(`/content/enforced/${p}`); };
  const viewFileFromQuickLink = (u) => { const fid=getParam(u,"fileId"); if(!fid) return null;
    return abs(`/d2l/common/viewFile.d2l?ou=${encodeURIComponent(courseId)}&fileId=${encodeURIComponent(fid)}`); };

  async function follow(url) {
    try { const res = await fetch(url, { credentials: "same-origin" }); const text = await res.clone().text().catch(()=> "");
      return { finalUrl: res.url || url, res, body: text }; } catch { return { finalUrl: url, res: null, body: "" }; }
  }
  function extractFileFromHtml(html) {
    if (!html) return null;
    const m1 = html.match(/href\s*=\s*"(\/content\/enforced\/[^"]+)"/i) || html.match(/"(\/content\/enforced\/[^"]+)"/i);
    if (m1) return abs(m1[1]);
    const m2 = html.match(/http-equiv=["']refresh["'][^>]*content=["'][^;]+;\s*url=([^"']+)["']/i);
    if (m2) return abs(m2[1]);
    const m3 = html.match(/(?:location\.href|window\.location(?:\.replace)?)\s*=\s*["']([^"']+)["']/i);
    if (m3) return abs(m3[1]);
    const m4 = html.match(/\/d2l\/common\/viewFile\.d2l\?[^"'<>]+/i);
    if (m4) return abs(m4[0]);
    return null;
  }
  async function resolveQuickLink(u) {
    const type = getParam(u, "type"); const fileId = getParam(u, "fileId");
    if (type === "coursefile" && fileId) { const direct = fileIdToDirect(fileId); if (direct && looksFile(direct)) return direct; }
    const vf = viewFileFromQuickLink(u); if (vf) return vf;
    const { finalUrl, body } = await follow(u);
    if (looksFile(finalUrl)) return finalUrl;
    const cand = extractFileFromHtml(body);
    return cand || u;
  }
  async function normalizeLink(u) { if (!u) return null; if (isQuickLink(u)) return await resolveQuickLink(u); return u; }

  /* ---------------- flatten topics ---------------- */
  const topics = [];
  (function walk(m){ (m.Topics||[]).forEach(t=>topics.push(t)); (m.Modules||[]).forEach(walk); })(toc);
  S(`Found ${topics.length} topics.`); log(`Topics: ${topics.length}`);

  /* ---------------- TabManager ---------------- */
  const TabManager = (() => {
    const opened = new Map(); // url -> {win, t0}
    const TIMEOUT_MS = 15000;   // hard close after 15s
    function openMany(urls) {
      const handles = [];
      for (const u of urls) {
        let w = null; try { w = window.open(u, "_blank"); } catch {}
        if (!w) { log("Popup blocked: " + u); handles.push({ url: u, win: null, blocked: true }); continue; }
        opened.set(u, { win: w, t0: Date.now() }); handles.push({ url: u, win: w, blocked: false }); log("Opened tab: "+u);
      }
      return handles;
    }
    function closeAll() {
      for (const [u, h] of opened) { try { if (h.win && !h.win.closed) h.win.close(); } catch {} opened.delete(u); log("Closed tab: "+u); }
    }
    function reapExpired() {
      const now = Date.now();
      for (const [u, h] of opened) {
        if (!h.win || h.win.closed || now - h.t0 > TIMEOUT_MS) {
          try { if (h.win && !h.win.closed) h.win.close(); } catch {}
          opened.delete(u); log("Reaped tab: "+u);
        }
      }
    }
    window.addEventListener("beforeunload", closeAll);
    setInterval(reapExpired, 2000);
    return { openMany, closeAll };
  })();
  ui.querySelector("#bs-canceltabs").onclick = () => TabManager.closeAll();

  /* ---------------- deep scan ---------------- */
  async function deepScan(url, depth = 0, visited = new Set()) {
    if (!url || visited.has(url) || depth > 3) return [];
    visited.add(url);
    try {
      const res = await fetch(url, { credentials: "same-origin" });
      if (!res.ok) return [];
      const html = await res.text();
      const d = new DOMParser().parseFromString(html, "text/html");
      const found = new Set();
      const nodes = d.querySelectorAll("a[href], source[src], iframe[src], embed[src], object[data]");
      for (const n of nodes) {
        const raw = abs(n.getAttribute("href") || n.getAttribute("src") || n.getAttribute("data") || "");
        if (!raw) continue;
        let link = raw;
        if (isQuickLink(link)) link = await resolveQuickLink(link);
        if (looksFile(link)) found.add(link);
        else if (isHtmlLike(link)) (await deepScan(link, depth+1, visited)).forEach(x => found.add(x));
      }
      return [...found];
    } catch { return []; }
  }

  /* ---------------- downloader helpers ---------------- */
  const skipped = [];
  ui.querySelector("#bs-show").onclick = () => {
    if (!skipped.length) return alert("No skipped items yet.");
    const lines = skipped.map(x => `${x.Title || x.title || "item"} → ${x.Url || x.url || "no URL"}`);
    try { navigator.clipboard?.writeText(lines.join("\n")); } catch {}
    alert(lines.join("\n"));
  };

  function decodeRFC5987(v) {
    try {
      const m = v.match(/filename\*\s*=\s*([^;]+)/i);
      if (m) return decodeURIComponent(m[1].split("''").pop().trim().replace(/^["']|["']$/g,""));
      const m2 = v.match(/filename\s*=\s*([^;]+)/i);
      if (m2) return m2[1].trim().replace(/^["']|["']$/g,"");
    } catch {}
    return null;
  }
  function extFromMime(ct) {
    const map = {
      "application/pdf": ".pdf", "video/mp4": ".mp4", "audio/mp3": ".mp3", "audio/mpeg": ".mp3",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
      "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
      "application/zip": ".zip", "text/plain": ".txt", "text/csv": ".csv", "image/png": ".png",
      "image/jpeg": ".jpg", "image/gif": ".gif", "image/svg+xml": ".svg"
    };
    ct = (ct || "").toLowerCase().split(";")[0].trim();
    return map[ct] || "";
  }

  async function saveLink(dir, title, link, allowSet) {
    try {
      const r = await fetch(link, { credentials: "same-origin" });
      if (!r.ok || r.status === 403) { zip.file(`${dir}${san(title)}.url`, `[InternetShortcut]\nURL=${link}\n`); skipped.push({ Title:title, Url:link, Type:"Shortcut" }); return false; }

      const cd = r.headers.get("content-disposition") || "";
      const hinted = decodeRFC5987(cd);
      const hintedExt = hinted ? (hinted.match(/\.[a-z0-9]{2,6}$/i) || [""])[0].toLowerCase() : "";
      const ctExt = extFromMime(r.headers.get("content-type"));
      const urlExt = extOf(link);
      const chosenExt = hintedExt || urlExt || ctExt || ".bin";

      if (allowSet.size && chosenExt && !allowSet.has(chosenExt.toLowerCase())) return false;

      const base = san(hinted ? hinted.replace(/\.[a-z0-9]{2,6}$/i, "") : (title || "file"));
      const blob = await r.blob();
      zip.file(dir + base + chosenExt, blob);
      return true;
    } catch {
      zip.file(`${dir}${san(title)}.url`, `[InternetShortcut]\nURL=${link}\n`); skipped.push({ Title:title, Url:link, Type:"Shortcut" });
      return false;
    }
  }

  /* ---------------- button handler (no autorun) ---------------- */
  ui.querySelector("#bs-start").onclick = async () => {
    try {
      ui.querySelector("#bs-start").disabled = true;
      const allowSet = allowedSet();
      const doDeep = ui.querySelector("#bs-deepscan").checked;
      const allowPopups = ui.querySelector("#bs-allowpopups").checked;

      // Preflight popup warning if needed
      const hasQuicklinks = topics.some(t => isQuickLink(t.Url || t.url || ""));
      if (hasQuicklinks && !allowPopups) {
        const proceed = confirm(
          "Detected instances that may require opening new tabs to initialize Brightspace resources.\n\n" +
          "If you continue with 'Allow Popups' enabled, you will see brief flicker as tabs open and close. " +
          "This is intentional and used to bypass dynamic loading.\n\nContinue?"
        );
        if (proceed) ui.querySelector("#bs-allowpopups").checked = true;
      }

      const finalAllowPopups = ui.querySelector("#bs-allowpopups").checked;

      /* normalize topics */
      const workList = topics.map(t => ({
        Title: t.Title || t.title || t.Name || "topic",
        Url:   t.Url || t.url || t.href || "",
        Module: (t.Module || t.moduleTitle || "Module")
      }));

      /* quicklink priming via real tabs (optional) */
      let quickPages = [...new Set(workList.map(x => x.Url).filter(u => isQuickLink(u)))];
      quickPages = quickPages.filter(u => { try { return new URL(u, location.href).origin === location.origin; } catch { return false; } });

      if (finalAllowPopups && quickPages.length) {
        const msg = `Detected ${quickPages.length} QuickLink launch pages. If you proceed, the script will open/close ${quickPages.length} tabs to prime them. Continue?`;
        if (confirm(msg)) {
          S(`Opening ${quickPages.length} helper tabs…`);
          const handles = TabManager.openMany(quickPages);
          // watchdog: wait a bit so the launch pages can set cookies/session
          await sleep(2000 + quickPages.length * 300);
          // prime by fetching each once
          for (const q of quickPages) { try { await fetch(q, { credentials: "same-origin" }); } catch {} }
          // close helper tabs now; reaper keeps cleaning too
          TabManager.closeAll();
        }
      }

      /* process and download */
      let done = 0, total = workList.length;
      C.textContent = `0/${total} processed`; B.style.width = "0%";
      S("Scanning & downloading…");

      async function processTopic(dir, t) {
        const raw = t.Url; if (!raw) { skipped.push(t); return; }
        let primary = await normalizeLink(raw);
        const targets = new Set();

        if (looksFile(primary)) targets.add(primary);
        if (doDeep && (isHtmlLike(primary) || /\.html?($|\?)/i.test(primary))) {
          const extras = await deepScan(primary);
          extras.forEach(x => targets.add(x));
        }
        if (!targets.size) { zip.file(`${dir}${san(t.Title)}.url`, `[InternetShortcut]\nURL=${primary}\n`); skipped.push({ ...t, Url: primary, Type:"Shortcut" }); return; }

        for (const link of targets) {
          await saveLink(dir, t.Title || "file", link, allowSet);
          done++; B.style.width = ((done / (total || 1)) * 100).toFixed(1) + "%";
          C.textContent = `${done}/${total} processed`;
          await sleep(60);
        }
      }

      async function walk(mods, pre = "") {
        for (const m of mods || []) {
          const dir = pre + san(m.Title || "Module") + "/";
          for (const t of m.Topics || []) await processTopic(dir, t);
          await walk(m.Modules, dir);
        }
      }

      // WATCHDOG: if progress stalls, nudge UI but continue gracefully
      let lastDone = 0;
      const watchdog = setInterval(() => {
        if (done === lastDone) { log("Watchdog: progress unchanged, continuing…"); }
        lastDone = done;
      }, 8000);

      await walk(toc.Modules || []);
      clearInterval(watchdog);

      // ensure all helper tabs are closed
      TabManager.closeAll();

      S("Building ZIP…");
      const name = `Brightspace_${courseName || "Course"}_${todayTag()}.zip`;
      const blob = await zip.generateAsync({ type: "blob" });
      saveAs(blob, name);
      S(`Done. ${done} items handled, ${skipped.length} shortcuts/skipped.`);
      log(`Saved ${name}`);
    } catch (e) {
      S("Error: " + (e.message || e)); log(e);
    } finally {
      // final cleanup
      TabManager.closeAll();
      ui.querySelector("#bs-start").disabled = false;
      window.__bs_scraper_active = false;
    }
  };

  log("Ready. Click Scan & Download to begin.");
  S("Ready.");
})();
