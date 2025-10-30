/* ===========================================================
   Brightspace_Scraper v3.9.3 (QuickLink-Strong + Header Filenames)
   by chaotic-mess | https://chaotic-mess.github.io/My-Code-/ | And partly ChatGPT🥀
   Old + new layouts. ToC, Lessons, Smart-Curriculum, QuickLinks.
   Follows redirects, resolves QuickLinks, deep-scans HTML,
   names files from Content-Disposition, and builds a ZIP.
   =========================================================== */
(async () => {
  if (window.__bs_scraper_active) { alert("Brightspace Scraper already running."); return; }
  window.__bs_scraper_active = true;

  /* ---------- Utils ---------- */
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const abs = (u) => { try { return new URL(u, location.href).href; } catch { return null; } };
  const san = (s) => (s || "").replace(/[<>:"/\\|?*]+/g, "_").trim();
  const extOf = (u) => { const m = u && u.match(/\.[a-z0-9]{2,5}(?:$|\?)/i); return m ? m[0].toLowerCase().replace(/\?.*$/,"") : ""; };
  const todayTag = () => { const d = new Date(); return `(${String(d.getFullYear()).slice(2)}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')})`; };

  const looksFile = (u) =>
    /\.(pdf|mp4|m4v|mov|avi|mkv|webm|mp3|wav|docx?|pptx?|xlsx?|zip|7z|rar|txt|csv|rtf|md|epub|png|jpe?g|gif|svg)(?:[?#].*)?$/i.test(u) ||
    /\/content\/enforced\//i.test(u) ||
    /\/d2l\/common\/viewFile\.d2l/i.test(u);

  const isHtmlLike = (u) =>
    /\.html?(?:[?#].*)?$/i.test(u) ||
    /\/viewContent\/\d+\/View/i.test(u) ||
    /\/content\/\d+(?:\/)?$/i.test(u) ||
    /\/d2l\/le\/lessons\//i.test(u) ||
    /\/d2l\/common\/dialogs\/quickLink\/quickLink\.d2l/i.test(u);

  const isQuickLink = (u) =>
    /\/d2l\/common\/dialogs\/quickLink\/quickLink\.d2l/i.test(u);

  const getParam = (url, key) => { try { return new URL(url, location.href).searchParams.get(key); } catch { return null; } };

  /* ---------- Course Detection ---------- */
  const m = location.pathname.match(/\/d2l\/le\/(?:content|lessons|home)\/(\d+)/);
  let courseId = m && m[1];
  const isLessons = /\/d2l\/le\/lessons\//.test(location.pathname);
  if (!courseId) {
    const a = [...document.querySelectorAll('a[href*="ou="]')].map(x => getParam(x.href, "ou")).find(Boolean);
    if (a) courseId = a;
  }
  if (!courseId) { alert("Open a course home, Content, or Lessons page first."); window.__bs_scraper_active = false; return; }

  /* ---------- UI (unchanged) ---------- */
  const ui = document.createElement("div");
  ui.style = `
    position:fixed;right:16px;bottom:16px;z-index:999999;
    background:#1e1e1e;color:#eee;padding:14px 16px;border-radius:10px;
    font:14px/1.4 system-ui,Segoe UI,Roboto,Arial;box-shadow:0 8px 20px rgba(0,0,0,.35);
    width:360px;max-height:90vh;overflow-y:auto;
  `;
  ui.innerHTML = `
    <b style="font-size:15px">Brightspace Scraper v3.9.3</b>
    <div id="bs-course" style="margin:6px 0;color:#9ca3af">Detecting course…</div>
    <div id="bs-status" style="margin:8px 0;color:#ccc">Initializing…</div>
    <div id="bs-exts" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px"></div>
    <label style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
      <input type="checkbox" id="bs-deepscan" checked> Deep Scan HTML topics
    </label>
    <div style="height:6px;background:#333;border-radius:6px;overflow:hidden;margin-bottom:6px">
      <div id="bs-bar" style="height:6px;width:0;background:#4ade80"></div>
    </div>
    <div id="bs-count" style="margin:6px 0;color:#bbb;font-size:12px"></div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px">
      <button id="bs-start" style="background:#0b67ff;color:#fff;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Scan & Download</button>
      <button id="bs-show" style="background:#333;color:#ccc;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Show Skipped</button>
      <button id="bs-close" style="background:#333;color:#ccc;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Close</button>
    </div>
  `;
  document.body.appendChild(ui);
  const S = (t) => ui.querySelector("#bs-status").textContent = t;
  const B = ui.querySelector("#bs-bar");
  const C = ui.querySelector("#bs-count");
  ui.querySelector("#bs-close").onclick = () => { ui.remove(); window.__bs_scraper_active = false; };

  /* ---------- JSZip + FileSaver ---------- */
  const load = (src) => new Promise(r => { const s=document.createElement("script"); s.src=src; s.onload=r; document.head.appendChild(s); });
  await load("https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js");
  await load("https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js");
  const zip = new JSZip();

  /* ---------- Type toggles ---------- */
  const types = [".pdf",".mp4",".docx",".pptx",".xlsx",".zip",".txt"];
  const exDiv = ui.querySelector("#bs-exts");
  types.forEach(x => {
    const L = document.createElement("label");
    L.innerHTML = `<input type="checkbox" data-ext="${x}" checked> ${x}`;
    exDiv.appendChild(L);
  });
  const allowedSet = () => new Set([...exDiv.querySelectorAll("input:checked")].map(c => c.dataset.ext));

  /* ---------- ToC fetch ---------- */
  S("Fetching Table of Contents…");
  let toc = null, courseName = "";
  async function fetchTOC() {
    for (const u of [
      `/d2l/api/le/1.68/${courseId}/content/toc?loadDescription=true`,
      `/d2l/api/le/1.68/${courseId}/content/toc`
    ]) {
      try { const r = await fetch(u, { credentials: "same-origin" }); if (r.ok) return r.json(); } catch {}
    }
    return null;
  }
  try { toc = await fetchTOC(); if (toc) courseName = san(toc.Title || ""); } catch {}
  ui.querySelector("#bs-course").textContent = courseName || `Course ID: ${courseId}`;

  /* ---------- DOM fallbacks (Lessons / Smart-Curriculum / html-block) ---------- */
  function scrapeDomLinks(doc = document) {
    const links = [];
    doc.querySelectorAll("a[href], source[src], iframe[src], embed[src], object[data]").forEach(el => {
      const href = abs(el.getAttribute("href") || el.getAttribute("src") || el.getAttribute("data"));
      if (!href) return;
      links.push({ Title: (el.textContent || el.getAttribute("title") || el.getAttribute("aria-label") || "link").trim(), Url: href });
    });
    doc.querySelectorAll("d2l-html-block[html]").forEach(b => {
      const h = b.getAttribute("html"); if (!h) return;
      try {
        const d = new DOMParser().parseFromString(h, "text/html");
        d.querySelectorAll("a[href], source[src], iframe[src], embed[src], object[data]").forEach(el => {
          const href = abs(el.getAttribute("href") || el.getAttribute("src") || el.getAttribute("data"));
          if (!href) return;
          links.push({ Title: (el.textContent || el.getAttribute("title") || el.getAttribute("aria-label") || "link").trim(), Url: href });
        });
      } catch {}
    });
    return { Modules: [{ Title: "Visible Page", Topics: links }] };
  }
  async function scrapeSmartCurriculum() {
    const iframe = document.querySelector('iframe[src*="smart-curriculum"]');
    if (!iframe) return null;
    try {
      const doc = iframe.contentDocument || iframe.contentWindow?.document;
      if (!doc) return null;
      return scrapeDomLinks(doc);
    } catch { return null; }
  }
  if (!toc || !toc.Modules?.length || isLessons) {
    S("No ToC found or Lessons detected — scanning DOM…");
    toc = await scrapeSmartCurriculum() || scrapeDomLinks(document);
  }

  /* ---------- QuickLink resolution ---------- */
  const fileIdToDirect = (fileId) => {
    const p = decodeURIComponent(fileId || "").replace(/^\/+/, "");
    return abs(`/content/enforced/${p}`);
  };
  const viewFileFromQuickLink = (u) => {
    const fileId = getParam(u, "fileId");
    if (!fileId) return null;
    return abs(`/d2l/common/viewFile.d2l?ou=${encodeURIComponent(courseId)}&fileId=${encodeURIComponent(fileId)}`);
  };

  async function follow(url) {
    try {
      const res = await fetch(url, { credentials: "same-origin" });
      const text = await res.clone().text().catch(()=> "");
      return { finalUrl: res.url || url, res, body: text };
    } catch { return { finalUrl: url, res: null, body: "" }; }
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
    const type = getParam(u, "type");
    const fileId = getParam(u, "fileId");

    // 1) If coursefile + fileId looks like a file, prefer direct /content/enforced
    if (type === "coursefile" && fileId) {
      const direct = fileIdToDirect(fileId);
      if (direct && looksFile(direct)) return direct;
    }

    // 2) Construct /common/viewFile as a universal fallback
    const vf = viewFileFromQuickLink(u);
    if (vf) return vf;

    // 3) Last resort: fetch the launch page and scrape
    const { finalUrl, body } = await follow(u);
    if (looksFile(finalUrl)) return finalUrl;
    const cand = extractFileFromHtml(body);
    return cand || u;
  }

  async function normalizeLink(u) {
    if (!u) return null;
    if (isQuickLink(u)) return await resolveQuickLink(u);
    return u;
  }

  /* ---------- Flatten topics ---------- */
  const topics = [];
  (function walk(m){ (m.Topics||[]).forEach(t=>topics.push(t)); (m.Modules||[]).forEach(walk); })(toc);
  S(`Found ${topics.length} topics.`);

  /* ---------- Deep Scan ---------- */
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

  /* ---------- Filename/extension from headers ---------- */
  function decodeRFC5987(v) {
    try {
      // filename*=UTF-8''...  or plain quoted
      const m = v.match(/filename\*\s*=\s*([^;]+)/i);
      if (m) {
        const val = m[1].split("''").pop().trim().replace(/^["']|["']$/g,"");
        return decodeURIComponent(val);
      }
      const m2 = v.match(/filename\s*=\s*([^;]+)/i);
      if (m2) return m2[1].trim().replace(/^["']|["']$/g,"");
    } catch {}
    return null;
  }
  function extFromMime(ct) {
    const map = {
      "application/pdf": ".pdf",
      "video/mp4": ".mp4",
      "audio/mp3": ".mp3",
      "audio/mpeg": ".mp3",
      "audio/wav": ".wav",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
      "application/msword": ".doc",
      "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
      "application/vnd.ms-powerpoint": ".ppt",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
      "application/vnd.ms-excel": ".xls",
      "text/plain": ".txt",
      "text/csv": ".csv",
      "image/png": ".png",
      "image/jpeg": ".jpg",
      "image/gif": ".gif",
      "image/svg+xml": ".svg",
      "application/zip": ".zip",
      "application/x-7z-compressed": ".7z",
      "application/vnd.rar": ".rar"
    };
    ct = (ct || "").toLowerCase().split(";")[0].trim();
    return map[ct] || "";
  }

  /* ---------- Download engine ---------- */
  const skipped = [];
  ui.querySelector("#bs-show").onclick = () => {
    if (!skipped.length) return alert("No skipped items yet.");
    const lines = skipped.map(x => `${x.Title || "item"} → ${x.Url || "no URL"}`);
    try { navigator.clipboard?.writeText(lines.join("\n")); } catch {}
    alert(lines.join("\n"));
  };

  ui.querySelector("#bs-start").onclick = async () => {
    ui.querySelector("#bs-start").disabled = true;
    const allow = allowedSet();
    const doDeep = ui.querySelector("#bs-deepscan").checked;

    const workList = topics.slice();  // process everything; filter later using headers
    let done = 0;
    C.textContent = `0/${workList.length} to process`;
    S("Scanning & downloading…");

    const addURLShortcut = (dir, title, link) => {
      zip.file(`${dir}${san(title)}.url`, `[InternetShortcut]\nURL=${link}\n`);
    };

    async function shouldKeepByExt(ext) {
      if (!allow.size) return true;
      if (!ext) return true; // if unknown, keep (we’ll still save or shortcut)
      return allow.has(ext.toLowerCase());
    }

    async function saveLink(dir, title, link) {
      try {
        const r = await fetch(link, { credentials: "same-origin" });
        if (!r.ok || r.status === 403) { addURLShortcut(dir, title, link); skipped.push({ Title:title, Url:link, Type:"Shortcut" }); return; }

        // infer filename from headers
        const cd = r.headers.get("content-disposition") || "";
        const hinted = decodeRFC5987(cd);
        const hintedExt = hinted ? (hinted.match(/\.[a-z0-9]{2,5}$/i) || [""])[0].toLowerCase() : "";
        const ctExt = extFromMime(r.headers.get("content-type"));
        const urlExt = extOf(link);

        const chosenExt = hintedExt || urlExt || ctExt || ".bin";
        if (!(await shouldKeepByExt(chosenExt))) { return; }

        const base = san(hinted ? hinted.replace(/\.[a-z0-9]{2,5}$/i, "") : (title || "file"));
        const blob = await r.blob();
        zip.file(dir + base + chosenExt, blob);
      } catch {
        addURLShortcut(dir, title, link);
        skipped.push({ Title:title, Url:link, Type:"Shortcut" });
      }
    }

    async function processTopic(dir, t) {
      const raw = t.Url; if (!raw) { skipped.push(t); return; }

      let primary = await normalizeLink(raw);
      const targets = new Set();

      // If it already looks like a file (including viewFile/enforced), queue it
      if (looksFile(primary)) targets.add(primary);

      // Deep-scan any HTML/launch page
      if (doDeep && (isHtmlLike(primary) || /\.html?($|\?)/i.test(primary))) {
        const extras = await deepScan(primary);
        extras.forEach(x => targets.add(x));
      }

      // Nothing resolved? leave shortcut to primary
      if (!targets.size) { addURLShortcut(dir, t.Title || "link", primary); skipped.push({ ...t, Url: primary, Type:"Shortcut" }); return; }

      for (const link of targets) {
        await saveLink(dir, t.Title || "file", link);
        done++;
        B.style.width = ((done / (workList.length || 1)) * 100).toFixed(1) + "%";
        C.textContent = `${done}/${workList.length} processed`;
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

    await walk(toc.Modules || []);
    S("Building ZIP…");
    const name = `Brightspace_${courseName || "Course"}_${todayTag()}.zip`;
    const blob = await zip.generateAsync({ type: "blob" });
    saveAs(blob, name);
    S(`✅ Done. ${done} items handled, ${skipped.length} shortcuts/skipped.`);
    setTimeout(() => { ui.remove(); window.__bs_scraper_active = false; }, 7000);
  };
})();
