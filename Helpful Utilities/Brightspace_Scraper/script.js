/* ===========================================================
   Brightspace_Scraper v3.9.2 (Deep Hybrid + QuickLink Resolver)
   by chaotic-mess | https://chaotic-mess.github.io/My-Code-/ | and Partly ChatGPT 🥀
   Dark-mode downloader for Brightspace (PDFs, MP4s, DOCXs, etc.)
   Auto-detects course, deep-scans HTML topics, resolves QuickLinks,
   follows redirects, and builds a ZIP.
   =========================================================== */
(async () => {
  if (window.__bs_scraper_active) { alert("Brightspace Scraper already running."); return; }
  window.__bs_scraper_active = true;

  /* ---------- Utility Helpers ---------- */
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const abs = (u) => { try { return new URL(u, location.href).href; } catch { return null; } };
  const san = (s) => (s || "").replace(/[<>:"/\\|?*]+/g, "_").trim();
  const todayTag = () => {
    const d = new Date();
    return `(${String(d.getFullYear()).slice(2)}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')})`;
  };

  // we treat these as directly downloadable “files”
  const FILE_EXT_RE = /\.(pdf|mp4|m4v|mov|avi|mkv|webm|mp3|wav|docx?|pptx?|xlsx?|zip|7z|rar|txt|csv|rtf|md|epub|png|jpe?g|gif|svg)(?:[?#].*)?$/i;
  const looksFile = (u) =>
    FILE_EXT_RE.test(u) ||
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

  const getParam = (url, key) => {
    try { return new URL(url, location.href).searchParams.get(key); } catch { return null; }
  };

  // safe logger
  const log = (...a) => { try { console.log("[BS-Scraper]", ...a); } catch {} };
  const warn = (...a) => { try { console.warn("[BS-Scraper]", ...a); } catch {} };

  /* ---------- Course Detection ---------- */
  const m = location.pathname.match(/\/d2l\/le\/(?:content|lessons|home)\/(\d+)/);
  let courseId = m && m[1];
  const isLessons = /\/d2l\/le\/lessons\//.test(location.pathname);

  // fallback: try from links (ou=)
  if (!courseId) {
    const a = [...document.querySelectorAll('a[href*="ou="]')].map(x => getParam(x.href, "ou")).find(Boolean);
    if (a) courseId = a;
  }

  if (!courseId) {
    alert("Open a course home, Content, or Lessons page first.");
    window.__bs_scraper_active = false;
    return;
  }

  /* ---------- UI Overlay (unchanged styling) ---------- */
  const ui = document.createElement("div");
  ui.style = `
    position:fixed;right:16px;bottom:16px;z-index:999999;
    background:#1e1e1e;color:#eee;padding:14px 16px;border-radius:10px;
    font:14px/1.4 system-ui,Segoe UI,Roboto,Arial;box-shadow:0 8px 20px rgba(0,0,0,.35);
    width:360px;max-height:90vh;overflow-y:auto;
  `;
  ui.innerHTML = `
    <b style="font-size:15px">Brightspace Scraper v3.9.2</b>
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

  /* ---------- Load JSZip + FileSaver ---------- */
  const load = (src) => new Promise(r => { const s=document.createElement("script"); s.src=src; s.onload=r; document.head.appendChild(s); });
  await load("https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js");
  await load("https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js");
  const zip = new JSZip();

  /* ---------- File Type Toggles ---------- */
  const types = [".pdf",".mp4",".docx",".pptx",".xlsx",".zip",".txt"];
  const exDiv = ui.querySelector("#bs-exts");
  types.forEach(x => {
    const L = document.createElement("label");
    L.innerHTML = `<input type="checkbox" data-ext="${x}" checked> ${x}`;
    exDiv.appendChild(L);
  });

  /* ---------- Fetch ToC (try rich version with descriptions) ---------- */
  S("Fetching Table of Contents…");
  let toc = null, courseName = "";
  async function fetchTOC() {
    const tryUrls = [
      `/d2l/api/le/1.68/${courseId}/content/toc?loadDescription=true`,
      `/d2l/api/le/1.68/${courseId}/content/toc`
    ];
    for (const u of tryUrls) {
      try {
        const r = await fetch(u, { credentials: "same-origin" });
        if (r.ok) return r.json();
      } catch {}
    }
    return null;
  }
  try {
    toc = await fetchTOC();
    if (toc) courseName = san(toc.Title || "");
  } catch {}

  ui.querySelector("#bs-course").textContent = courseName || `Course ID: ${courseId}`;

  /* ---------- DOM + Lessons + Smart-Curriculum fallback ---------- */
  // Parse visible page including <d2l-html-block html="..."> attribute payloads
  function scrapeDomLinks(doc = document) {
    const links = [];

    // 1) real DOM anchors / media
    doc.querySelectorAll("a[href], source[src], iframe[src], embed[src], object[data]").forEach(el => {
      const href = abs(el.getAttribute("href") || el.getAttribute("src") || el.getAttribute("data"));
      if (!href) return;

      // Resolve classic quickLink anchors into something actionable (we still push raw; we'll resolve later)
      links.push({
        Title: (el.textContent || el.getAttribute("title") || el.getAttribute("aria-label") || "link").trim(),
        Url: href
      });
    });

    // 2) d2l-html-block holds raw HTML in an attribute called "html"
    doc.querySelectorAll("d2l-html-block[html]").forEach(b => {
      const h = b.getAttribute("html");
      if (!h) return;
      try {
        const d = new DOMParser().parseFromString(h, "text/html");
        d.querySelectorAll("a[href], source[src], iframe[src], embed[src], object[data]").forEach(el => {
          const href = abs(el.getAttribute("href") || el.getAttribute("src") || el.getAttribute("data"));
          if (!href) return;
          links.push({
            Title: (el.textContent || el.getAttribute("title") || el.getAttribute("aria-label") || "link").trim(),
            Url: href
          });
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
    } catch (e) {
      warn("Smart-Curriculum blocked:", e);
      return null;
    }
  }

  if (!toc || !toc.Modules?.length || isLessons) {
    S("No ToC found or Lessons detected — scanning DOM…");
    toc = await scrapeSmartCurriculum() || scrapeDomLinks(document);
  }

  /* ---------- Normalize / Resolve Links ---------- */

  // Follow redirects to find a final URL (same-origin)
  async function follow(url) {
    try {
      const res = await fetch(url, { credentials: "same-origin" });
      // fetch follows 3xx by default; res.url is final URL when same-origin
      return { finalUrl: res.url || url, response: res, bodyText: await res.text().catch(()=> "") };
    } catch (e) {
      return { finalUrl: url, response: null, bodyText: "" };
    }
  }

  // Extract /content/enforced or viewFile link from a quickLink HTML page
  function extractFileFromHtml(html) {
    if (!html) return null;
    // direct enforced path
    const m1 = html.match(/href\s*=\s*"(\/content\/enforced\/[^"]+)"/i) || html.match(/"(\/content\/enforced\/[^"]+)"/i);
    if (m1) return abs(m1[1]);

    // meta refresh
    const m2 = html.match(/http-equiv=["']refresh["'][^>]*content=["'][^;]+;\s*url=([^"']+)["']/i);
    if (m2) return abs(m2[1]);

    // JS redirect
    const m3 = html.match(/(?:location\.href|window\.location(?:\.replace)?)\s*=\s*["']([^"']+)["']/i);
    if (m3) return abs(m3[1]);

    // viewFile pattern inside HTML
    const m4 = html.match(/\/d2l\/common\/viewFile\.d2l\?[^"'<>]+/i);
    if (m4) return abs(m4[0]);

    return null;
  }

  // Build a /d2l/common/viewFile.d2l URL from a quickLink’s fileId
  function viewFileFromQuickLink(quickLinkUrl) {
    const fileId = getParam(quickLinkUrl, "fileId");
    if (!fileId) return null;
    const full = `/d2l/common/viewFile.d2l?ou=${encodeURIComponent(courseId)}&fileId=${encodeURIComponent(fileId)}`;
    return abs(full);
    // this streams the file via Brightspace and works even when content/enforced root is unknown
  }

  // Resolve QuickLink → direct file if possible (content/enforced or viewFile)
  async function resolveQuickLink(u) {
    // 1) If we can derive viewFile directly from query param, do it.
    const vf = viewFileFromQuickLink(u);
    if (vf) return vf;

    // 2) Otherwise, try fetching & parsing the landing page
    const { finalUrl, response, bodyText } = await follow(u);

    // Same-origin & already ended at a file?
    if (looksFile(finalUrl)) return finalUrl;

    // Parse HTML for a deeper destination
    const candidate = extractFileFromHtml(bodyText);
    if (candidate && looksFile(candidate)) return candidate;

    // As a last resort, just return the original (we'll save a .url)
    return u;
  }

  // Normalize: quickLink → resolved; otherwise pass-through
  async function normalizeLink(u) {
    if (!u) return null;
    if (isQuickLink(u)) return await resolveQuickLink(u);
    return u;
  }

  /* ---------- Collect Topics (flatten) ---------- */
  const topics = [];
  (function walk(m) {
    (m.Topics || []).forEach(t => topics.push(t));
    (m.Modules || []).forEach(walk);
  })(toc);

  // Count includes html-like too (so you see progress even if original item is html)
  S(`Found ${topics.length} topics.`);

  /* ---------- Deep Scan HTML / QuickLink pages ---------- */
  async function deepScan(url, depth = 0, visited = new Set()) {
    if (!url || visited.has(url) || depth > 3) return [];
    visited.add(url);

    try {
      const res = await fetch(url, { credentials: "same-origin" });
      if (!res.ok) return [];
      const html = await res.text();
      const d = new DOMParser().parseFromString(html, "text/html");

      // parse everything in page (incl. quickLinks)
      const found = new Set();

      const nodes = d.querySelectorAll("a[href], source[src], iframe[src], embed[src], object[data]");
      for (const n of nodes) {
        const raw = abs(n.getAttribute("href") || n.getAttribute("src") || n.getAttribute("data") || "");
        if (!raw) continue;

        let link = raw;
        if (isQuickLink(link)) {
          link = await resolveQuickLink(link);
        }

        if (looksFile(link)) {
          found.add(link);
        } else if (isHtmlLike(link)) {
          const sub = await deepScan(link, depth + 1, visited);
          sub.forEach(x => found.add(x));
        }
      }

      return [...found];
    } catch (e) {
      warn("deepScan failed:", e);
      return [];
    }
  }

  /* ---------- Download Engine ---------- */
  const skipped = [];
  ui.querySelector("#bs-show").onclick = () => {
    if (!skipped.length) return alert("No skipped items yet.");
    const lines = skipped.map(x => `${x.Title || "item"} → ${x.Url || "no URL"}`);
    try {
      navigator.clipboard?.writeText(lines.join("\n"));
    } catch {}
    alert(lines.join("\n"));
  };

  ui.querySelector("#bs-start").onclick = async () => {
    ui.querySelector("#bs-start").disabled = true;

    const allow = [...exDiv.querySelectorAll("input:checked")].map(c => c.dataset.ext);
    const doDeep = ui.querySelector("#bs-deepscan").checked;

    // We PROCESS if the topic is a direct file OR html-like/quicklink (so deep-scan can run)
    const shouldProcess = (t) => {
      const u = t.Url || "";
      const e = (u.match(/\.[a-z0-9]{2,5}(?:$|\?)/i) || [""])[0].toLowerCase();
      return looksFile(u) ? (!allow.length || allow.includes(e)) : isHtmlLike(u);
    };

    const workList = topics.filter(shouldProcess);
    let done = 0;
    C.textContent = `0/${workList.length} to process`;
    S("Scanning & downloading…");

    const addURLShortcut = (dir, title, link) => {
      zip.file(`${dir}${san(title)}.url`, `[InternetShortcut]\nURL=${link}\n`);
    };

    async function processTopic(dir, t) {
      const raw = t.Url;
      if (!raw) { skipped.push(t); return; }

      // normalize (resolve quickLinks, follow redirects if needed)
      let primary = await normalizeLink(raw);
      const targets = new Set();

      if (looksFile(primary)) {
        targets.add(primary);
      }

      // optionally deep-scan HTML/launch/landing pages to find embedded files
      if (doDeep && (isHtmlLike(primary) || /\.html?($|\?)/i.test(primary))) {
        const extras = await deepScan(primary);
        extras.forEach(x => targets.add(x));
      }

      // If nothing resolved as a file, at least leave a shortcut to what we saw
      if (!targets.size) {
        addURLShortcut(dir, t.Title || "link", primary);
        skipped.push({ ...t, Url: primary, Type: "Shortcut" });
        return;
      }

      // download each target (limit network spikiness slightly)
      for (const link of targets) {
        try {
          const r = await fetch(link, { credentials: "same-origin" });
          if (!r.ok || r.status === 403) {
            addURLShortcut(dir, t.Title || "link", link);
            skipped.push({ ...t, Url: link, Type: "Shortcut" });
            continue;
          }
          const blob = await r.blob();
          const extMatch = link.match(/\.[a-z0-9]{2,5}(?:$|\?)/i);
          const ext = extMatch ? extMatch[0].toLowerCase().replace(/\?.*$/, "") : ".bin";
          const base = san(t.Title || "file");
          zip.file(dir + base + ext, blob);
          done++;
          B.style.width = ((done / (workList.length || 1)) * 100).toFixed(1) + "%";
          C.textContent = `${done}/${workList.length} processed`;
        } catch (e) {
          addURLShortcut(dir, t.Title || "link", link);
          skipped.push({ ...t, Url: link, Type: "Shortcut" });
        }
        await sleep(60);
      }
    }

    async function walk(mods, pre = "") {
      for (const m of mods || []) {
        const dir = pre + san(m.Title || "Module") + "/";
        for (const t of m.Topics || []) {
          if (!shouldProcess(t)) { skipped.push(t); continue; }
          await processTopic(dir, t);
        }
        await walk(m.Modules, dir);
      }
    }

    await walk(toc.Modules || []);
    S("Building ZIP…");
    const name = `Brightspace_${courseName || "Course"}_${todayTag()}.zip`;
    const blob = await zip.generateAsync({ type: "blob" });
    saveAs(blob, name);
    S(`✅ Done: ${done} items saved, ${skipped.length} shortcuts/skipped.`);
    setTimeout(() => { ui.remove(); window.__bs_scraper_active = false; }, 7000);
  };

  log("Ready. Detected course:", courseId, "Name:", courseName || "(unknown)");
})();
