/* ===========================================================
   Brightspace Scraper v3.9.4 "TabCrawler"
   by chaotic-mess | https://chaotic-mess.github.io/My-Code-/ | and part ChatGPT ❤️
   Full detection: API ToC, Lessons, Smart-Curriculum, QuickLinks.
   Opens real tabs to expand internal pages, scrapes files, zips them.
   =========================================================== */
(async () => {
  if (window.__bs_scraper_active) {
    alert("Brightspace Scraper already running.");
    return;
  }
  window.__bs_scraper_active = true;

  /* ---------- Utilities ---------- */
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const abs = u => { try { return new URL(u, location.href).href; } catch { return null; } };
  const san = s => (s || "").toString().replace(/[<>:"/\\|?*]+/g, "_").trim();
  const extOf = u => { try { const m = u && u.match(/(\.[a-z0-9]{2,6})(?:[?#].*)?$/i); return m ? m[1].toLowerCase() : ""; } catch { return ""; } };
  const isBrightspaceInternal = u => !!(u && (/\/d2l\//i.test(u) || /brightspace\.|bright\.uvic\.ca/i.test(u) || /content\/enforced\//i.test(u)));
  const looksFile = u => !!(u && (/\.(pdf|mp4|m4v|mov|webm|mp3|docx?|pptx?|xlsx?|zip|txt|csv|rtf|md|epub)(?:[?#].*)?$/i.test(u)
    || /\/content\/enforced\//i.test(u) || /\/d2l\/common\/viewFile\.d2l/i.test(u) || /download=1/i.test(u)));
  const looksHtmlLike = u => !!(u && (/\.(?:html?|php|aspx)(?:[?#].*)?$/i.test(u) || /viewContent|lessons|lessons\/|units\/|topics\/|d2l\/le\/|quickLink/i.test(u)));
  const sanitizeFilename = s => san(s || "file");
  const humanDateTag = () => {
    const d = new Date();
    const yy = String(d.getFullYear()).slice(2);
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    return `(${yy}-${mm}-${dd})`;
  };

  /* ---------- Course detection ---------- */
  const courseMatch = location.pathname.match(/\/d2l\/le\/(?:content|lessons|home)\/(\d+)/i) ||
                      location.pathname.match(/\/d2l\/le\/lessons\/(\d+)/i) ||
                      location.pathname.match(/\/d2l\/home\/(\d+)/i);
  const courseId = courseMatch && courseMatch[1];
  if (!courseId) {
    alert("Open a Brightspace course home, Content, or Lessons page first.");
    window.__bs_scraper_active = false;
    return;
  }
  const isLessons = /\/d2l\/le\/lessons\//i.test(location.pathname);

  /* ---------- Build UI (draggable with hazard topbar) ---------- */
  const existingUi = document.querySelector("#bsx-root");
  if (existingUi) existingUi.remove();

  const root = document.createElement("div");
  root.id = "bsx-root";
  Object.assign(root.style, {
    position: "fixed",
    right: "16px",
    bottom: "16px",
    zIndex: 2147483647,
    width: "380px",
    maxHeight: "88vh",
    overflow: "auto",
    font: "13px/1.35 system-ui,Segoe UI,Roboto,Arial",
    boxShadow: "0 10px 30px rgba(0,0,0,.35)",
    borderRadius: "10px",
  });

  // Inner container + theme
  root.innerHTML = `
  <div id="bsx-panel" style="background:#111217;color:#e6eef8;border-radius:10px;overflow:hidden">
    <div id="bsx-top" style="cursor:move;padding:8px 10px;background:repeating-linear-gradient(45deg,#2b2b2b 0 6px,#3b3b3b 6px 12px);color:#fff;display:flex;align-items:center;justify-content:space-between">
      <div style="display:flex;gap:10px;align-items:center">
        <strong style="font-size:14px">Brightspace Scraper v3.9.4</strong>
        <span id="bsx-course" style="font-size:12px;color:#cbd5e1;margin-left:6px"></span>
      </div>
      <div style="display:flex;gap:6px;align-items:center">
        <button id="bsx-gear" title="Settings" style="background:#222;border:0;color:#fff;padding:6px;border-radius:6px;cursor:pointer">⚙</button>
        <button id="bsx-close" title="Close" style="background:#222;border:0;color:#fff;padding:6px;border-radius:6px;cursor:pointer">✖</button>
      </div>
    </div>

    <div style="padding:10px">
      <div id="bsx-status" style="color:#94a3b8;margin-bottom:8px">Initializing…</div>

      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px" id="bsx-exts"></div>

      <label style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        <input type="checkbox" id="bsx-deepscan" checked> Aggressive Deep Scan (HTML pages)
      </label>

      <div style="height:8px;background:#222;border-radius:8px;overflow:hidden;margin-bottom:8px">
        <div id="bsx-bar" style="height:8px;width:0;background:#16a34a"></div>
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div id="bsx-count" style="color:#a6b0bf">0 / 0 downloaded</div>
        <div style="display:flex;gap:6px">
          <button id="bsx-start" style="background:#0b67ff;color:#fff;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Scan & Download</button>
          <button id="bsx-show" style="background:#222;border:0;color:#fff;padding:6px 10px;border-radius:6px;cursor:pointer">Show Skipped</button>
        </div>
      </div>

      <div style="color:#98a7b9;font-size:12px;margin-bottom:8px">
        <label style="display:flex;align-items:center;gap:6px">
          <input type="checkbox" id="bsx-autoclose" checked> Auto-close tabs after scrape
        </label>
      </div>

      <div id="bsx-log" style="background:#0b0c0e;color:#9fb3c9;padding:8px;border-radius:6px;height:160px;overflow:auto;font-family:monospace;font-size:12px"></div>
    </div>
  </div>
  `;
  document.body.appendChild(root);

  // Dragging
  const panel = document.getElementById("bsx-panel");
  const topbar = document.getElementById("bsx-top");
  let dragging = false, dragOffsetX = 0, dragOffsetY = 0;
  topbar.addEventListener("mousedown", e => {
    dragging = true;
    const rect = panel.getBoundingClientRect();
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;
    document.body.style.userSelect = "none";
  });
  window.addEventListener("mousemove", e => {
    if (!dragging) return;
    panel.style.position = "fixed";
    panel.style.left = (e.clientX - dragOffsetX) + "px";
    panel.style.top = (e.clientY - dragOffsetY) + "px";
    panel.style.right = "auto";
    panel.style.bottom = "auto";
  });
  window.addEventListener("mouseup", () => { dragging = false; document.body.style.userSelect = ""; });

  document.getElementById("bsx-close").onclick = () => { root.remove(); window.__bs_scraper_active = false; };

  const logEl = document.getElementById("bsx-log");
  const statusEl = document.getElementById("bsx-status");
  const barEl = document.getElementById("bsx-bar");
  const countEl = document.getElementById("bsx-count");
  const courseLabel = document.getElementById("bsx-course");
  const appendLog = (t) => { const p = document.createElement("div"); p.textContent = t; logEl.appendChild(p); logEl.scrollTop = logEl.scrollHeight; console.log("[BSX]", t); };
  const setStatus = s => { statusEl.textContent = s; appendLog(s); };

  /* ---------- Load dependencies ---------- */
  const loadScript = src => new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) { res(); return; }
    const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s);
  });

  try {
    await loadScript("https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js");
    await loadScript("https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js");
  } catch (e) {
    setStatus("Failed to load libraries. Check network.");
    throw e;
  }
  const JSZip = window.JSZip;
  const saveAs = window.saveAs || (window.saveAs = (blob, name) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  /* ---------- Extension toggles ---------- */
  const exts = [".pdf", ".mp4", ".m4v", ".mov", ".webm", ".docx", ".pptx", ".xlsx", ".zip", ".txt", ".csv"];
  const exDiv = document.getElementById("bsx-exts");
  exts.forEach(x => {
    const l = document.createElement("label");
    l.style = "display:inline-flex;align-items:center;gap:6px;background:#121214;padding:6px;border-radius:6px;color:#dbe7f5;font-size:12px";
    l.innerHTML = `<input type="checkbox" data-ext="${x}" checked> ${x}`;
    exDiv.appendChild(l);
  });

  /* ---------- ToC attempt ---------- */
  setStatus("Fetching Table of Contents...");
  let toc = null, courseName = "", modules = null;
  try {
    const r = await fetch(`/d2l/api/le/1.68/${courseId}/content/toc`, { credentials: "same-origin" });
    if (r.ok) {
      toc = await r.json();
      courseName = sanitizeFilename(toc.Title || "");
      courseLabel.textContent = courseName || `Course ${courseId}`;
      appendLog("ToC loaded via API.");
    } else {
      appendLog("ToC API not available or returned non-ok.");
    }
  } catch (e) {
    appendLog("ToC API error or blocked.");
  }

  /* ---------- Smart-curriculum iframe detection ---------- */
  async function scrapeSmartCurriculumIframe() {
    try {
      const iframe = document.querySelector('iframe[src*="smart-curriculum"], iframe[src*="smart-curriculum"]');
      if (!iframe) return null;
      // try to access iframe doc
      try {
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        const results = [];
        doc.querySelectorAll('a[href], source[src], iframe[src], embed[src]').forEach(n => {
          const u = abs(n.getAttribute("href") || n.getAttribute("src"));
          if (!u) return;
          if (looksFile(u)) results.push({ Title: n.textContent.trim() || n.getAttribute("title") || "file", Url: u });
        });
        return { Modules: [{ Title: "SmartCurriculum", Topics: results }] };
      } catch (e) {
        appendLog("Smart-curriculum iframe blocked by cross-origin.");
        return null;
      }
    } catch (e) { return null; }
  }

  /* ---------- DOM link sniffer fallback ---------- */
  function scrapeDomLinks() {
    appendLog("Scanning page DOM for links and media sources...");
    const links = [];
    document.querySelectorAll("a[href], source[src], iframe[src], embed[src]").forEach(el => {
      const href = abs(el.getAttribute("href") || el.getAttribute("src"));
      if (!href) return;
      const title = (el.textContent || el.getAttribute("title") || "").trim() || "file";
      // QuickLink special-case: quickLink.d2l? ... fileId=
      if (/quickLink/i.test(href) && /fileId=/i.test(href)) {
        // Try to infer enforced path if possible; otherwise store quicklink
        const fid = decodeURIComponent((href.match(/fileId=([^&]+)/i) || [,""])[1] || "");
        const enforced = fid ? (`/content/enforced/${fid}`) : href;
        links.push({ Title: title || "QuickLink File", Url: enforced });
      } else {
        links.push({ Title: title, Url: href });
      }
    });
    return { Modules: [{ Title: "Loose Files", Topics: links }] };
  }

  // If toc not present or empty OR we are on Lessons page, attempt SmartCurriculum or DOM scan
  if (!toc || !toc.Modules || !toc.Modules.length || isLessons) {
    setStatus(isLessons ? "Lessons layout detected — scanning page…" : "No ToC found; scanning page links…");
    const smart = await scrapeSmartCurriculumIframe();
    if (smart && smart.Modules && smart.Modules.length) {
      toc = smart;
      appendLog("Using SmartCurriculum iframe scan.");
    } else {
      toc = scrapeDomLinks();
      appendLog("Using DOM fallback scan.");
    }
  }

  // Collect all topics into a flat array while preserving module path
  const topics = [];
  function walkModules(mods, pathParts = []) {
    for (const m of mods || []) {
      const myPath = pathParts.concat([m.Title || "Module"]);
      (m.Topics || []).forEach(t => {
        topics.push({ Topic: t, ModulePath: myPath.slice() });
      });
      if (m.Modules && m.Modules.length) walkModules(m.Modules, myPath);
    }
  }
  walkModules(toc.Modules || []);
  setStatus(`Found ${topics.length} topics (initial).`);
  appendLog(`${topics.length} topic entries collected.`);

  /* ---------- QuickLink resolver ---------- */
  function resolveQuickLinkUrl(raw) {
    if (!raw) return raw;
    try {
      const u = new URL(raw, location.href);
      // If it's quickLink with fileId=..., try to build enforced content link
      if (/quickLink/i.test(u.pathname) || /quickLink/i.test(u.search)) {
        const fid = decodeURIComponent((u.search.match(/fileId=([^&]+)/i) || [,""])[1] || "");
        // if fileId looks like "Folder%2fFile.pdf" or "Print%20outs%2fFile.pdf"
        if (fid) {
          // If it already looks like an enforced path, return content/enforced path
          return `/content/enforced/${fid}`;
        }
      }
      return u.href;
    } catch {
      return raw;
    }
  }

  /* ---------- Tab Crawler ---------- */
  // queue of URLs to open in new tabs for scraping
  const tabQueue = [];
  const openedTabs = new Set();
  const tabResults = {}; // url -> array of found file urls
  let aborted = false;

  // Max concurrent tabs
  const MAX_TABS = 4;

  // Open a visible tab and scrape once loaded. Return array of downloadable urls found.
  async function openTabAndScrape(url, timeoutMs = 20_000, allowAutoclose = true) {
    if (!url) return [];
    appendLog(`TabCrawler: opening ${url}`);
    let win;
    try {
      // Open with noopener/noreferrer to avoid some cross-window leaks; still accessible in most browsers
      win = window.open(url, "_blank");
    } catch (e) {
      appendLog("Tab open failed (popup blocked).");
      return [];
    }
    if (!win) {
      appendLog("Tab open blocked (window.open returned null).");
      return [];
    }
    openedTabs.add(win);

    // Wait for load via polling on document.readyState, with timeout
    const start = Date.now();
    const pollInterval = 200;
    let success = false;
    while (Date.now() - start < timeoutMs) {
      try {
        // If window closed externally
        if (win.closed) { appendLog("Tab closed before load."); break; }
        const ready = win.document && win.document.readyState;
        if (ready === "complete" || ready === "interactive") { success = true; break; }
      } catch (e) {
        // Access denied until same-origin; but Brightspace pages opened in same origin should be accessible
      }
      await sleep(pollInterval);
    }

    const found = new Set();
    if (success) {
      appendLog(`TabCrawler: loaded ${url}`);
      try {
        // Scrape same-origin DOM
        const doc = win.document;
        // Add also any text-based quicklink in innerHTML (sometimes links are embedded)
        const nodes = doc.querySelectorAll("a[href], source[src], iframe[src], embed[src]");
        nodes.forEach(n => {
          const raw = n.getAttribute("href") || n.getAttribute("src") || "";
          const u = abs(raw);
          if (!u) return;
          if (looksFile(u)) found.add(u);
          else if (/quickLink.*fileId=/i.test(u)) {
            const fid = decodeURIComponent((u.match(/fileId=([^&]+)/i) || [,""])[1] || "");
            if (fid) found.add(`/content/enforced/${fid}`);
          } else if (looksHtmlLike(u)) {
            // find candidate direct download links in the page text (some pages include direct enforced links)
            try {
              // attempt to find enforced style links inside attributes or text nodes
              if (/\/content\/enforced\//i.test(u)) found.add(u);
            } catch {}
          }
        });

        // Some Brightspace pages embed quickLink launcher buttons that perform XHR to generate a download URL.
        // Attempt to introspect inline scripts for /content/enforced/ patterns
        try {
          const scripts = Array.from(doc.scripts || []);
          for (const s of scripts) {
            if (!s.innerText) continue;
            const matches = s.innerText.match(/\/content\/enforced\/[A-Za-z0-9%_\-\/\.]+/g);
            if (matches) matches.forEach(m => found.add(abs(m)));
          }
        } catch (e) { /* ignore */ }
      } catch (e) {
        appendLog(`TabCrawler: scraping blocked for ${url}`);
      }
    } else {
      appendLog(`TabCrawler: timed out waiting for ${url}`);
    }

    // Optionally close tab
    if (allowAutoclose && !win.closed) {
      try { win.close(); } catch(e) { /* ignore */ }
    }
    openedTabs.delete(win);
    const arr = [...found];
    tabResults[url] = arr;
    appendLog(`TabCrawler: found ${arr.length} items in ${url}`);
    return arr;
  }

  // Conveyor to run tab tasks limited concurrency
  async function runTabQueue(concurrency = MAX_TABS) {
    const results = {};
    let idx = 0;
    async function worker() {
      while (idx < tabQueue.length && !aborted) {
        const i = idx++;
        const job = tabQueue[i];
        try {
          const res = await openTabAndScrape(job.url, job.timeout || 20000, job.autoclose);
          results[job.url] = res;
        } catch (e) {
          appendLog("TabCrawler job error: " + (e && e.message || e));
          results[job.url] = [];
        }
        await sleep(200); // small gap between tab opens
      }
    }
    await Promise.all(Array(concurrency).fill(0).map(worker));
    return results;
  }

  /* ---------- Low-level fetcher (handles content-disposition) ---------- */
  async function fetchBlobWithName(url, headers = {}, timeoutMs = 20000) {
    try {
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), timeoutMs);
      const resp = await fetch(url, { credentials: "same-origin", headers, signal: controller.signal, redirect: "follow" });
      clearTimeout(id);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const contentDisposition = resp.headers.get("Content-Disposition") || "";
      let filename = "";
      const m = contentDisposition.match(/filename\*?=(?:UTF-8'')?["']?([^;"']+)["']?/i);
      if (m && m[1]) {
        filename = decodeURIComponent(m[1].replace(/["']/g, ""));
      } else {
        // fallback to URL path
        try {
          const u = new URL(resp.url || url, location.href);
          filename = decodeURIComponent((u.pathname.split("/").pop() || "file"));
        } catch { filename = "file"; }
      }
      const blob = await resp.blob();
      return { blob, filename, finalUrl: resp.url || url, status: resp.status };
    } catch (e) {
      return { error: e.message || String(e) };
    }
  }

  /* ---------- Main Download Logic ---------- */
  const zip = new JSZip();
  const skipped = [];
  const downloadedSet = new Set();

  // Helper: Determine if extension is allowed by UI checkboxes
  function allowedByExt(u) {
    const checked = Array.from(exDiv.querySelectorAll("input[data-ext]:checked")).map(i => i.dataset.ext.toLowerCase());
    const e = (extOf(u) || "").toLowerCase();
    if (!checked.length) return true;
    if (e && checked.includes(e)) return true;
    // sometimes links lack extension but are enforced content; allow and attempt fetch
    if (!e && isBrightspaceInternal(u)) return true;
    return false;
  }

  // Prepare list of candidate targets per topic
  const candidates = []; // {title, modulePath, originalUrl, type}
  for (const t of topics) {
    const item = t.Topic || t;
    const rawUrl = item.Url || item.UrlString || item.href || item.Href || "";
    const resolved = resolveQuickLinkUrl(rawUrl);
    const title = item.Title || item.TitleText || item.TitleString || (typeof item === "string" ? item : "") || "file";
    // If resolved is null use raw
    const finalUrl = resolved || rawUrl;
    if (!finalUrl) continue;
    candidates.push({ title, modulePath: t.ModulePath || ["Module"], url: finalUrl });
  }

  setStatus(`Prepared ${candidates.length} candidate links.`);

  /* ---------- Pre-scan: find which candidates are direct files, which need expansion ---------- */
  const directFiles = []; // download directly
  const expansions = []; // need deep scan or tab crawl
  for (const c of candidates) {
    const u = c.url;
    if (looksFile(u)) {
      directFiles.push(c);
    } else if (/quickLink.*fileId=/i.test(u) || /\/content\/enforced\//i.test(u)) {
      // quicklink-ish or enforced content might be direct downloadable (try)
      directFiles.push(c);
    } else if (looksHtmlLike(u) || isBrightspaceInternal(u)) {
      expansions.push(c);
    } else {
      // external - treat as shortcut
      skipped.push({ ...c, reason: "external_not_downloaded" });
    }
  }

  appendLog(`${directFiles.length} direct candidates, ${expansions.length} pages to expand via tab crawler.`);

  /* ---------- If expansions exist: prompt user about tabs ---------- */
  const allowTabs = expansions.length > 0;
  let doTabs = false;
  if (allowTabs) {
    const proceed = confirm(`Detected ${expansions.length} internal page instances that require opening new tabs. If you proceed, you will see your screen flicker with new tabs opening and closing. This is not a virus; this bypasses Brightspace dynamic loading. Continue?`);
    if (!proceed) {
      appendLog("User denied opening tabs. Expansion links will be saved as shortcuts.");
      // convert expansions into shortcuts
      expansions.forEach(x => skipped.push({ ...x, reason: "user_declined_tabs" }));
    } else {
      doTabs = true;
    }
  }

  /* ---------- If user allowed tabs, populate tabQueue with URLs to crawl ---------- */
  if (doTabs) {
    for (const e of expansions) {
      // For each expansion, push job to queue. We'll try to open the url and let TabCrawler extract downloadable links inside.
      tabQueue.push({ url: e.url, timeout: 22000, autoclose: document.getElementById("bsx-autoclose").checked });
    }
    appendLog(`Starting TabCrawler with ${tabQueue.length} jobs (concurrency ${MAX_TABS}).`);
    setStatus("Opening tabs to expand internal pages...");
    const tabResultsAll = await runTabQueue(MAX_TABS);
    // Now tabResultsAll contains arrays of discovered file urls per expansion url
    // Convert them to further direct file tasks
    for (const [srcUrl, arr] of Object.entries(tabResultsAll)) {
      if (!arr || !arr.length) {
        appendLog(`No downloadable links found in ${srcUrl}; will create shortcuts.`);
        const original = expansions.find(x => x.url === srcUrl);
        if (original) skipped.push({ ...original, reason: "no_links_found" });
        continue;
      }
      // Add each discovered url as a directFile candidate, preserving module path/title
      for (const foundUrl of arr) {
        // ensure not already captured via directFiles
        if (downloadedSet.has(foundUrl)) continue;
        directFiles.push({ title: `from_${(new URL(srcUrl)).pathname.split("/").pop()}`, modulePath: ["Expanded"], url: foundUrl });
      }
    }
  }

  /* ---------- Final download pass ---------- */
  setStatus("Starting downloads...");
  let totalToProcess = directFiles.length;
  let doneCount = 0;
  countEl.textContent = `${doneCount}/${totalToProcess} downloaded`;
  barEl.style.width = "0%";

  // Limit parallel network fetch concurrency
  const FETCH_CONCURRENCY = 6;
  let index = 0;
  async function workerDownload() {
    while (index < directFiles.length && !aborted) {
      const i = index++;
      const entry = directFiles[i];
      const targetUrl = entry.url;
      if (!allowedByExt(targetUrl)) {
        appendLog(`Skipping (ext filter): ${targetUrl}`);
        skipped.push({ ...entry, reason: "ext_filter" });
        doneCount++;
        countEl.textContent = `${doneCount}/${totalToProcess} downloaded`;
        barEl.style.width = ((doneCount / (totalToProcess || 1)) * 100).toFixed(1) + "%";
        continue;
      }
      try {
        appendLog(`Fetching: ${targetUrl}`);
        // If quickLink or enforced path may redirect; try fetchBlobWithName
        const res = await fetchBlobWithName(targetUrl, {}, 20000);
        if (res && res.blob && !res.error) {
          // determine filename
          let filename = res.filename || sanitizeFilename(entry.title || (new URL(res.finalUrl || targetUrl)).pathname.split("/").pop() || "file");
          // ensure extension
          const ext = extOf(filename) || extOf(res.finalUrl) || extOf(targetUrl) || ".bin";
          if (!filename.toLowerCase().endsWith(ext)) filename = filename + ext;
          // avoid duplicates
          let safeName = [...entry.modulePath, filename].map(sanitizeFilename).join("/") ;
          // ensure unique in zip
          let suffix = 1;
          let candidate = safeName;
          while (zip.file(candidate) || downloadedSet.has(res.finalUrl)) {
            candidate = safeName.replace(/(\.[^/.]+)$/, `_${suffix}$1`);
            suffix++;
          }
          zip.file(candidate, res.blob);
          downloadedSet.add(res.finalUrl || targetUrl);
          appendLog(`Added to zip: ${candidate}`);
        } else {
          appendLog(`Failed to fetch ${targetUrl}: ${res && res.error || "unknown"}`);
          // create shortcut fallback
          const dir = [...entry.modulePath].map(sanitizeFilename).join("/") + "/";
          const name = sanitizeFilename(entry.title || targetUrl);
          zip.file(`${dir}${name}.url`, `[InternetShortcut]\nURL=${targetUrl}\n`);
          skipped.push({ ...entry, reason: "fetch_failed", detail: res && res.error });
        }
      } catch (e) {
        appendLog(`Download error: ${e && e.message || e}`);
        const dir = [...entry.modulePath].map(sanitizeFilename).join("/") + "/";
        const name = sanitizeFilename(entry.title || targetUrl);
        zip.file(`${dir}${name}.url`, `[InternetShortcut]\nURL=${targetUrl}\n`);
        skipped.push({ ...entry, reason: "exception", detail: e && e.message });
      }
      doneCount++;
      countEl.textContent = `${doneCount}/${totalToProcess} downloaded`;
      barEl.style.width = ((doneCount / (totalToProcess || 1)) * 100).toFixed(1) + "%";
      await sleep(80);
    }
  }

  // start workers
  const workers = Array(FETCH_CONCURRENCY).fill(0).map(() => workerDownload());
  await Promise.all(workers);

  setStatus("Packaging ZIP...");

  // Smart name
  const tag = humanDateTag();
  const courseLabelName = courseName || `Course${courseId}`;
  const zipName = `Brightspace_${courseLabelName}_${tag}.zip`;

  try {
    const blob = await zip.generateAsync({ type: "blob" });
    saveAs(blob, zipName);
    setStatus(`Done: ${downloadedSet.size} files added, ${skipped.length} skipped. Saved as ${zipName}`);
    appendLog("All done.");
  } catch (e) {
    setStatus("Failed to generate ZIP: " + (e && e.message || e));
    appendLog("ZIP generation error: " + (e && e.message || e));
  }

  // Show skipped details on button
  document.getElementById("bsx-show").onclick = () => {
    if (!skipped.length) return alert("No skipped items.");
    const out = skipped.map(s => {
      return `${s.title || s.Topic?.Title || "unknown"} -> ${s.url || s.Topic?.Url || ""} ${s.reason ? ` [${s.reason}]` : ""} ${s.detail ? ` - ${s.detail}` : ""}`;
    }).join("\n\n");
    // show in a prompt-style box (too large for alert)
    const w = window.open("", "_blank", "noopener,noreferrer,width=800,height=600");
    w.document.title = "Brightspace Scraper - Skipped Items";
    w.document.body.style.fontFamily = "system-ui,Segoe UI,Roboto,Arial";
    w.document.body.innerHTML = `<pre style="white-space:pre-wrap">${out.replace(/</g,"&lt;")}</pre>`;
  };

  // Cleanup: close any still-opened tabs if autoclose is true/allowed
  try {
    if (openedTabs.size) {
      appendLog("Cleaning up opened tabs...");
      for (const t of Array.from(openedTabs)) {
        try { if (!t.closed) t.close(); } catch (e) { /* ignore */ }
      }
      openedTabs.clear();
    }
  } catch (e) { /* ignore */ }

  // Reset active flag after a short delay so user sees status
  setTimeout(() => {
    window.__bs_scraper_active = false;
  }, 1500);

})();
