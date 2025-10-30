/* ===========================================================
   Brightspace_Scraper v3.9.5 "TabCrawler+SafePrompt"
   by Zac Matthias and Partly ChatGPT 💞 | https://chaotic-mess.github.io/My-Code-/
   Single-file script for console/bookmarklet use.
   Full detection: API ToC, Lessons, Smart-Curriculum, QuickLinks.
   Adds controlled popup/tab crawling + robust queueing.
   =========================================================== */
(async () => {
  if (window.__bs_scraper_active) {
    alert("Brightspace Scraper already running.");
    return;
  }
  window.__bs_scraper_active = true;

  /* ---------- tiny utilities ---------- */
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const abs = u => { try { return new URL(u, location.href).href; } catch { return null; } };
  const san = s => (s || "").replace(/[<>:"/\\|?*]+/g, "_").trim();
  const extOf = u => { const m = u && u.match(/\.[a-z0-9]{2,6}(?:$|\?)/i); return m ? m[0].toLowerCase() : ""; };
  const looksFile = u => !!u && (/\.(pdf|mp4|m4v|mov|mp3|docx?|pptx?|xlsx?|zip|txt|csv|rtf|md|epub)(?:[?#].*)?$/i.test(u)
    || /\/content\/enforced\//i.test(u)
    || /\/d2l\/common\/viewFile\.d2l/i.test(u)
    || /type=coursefile/i.test(u));
  const tryJSON = async (r) => { try { return await r.json(); } catch { return null; } };

  /* ---------- detect course ---------- */
  const pathMatch = location.pathname.match(/\/d2l\/le\/(?:content|lessons|home)\/(\d+)/);
  const courseId = pathMatch && pathMatch[1];
  const isLessons = /\/d2l\/le\/lessons\//.test(location.pathname);

  if (!courseId) {
    alert("Open a course home, Content, or Lessons page first.");
    window.__bs_scraper_active = false;
    return;
  }

  /* ---------- UI Overlay (kept visually consistent) ---------- */
  const ui = document.createElement("div");
  ui.id = "bs-scraper-ui-root";
  ui.style = `
    position:fixed;right:16px;bottom:16px;z-index:999999;
    background:#1e1e1e;color:#eee;padding:14px 16px;border-radius:10px;
    font:14px/1.4 system-ui,Segoe UI,Roboto,Arial;box-shadow:0 8px 20px rgba(0,0,0,.35);
    width:360px;max-height:90vh;overflow-y:auto;
  `;
  ui.innerHTML = `
    <b style="font-size:15px">Brightspace Scraper v3.9.5</b>
    <div id="bs-course" style="margin:6px 0;color:#9ca3af">Detecting course…</div>
    <div id="bs-status" style="margin:8px 0;color:#ccc">Idle</div>
    <div id="bs-exts" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px"></div>
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
      <button id="bs-close" style="background:#333;color:#ccc;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Close</button>
    </div>
    <div id="bs-log" style="margin-top:8px;color:#9b9b9b;font-size:12px;max-height:120px;overflow:auto"></div>
  `;
  document.body.appendChild(ui);

  const S = t => ui.querySelector("#bs-status").textContent = t;
  const B = ui.querySelector("#bs-bar");
  const C = ui.querySelector("#bs-count");
  const LOG = msg => {
    const el = ui.querySelector("#bs-log");
    const p = document.createElement("div");
    p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    el.appendChild(p);
    el.scrollTop = el.scrollHeight;
    console.log("[BScraper]", msg);
  };

  ui.querySelector("#bs-close").onclick = () => { ui.remove(); window.__bs_scraper_active = false; };

  /* ---------- load zip libs ---------- */
  const loadScript = src => new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) return res();
    const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s);
  });
  await loadScript("https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js");
  await loadScript("https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js");
  const zip = new JSZip();

  /* ---------- file type toggles ---------- */
  const fileTypes = [".pdf",".mp4",".mp3",".m4v",".docx",".pptx",".xlsx",".zip",".txt"];
  const exDiv = ui.querySelector("#bs-exts");
  fileTypes.forEach(x => {
    const L = document.createElement("label");
    L.style = "font-size:12px;color:#d1d5db";
    L.innerHTML = `<input type="checkbox" data-ext="${x}" checked> ${x}`;
    exDiv.appendChild(L);
  });

  /* ---------- ToC fetch attempt ---------- */
  S("Fetching Table of Contents…");
  let toc = null, courseName = "";
  try {
    const r = await fetch(`/d2l/api/le/1.68/${courseId}/content/toc`, { credentials: "same-origin" });
    if (r.ok) {
      toc = await r.json();
      courseName = san(toc.Title || "") || courseName;
      LOG("ToC fetched via API.");
    } else {
      LOG("ToC API responded non-OK.");
    }
  } catch (e) {
    LOG("ToC API fetch failed.");
  }
  ui.querySelector("#bs-course").textContent = courseName || `Course ID: ${courseId}`;

  /* ---------- Smart Curriculum iframe extraction ---------- */
  async function getSmartCurrLinks() {
    const iframe = document.querySelector('iframe[src*="smart-curriculum"]');
    if (!iframe) return null;
    try {
      const doc = iframe.contentDocument || iframe.contentWindow.document;
      // wait a bit if it's a React app (best-effort)
      await sleep(250);
      const links = [];
      doc.querySelectorAll('a[href], source[src], iframe[src], embed[src]').forEach(n => {
        const u = abs(n.getAttribute("href") || n.getAttribute("src") || "");
        if (!u) return;
        if (looksFile(u)) links.push({ Title: n.textContent.trim() || n.getAttribute("title") || "file", Url: u });
        else if (/quickLink.*fileId=/i.test(u)) {
          links.push({ Title: n.textContent.trim() || "QuickLink File", Url: u });
        }
      });
      if (links.length) {
        LOG("Extracted links from Smart Curriculum iframe.");
        return { Modules: [{ Title: "SmartCurriculum", Topics: links }] };
      }
      return null;
    } catch (e) {
      LOG("SmartCurriculum access blocked or failed.");
      return null;
    }
  }

  /* ---------- DOM fallback scrapers ---------- */
  function parseQuickLinkUrl(url) {
    try {
      const u = new URL(url, location.href);
      const q = u.searchParams;
      const fileId = q.get("fileId") || q.get("id");
      return fileId ? decodeURIComponent(fileId) : null;
    } catch { return null; }
  }

  function scrapeDomLinksFromPage() {
    const links = [];
    document.querySelectorAll("a[href], source[src], iframe[src], embed[src]").forEach(el => {
      const hrefRaw = el.getAttribute("href") || el.getAttribute("src");
      const href = abs(hrefRaw);
      if (!href) return;
      if (looksFile(href)) {
        links.push({ Title: (el.textContent || el.title || href).trim().slice(0,120) || "file", Url: href });
      } else {
        // catch quickLink -> coursefile
        if (/quickLink.*fileId=/i.test(href)) {
          // convert to enforced path if possible
          const fileId = parseQuickLinkUrl(href);
          if (fileId) {
            // Brightspace enforced path often uses /content/enforced/<path>
            // Build a best-effort enforced URL using courseId + fileId
            const enforced = `/content/enforced/${courseId}-${fileId}`; // fallback placeholder; deep-scan will follow correctly
            links.push({ Title: (el.textContent || "QuickLink File").trim(), Url: href, quickFileId: fileId });
          } else {
            links.push({ Title: (el.textContent || "QuickLink").trim(), Url: href });
          }
        } else {
          // generic HTML link — include for deep scan if desired
          links.push({ Title: (el.textContent || el.title || href).trim(), Url: href });
        }
      }
    });
    return { Modules: [{ Title: "Page Links", Topics: links }] };
  }

  /* ---------- decide initial toc to use ---------- */
  if (!toc || !toc.Modules?.length || isLessons) {
    S("No ToC or Lessons layout detected — scanning DOM + SmartCurriculum…");
    const smart = await getSmartCurrLinks();
    toc = smart || scrapeDomLinksFromPage();
  }

  /* ---------- collect topics (flatten) ---------- */
  const topics = [];
  const walk = m => {
    (m.Topics || []).forEach(t => topics.push(t));
    (m.Modules || []).forEach(sub => walk(sub));
  };
  (toc.Modules || []).forEach(walk);
  S(`Found ${topics.length} topics.`);
  LOG(`Initial topics gathered: ${topics.length}`);

  /* ---------- deep scan / crawl helpers ---------- */
  // fetch with timeout + retries
  async function fetchWithTimeout(url, opts = {}, timeout = 15000, retries = 1) {
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeout);
        const r = await fetch(url, { ...opts, credentials: 'same-origin', signal: controller.signal });
        clearTimeout(id);
        return r;
      } catch (e) {
        if (attempt === retries) throw e;
        await sleep(500 + attempt * 200);
      }
    }
  }

  // deepScan: returns array of downloadable URLs found under this page
  async function deepScan(url, depth = 0, visited = new Set(), openTabsPolicy = { allowPopups: false, uiOpenTabs: [] }) {
    if (!url || visited.has(url) || depth > 4) return [];
    visited.add(url);

    LOG(`deepScan(${depth}): ${url}`);
    // try to fetch HTML as same-origin
    try {
      const r = await fetchWithTimeout(url, {}, 12000, 1);
      if (!r.ok) {
        LOG(`deepScan: fetch failed ${r.status} for ${url}`);
        return [];
      }
      const html = await r.text();
      // parse and extract
      const doc = new DOMParser().parseFromString(html, "text/html");
      const found = new Set();

      // Save a copy of the HTML for inspection
      try {
        zip.file(`html_pages/page_${depth}_${Date.now()}.html`, html);
      } catch (e) {
        /* ignore zip write failures */
      }

      // find direct file links and quickLinks
      doc.querySelectorAll("a[href], source[src], iframe[src], embed[src]").forEach(n => {
        const raw = n.getAttribute("href") || n.getAttribute("src") || "";
        const u = abs(raw);
        if (!u) return;
        if (looksFile(u)) found.add(u);
        else if (/quickLink.*fileId=/i.test(u)) {
          // quickLink might redirect to file — attempt to convert to direct quicklink endpoint
          const fileId = parseQuickLinkUrl(u);
          if (fileId) {
            // create the quickLink dialog url and also attempt to generate an enforced content URL
            const quickUrl = u;
            found.add(quickUrl); // keep quickUrl so caller can follow
          } else {
            found.add(u);
          }
        } else if (/\/d2l\/le\/lessons|viewContent|\/content\//i.test(u)) {
          // recursively follow pages (but don't flood)
          // Use thenable recursion to avoid blocking the rest of the parsing
          // We'll await the nested promises after enumerating all nodes.
          // Push a marker so we will traverse recursively below.
          found.add(u);
        }
      });

      // For every candidate that is not a direct file, recursively crawl
      const results = [...found];
      const final = new Set();
      for (const candidate of results) {
        if (!candidate) continue;
        if (looksFile(candidate)) final.add(candidate);
        else {
          // don't recurse on external origins to avoid CORS complexity
          const sameOrigin = (() => {
            try { return new URL(candidate, location.href).origin === location.origin; } catch { return false; }
          })();
          if (sameOrigin) {
            const nested = await deepScan(candidate, depth + 1, visited, openTabsPolicy);
            nested.forEach(x => final.add(x));
          } else {
            // if it's quickLink pointing to same host but with parameters, add it
            if (/quickLink/i.test(candidate)) final.add(candidate);
          }
        }
      }
      return [...final];
    } catch (e) {
      LOG(`deepScan error: ${e.message}`);
      return [];
    }
  }

  /* ---------- precompute download targets (with optional deep scan) ---------- */
  async function computeTargets(topicsList, doDeep, allowPopups) {
    const targets = []; // {title, moduleTitle, url, reason}
    // helper to push unique
    const pushed = new Set();
    function pushIfNew(obj) {
      const key = `${obj.moduleTitle}::${obj.title}::${obj.url}`;
      if (!pushed.has(key)) {
        pushed.add(key);
        targets.push(obj);
      }
    }

    LOG(`Computing targets. Topics: ${topicsList.length}. DeepScan: ${doDeep ? "ON" : "OFF"}. AllowPopups: ${allowPopups}`);

    for (const t of topicsList) {
      // If the topic has the quickFileId, try to convert to a direct 'quickLink' fetchable URL
      let url = t.Url || t.url || t.UrlRaw || t.UrlRaw || '';
      const title = t.Title || t.title || (t.Name || 'file').slice(0,200);

      // If the URL is a quickLink dialog: prefer to include the quickLink URL (we will follow later)
      if (/quickLink.*fileId=/i.test(url)) {
        pushIfNew({ title, moduleTitle: t.Module || "Module", url, reason: "quicklink" });
        continue;
      }

      // If the URL looks like an enforced content path, add directly
      if (looksFile(url)) {
        pushIfNew({ title, moduleTitle: t.Module || "Module", url, reason: "direct" });
        continue;
      }

      // If deep scan enabled and URL is HTML-ish, run deepScan to harvest inner files
      if (doDeep && /\.html?/i.test(url) || doDeep && /viewContent|lessons|content\//i.test(url)) {
        // deep scan may return direct file URLs and quickLinks
        try {
          const found = await deepScan(url, 0, new Set(), { allowPopups: allowPopups });
          if (found && found.length) {
            for (const f of found) {
              // prefer real file URLs first
              if (looksFile(f)) pushIfNew({ title: `${title}`, moduleTitle: t.Module || "Module", url: f, reason: "deep" });
              else pushIfNew({ title: `${title}`, moduleTitle: t.Module || "Module", url: f, reason: "deep-quick" });
            }
            continue;
          } else {
            // no inner found — still add the original as a fallback (so a .url will be created)
            pushIfNew({ title, moduleTitle: t.Module || "Module", url, reason: "fallback" });
            continue;
          }
        } catch (e) {
          LOG("deepScan failed during computeTargets: " + e.message);
          pushIfNew({ title, moduleTitle: t.Module || "Module", url, reason: "fallback" });
        }
      } else {
        // default: add as-is
        pushIfNew({ title, moduleTitle: t.Module || "Module", url, reason: "raw" });
      }
    }
    return targets;
  }

  /* ---------- Tab opener helper ---------- */
  // Opens a list of URLs in background tabs (but visible) and returns handles.
  async function openTabs(urls, onOpen = () => {}) {
    const handles = [];
    for (const u of urls) {
      try {
        // open in new tab/window
        const w = window.open(u, "_blank");
        if (!w) {
          LOG("Popup was blocked by browser for " + u);
          handles.push({ url: u, window: null, blocked: true });
          continue;
        }
        handles.push({ url: u, window: w, blocked: false });
        onOpen(u, w);
        // small spacing to avoid overwhelming the browser
        await sleep(250);
      } catch (e) {
        LOG("openTabs error: " + e.message);
        handles.push({ url: u, window: null, blocked: true });
      }
    }
    return handles;
  }

  /* ---------- Download worker: fetch files, or create .url shortcuts ---------- */
  async function downloadWorker(tasks, concurrency = 4, uiUpdate = () => {}) {
    let index = 0, done = 0, failed = 0;
    const total = tasks.length;

    async function worker() {
      while (true) {
        const i = index++;
        if (i >= total) break;
        const t = tasks[i];
        uiUpdate({ index: i, total, title: t.title, status: "starting", done, failed });
        try {
          // if quicklink dialog url -> we try to fetch it and follow to real file
          if (/quickLink.*fileId=/i.test(t.url)) {
            // attempt to fetch quickLink endpoint and see content
            try {
              const r = await fetchWithTimeout(t.url, {}, 10000, 1);
              // if quickLink returns a redirect or file, follow final url from response.url
              if (r && (r.ok || r.status === 302 || r.status === 200)) {
                // if content-type is binary, just save it
                const ct = r.headers.get("content-type") || "";
                if (/application\/pdf|application\/octet-stream|application\/zip|image\//i.test(ct) || looksFile(r.url)) {
                  const blob = await r.blob();
                  const ext = extOf(r.url) || extOf(t.url) || ".bin";
                  zip.file(`${san(t.moduleTitle)}/${san(t.title)}${ext}`, blob);
                  done++;
                  uiUpdate({ done, failed });
                  continue;
                } else {
                  // fallback: parse body to find direct links
                  const html = await r.text();
                  const d = new DOMParser().parseFromString(html, "text/html");
                  const sublinks = [];
                  d.querySelectorAll("a[href], source[src]").forEach(n => {
                    const u = abs(n.getAttribute("href") || n.getAttribute("src") || "");
                    if (u) sublinks.push(u);
                  });
                  // find download-like link
                  const dl = sublinks.find(u => looksFile(u));
                  if (dl) {
                    const rr = await fetchWithTimeout(dl, {}, 12000, 1);
                    if (rr && rr.ok) {
                      const blob = await rr.blob();
                      const ext = extOf(dl) || ".bin";
                      zip.file(`${san(t.moduleTitle)}/${san(t.title)}${ext}`, blob);
                      done++;
                      uiUpdate({ done, failed });
                      continue;
                    }
                  }
                }
              }
              // if we reach here, quicklink couldn't be directly pulled — create shortcut
              const content = `[InternetShortcut]\nURL=${t.url}\n`;
              zip.file(`${san(t.moduleTitle)}/${san(t.title)}.url`, content);
              failed++;
              uiUpdate({ done, failed });
              continue;
            } catch (e) {
              // quicklink fetch failed; create shortcut
              const content = `[InternetShortcut]\nURL=${t.url}\n`;
              zip.file(`${san(t.moduleTitle)}/${san(t.title)}.url`, content);
              failed++;
              uiUpdate({ done, failed });
              continue;
            }
          }

          // general direct file
          const r = await fetchWithTimeout(t.url, {}, 15000, 1);
          if (!r || !r.ok) {
            // attempt to detect 403 vs blocked — create shortcut if can't fetch
            const content = `[InternetShortcut]\nURL=${t.url}\n`;
            zip.file(`${san(t.moduleTitle)}/${san(t.title)}.url`, content);
            failed++;
            uiUpdate({ done, failed });
            continue;
          }
          // if content-type indicates HTML (and not a binary), treat as HTML and try to extract a direct file
          const ct = r.headers.get("content-type") || "";
          if (ct.includes("text/html") && !looksFile(t.url)) {
            const body = await r.text();
            const d = new DOMParser().parseFromString(body, "text/html");
            const sublinks = [];
            d.querySelectorAll("a[href], source[src], iframe[src], embed[src]").forEach(n => {
              const u = abs(n.getAttribute("href") || n.getAttribute("src") || "");
              if (u) sublinks.push(u);
            });
            // pick the first download-looking item
            const dl = sublinks.find(u => looksFile(u));
            if (dl) {
              // fetch that
              try {
                const rr = await fetchWithTimeout(dl, {}, 15000, 1);
                if (rr && rr.ok) {
                  const blob = await rr.blob();
                  const ext = extOf(dl) || ".bin";
                  zip.file(`${san(t.moduleTitle)}/${san(t.title)}${ext}`, blob);
                  done++;
                  uiUpdate({ done, failed });
                  continue;
                }
              } catch (e) {
                // fallback to shortcut
                zip.file(`${san(t.moduleTitle)}/${san(t.title)}.url`, `[InternetShortcut]\nURL=${t.url}\n`);
                failed++;
                uiUpdate({ done, failed });
                continue;
              }
            } else {
              // no inner downloads: save HTML as snapshot
              try { zip.file(`${san(t.moduleTitle)}/${san(t.title)}.html`, body); } catch {}
              failed++;
              uiUpdate({ done, failed });
              continue;
            }
          }

          // otherwise it's a binary / file
          try {
            const blob = await r.blob();
            const ext = extOf(t.url) || ".bin";
            zip.file(`${san(t.moduleTitle)}/${san(t.title)}${ext}`, blob);
            done++;
            uiUpdate({ done, failed });
            continue;
          } catch (e) {
            // fallback: .url shortcut
            zip.file(`${san(t.moduleTitle)}/${san(t.title)}.url`, `[InternetShortcut]\nURL=${t.url}\n`);
            failed++;
            uiUpdate({ done, failed });
            continue;
          }
        } catch (err) {
          LOG(`Download error for ${t.url}: ${err.message || err}`);
          // create shortcut as safe fallback
          try { zip.file(`${san(t.moduleTitle)}/${san(t.title)}.url`, `[InternetShortcut]\nURL=${t.url}\n`); } catch {}
          failed++;
          uiUpdate({ done, failed });
        }
      }
    }

    // start workers
    await Promise.all(Array(concurrency).fill(0).map(worker));
    return { done, failed };
  }

  /* ---------- Show skipped helper ---------- */
  const skipped = [];
  ui.querySelector("#bs-show").onclick = () => {
    if (!skipped.length) return alert("No skipped items yet.");
    const txt = skipped.map(x => `${x.title || x.Title} → ${x.url || x.Url}`).join("\n");
    // copy to clipboard and present
    try { navigator.clipboard.writeText(txt); alert("Skipped list copied to clipboard. Showing list now."); } catch {}
    alert(txt);
  };

  /* ---------- main button handler (no autorun) ---------- */
  ui.querySelector("#bs-start").onclick = async () => {
    try {
      ui.querySelector("#bs-start").disabled = true;
      S("Preparing scan…");
      LOG("Start button clicked. Preparing targets...");

      const doDeep = ui.querySelector("#bs-deepscan").checked;
      const allowPopups = ui.querySelector("#bs-allowpopups").checked;

      // Build a simple normalized topic list to pass into computeTargets
      const normalizedTopics = topics.map(t => ({
        Title: t.Title || t.title || (t.Name || "topic"),
        Url: t.Url || t.url || t.UrlRaw || t.href || t.link || "",
        Module: (t.Module || t.moduleTitle || "Module")
      }));

      // Preflight: if quicklinks present, prompt user about opening tabs
      const hasQuicklinks = normalizedTopics.some(t => /quickLink.*fileId=/i.test(t.Url) || /quickLink/i.test(t.Url));
      if (hasQuicklinks && !allowPopups) {
        const proceed = confirm(
          "Detected QuickLink items that may need additional loading. " +
          "If you proceed with 'Allow Popups' checked, the script may open and close several tabs to force Brightspace to render the final resources. " +
          "This will cause brief tab flickering. Continue?"
        );
        if (!proceed) {
          S("User canceled tab-opening prompt. Running without popups.");
        } else {
          // user agreed; set the checkbox so we proceed
          ui.querySelector("#bs-allowpopups").checked = true;
        }
      }

      const finalAllowPopups = ui.querySelector("#bs-allowpopups").checked;

      // Compute targets (this may run deepScan)
      S("Computing targets (this may take a moment)...");
      const targets = await computeTargets(normalizedTopics, doDeep, finalAllowPopups);

      // If there are quicklinks and user allowed popups, open those quicklink pages in tabs
      // but only open the unique quicklink pages that are not direct file urls and are same-origin
      let quicklinkPages = [...new Set(targets.filter(t => /quickLink/i.test(t.url) && !looksFile(t.url)).map(t => t.url))];
      quicklinkPages = quicklinkPages.filter(u => {
        try { return new URL(u, location.href).origin === location.origin; } catch { return false; }
      });

      if (quicklinkPages.length && finalAllowPopups) {
        const confirmMsg = `Detected ${quicklinkPages.length} QuickLink pages that may need a real tab to initialize. ` +
          `If you continue, ${quicklinkPages.length} tabs will briefly open and be closed by the script. Continue?`;
        if (!confirm(confirmMsg)) {
          LOG("User declined opening quicklink tabs.");
        } else {
          S(`Opening ${quicklinkPages.length} helper tabs...`);
          LOG("Opening quicklink pages to prime server-side rendering.");
          const handles = await openTabs(quicklinkPages, (u, w) => LOG("Opened tab for " + u));
          // give them some time to finish loading
          await sleep(2000 + quicklinkPages.length * 300);
          // attempt to fetch each quicklink once to 'prime' the session & extract final URLs
          for (const q of quicklinkPages) {
            try {
              LOG("Priming quicklink by fetching: " + q);
              await fetchWithTimeout(q, {}, 8000, 1);
            } catch (e) {
              LOG("Priming fetch failed for " + q);
            }
          }
          // close opened windows
          for (const h of handles) {
            try { if (h.window && !h.window.closed) { h.window.close(); LOG("Closed tab for " + h.url); } } catch {}
          }
          await sleep(500);
        }
      }

      // Recompute targets after priming (deep-scan again if necessary) to pick up any resources that now exist
      S("Recomputing targets after priming quicklinks...");
      const recomputedTargets = await computeTargets(normalizedTopics, doDeep, finalAllowPopups);
      LOG(`Targets discovered: ${recomputedTargets.length}`);

      if (!recomputedTargets.length) {
        S("No download targets found.");
        ui.querySelector("#bs-start").disabled = false;
        return;
      }

      // Filter by selected extensions
      const allowedExts = [...exDiv.querySelectorAll("input:checked")].map(i => i.dataset.ext.toLowerCase());
      const filtered = recomputedTargets.filter(t => {
        const e = extOf(t.url).toLowerCase();
        if (e && allowedExts.length) return allowedExts.includes(e);
        // if no extension or unknown, still keep to try deep-resolution
        return true;
      });

      // Build tasks for the downloader
      const tasks = filtered.map(t => ({
        title: t.title || t.Title,
        moduleTitle: t.moduleTitle || t.Module || "Module",
        url: t.url || t.Url,
        reason: t.reason || "auto"
      }));

      // UI prepare
      let done = 0, failed = 0;
      C.textContent = `0/${tasks.length} downloaded`;
      B.style.width = "0%";
      S(`Downloading ${tasks.length} items...`);
      LOG(`Starting download of ${tasks.length} items.`);

      // uiUpdate callback
      const uiUpdate = ({ done: d = done, failed: f = failed } = {}) => {
        done = d; failed = f;
        const total = tasks.length;
        C.textContent = `${done}/${total} downloaded (failed ${failed})`;
        B.style.width = (((done + failed) / (total || 1)) * 100).toFixed(1) + "%";
      };

      // run downloader with concurrency 6
      const res = await downloadWorker(tasks, 6, ({ done: d, failed: f } = {}) => uiUpdate({ done: d, failed: f }));

      // write ZIP and finalize
      S("Building ZIP...");
      LOG("Generating zip...");
      const date = new Date();
      const tag = `(${String(date.getFullYear()).slice(2)}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")})`;
      const filename = `Brightspace_${(courseName || courseId)}_${tag}.zip`;
      try {
        const blob = await zip.generateAsync({ type: "blob" });
        saveAs(blob, filename);
        S(`Done: ${res.done} files, ${res.failed} skipped.`);
        LOG(`Saved ${filename} with ${res.done} files and ${res.failed} skipped.`);
      } catch (e) {
        S("ZIP generation failed: " + (e.message || e));
        LOG("ZIP generation error: " + e.message);
      }

    } catch (e) {
      LOG("Fatal error in main: " + (e.message || e));
      S("Error: " + (e.message || e));
    } finally {
      ui.querySelector("#bs-start").disabled = false;
      window.__bs_scraper_active = false;
    }
  };

  LOG("Ready. Click Scan & Download to begin.");
  S("Ready.");
})();
