/* ===========================================================
   Brightspace_Scraper v3.0
   by chaotic-mess | https://chaotic-mess.github.io/My-Code-/
   Dark-mode downloader for Brightspace (PDFs, MP4s, DOCXs, etc.)
   Auto-detects course, deep-scans HTML topics, and builds a ZIP.
   =========================================================== */
(async () => {
  if (window.__bs_scraper_active) {
    alert("Brightspace Scraper already running.");
    return;
  }
  window.__bs_scraper_active = true;

  /* ---------- Utility Helpers ---------- */
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const abs = u => { try { return new URL(u, location.href).href; } catch { return null; } };
  const san = s => (s || "").replace(/[<>:"/\\|?*]+/g, "_").trim();
  const extOf = u => { const m = u && u.match(/\.[a-z0-9]{2,5}(?:$|\?)/i); return m ? m[0].toLowerCase() : ""; };
  const looksFile = u => /\.(pdf|mp4|docx?|pptx?|xlsx?|zip|txt|csv|rtf|md|epub)(?:[?#].*)?$/i.test(u) ||
                         /\/content\/enforced\//i.test(u) || /\/d2l\/common\/viewFile\.d2l/i.test(u);

  /* ---------- Course Detection ---------- */
  const m = location.pathname.match(/\/d2l\/(?:le\/content\/|lessons\/|home\/)(\d+)/);
  const courseId = m && m[1];
  if (!courseId) {
    alert("Open a course home, Content, or Lessons page first.");
    window.__bs_scraper_active = false;
    return;
  }

  /* ---------- UI Overlay ---------- */
  const ui = document.createElement("div");
  ui.style = `
    position:fixed;right:16px;bottom:16px;z-index:999999;
    background:#1e1e1e;color:#eee;padding:14px 16px;border-radius:10px;
    font:14px/1.4 system-ui,Segoe UI,Roboto,Arial;box-shadow:0 8px 20px rgba(0,0,0,.35);
    width:360px;max-height:90vh;overflow-y:auto;
  `;
  ui.innerHTML = `
    <b style="font-size:15px">Brightspace Scraper v3.0</b>
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
  const S = t => ui.querySelector("#bs-status").textContent = t;
  const B = ui.querySelector("#bs-bar"), C = ui.querySelector("#bs-count");

  ui.querySelector("#bs-close").onclick = () => { ui.remove(); window.__bs_scraper_active = false; };

  /* ---------- Load JSZip + FileSaver ---------- */
  const load = src => new Promise(r => { const s=document.createElement("script"); s.src=src; s.onload=r; document.head.appendChild(s); });
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

  /* ---------- Attempt ToC Fetch ---------- */
  S("Fetching Table of Contents…");
  let toc = null, courseName = "";
  try {
    const r = await fetch(`/d2l/api/le/1.68/${courseId}/content/toc`, { credentials: "same-origin" });
    if (r.ok) {
      toc = await r.json();
      courseName = san(toc.Title || "");
    }
  } catch {}
  ui.querySelector("#bs-course").textContent = courseName || `Course ID: ${courseId}`;

  /* ---------- Fallback: DOM scrape ---------- */
  const scrapeDomLinks = () => {
    const links = [];
    document.querySelectorAll("a[href]").forEach(a => {
      const href = abs(a.getAttribute("href"));
      if (!href) return;
      if (looksFile(href))
        links.push({ Title: a.textContent.trim() || "file", Url: href });
    });
    return { Modules:[{ Title:"Loose Files", Topics:links }] };
  };
  if (!toc || !toc.Modules?.length) {
    S("No ToC found; scanning page links…");
    toc = scrapeDomLinks();
  }

  /* ---------- Collect Topics ---------- */
  const topics = [];
  const walk = m => {
    (m.Topics || []).forEach(t => topics.push(t));
    (m.Modules || []).forEach(walk);
  };
  (toc.Modules || []).forEach(walk);
  S(`Found ${topics.length} topics.`);

  /* ---------- Deep Scan HTML pages ---------- */
  const deepScan = async (url) => {
    try {
      const r = await fetch(url, { credentials:"same-origin" });
      if (!r.ok) return [];
      const html = await r.text();
      const d = new DOMParser().parseFromString(html, "text/html");
      const found = [];
      d.querySelectorAll("a[href],source[src],iframe[src],embed[src]").forEach(n => {
        const u = abs(n.getAttribute("href") || n.getAttribute("src") || "");
        if (u && looksFile(u)) found.push(u);
      });
      return found;
    } catch { return []; }
  };

  /* ---------- Button Logic ---------- */
  const skipped = [];
  ui.querySelector("#bs-show").onclick = () => {
    if (!skipped.length) return alert("No skipped items yet.");
    alert(skipped.map(x => x.Title + " → " + (x.Url || "no URL")).join("\n"));
  };

  ui.querySelector("#bs-start").onclick = async () => {
    ui.querySelector("#bs-start").disabled = true;
    const allow = [...exDiv.querySelectorAll("input:checked")].map(c=>c.dataset.ext);
    const want = t => { const e=extOf(t.Url||""); return !allow.length || allow.includes(e); };
    const wanted = topics.filter(want);
    const doDeep = ui.querySelector("#bs-deepscan").checked;

    let done = 0;
    C.textContent = `0/${wanted.length} downloaded`;
    S("Downloading…");

    const addURLShortcut = (dir, title, link) => {
      const content = `[InternetShortcut]\nURL=${link}\n`;
      zip.file(`${dir}${san(title)}.url`, content);
    };

    const add = async (mods, pre="") => {
      for (const m of mods || []) {
        const dir = pre + san(m.Title || "Module") + "/";
        for (const t of m.Topics || []) {
          if (!t.Url) { skipped.push(t); continue; }
          const u = t.Url;
          if (!want(t)) { skipped.push(t); continue; }

          let targets = [u];
          if (doDeep && /\.html?/i.test(u)) {
            const extra = await deepScan(u);
            extra.forEach(x => targets.push(x));
          }

          for (const link of targets) {
            try {
              const r = await fetch(link, { credentials:"same-origin" });
              if (!r.ok || r.status===403) {
                addURLShortcut(dir, t.Title || "link", link);
                skipped.push({...t, Url:link, Type:"Shortcut"});
                continue;
              }
              const blob = await r.blob();
              const ext = extOf(link) || ".bin";
              zip.file(dir + san(t.Title || "file") + ext, blob);
              done++;
              B.style.width = ((done/(wanted.length||1))*100).toFixed(1) + "%";
              C.textContent = `${done}/${wanted.length} downloaded`;
            } catch {
              addURLShortcut(dir, t.Title || "link", link);
              skipped.push({...t, Url:link, Type:"Shortcut"});
            }
            await sleep(60);
          }
        }
        await add(m.Modules, dir);
      }
    };

    await add(toc.Modules);
    S("Building ZIP…");
    const blob = await zip.generateAsync({ type:"blob" });
    const name = `Brightspace_${courseName || courseId}.zip`;
    saveAs(blob, name);
    S(`Done: ${done} files, ${skipped.length} skipped.`);
    setTimeout(() => { ui.remove(); window.__bs_scraper_active=false; }, 7000);
  };
})();
