/* ===========================================================
   Moodle_Scraper V1.0 – "Hierarchical Organization"
   by chaotic-mess (Zac) + AI
   Downloads all course materials from Moodle with folder structure!
   =========================================================== */
(async () => {
  if (window.__moodle_scraper_active) { alert("Moodle Scraper already running."); return; }
  window.__moodle_scraper_active = true;

  /* ---------------- tiny utils ---------------- */
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const abs = (u) => { try { return new URL(u, location.href).href; } catch { return null; } };
  const san = (s) => (s || "").replace(/[<>:"/\\|?*]+/g, "_").trim().substring(0, 100);
  const extOf = (u) => { const m = u && u.match(/\.[a-z0-9]{2,6}(?:$|\?)/i); return m ? m[0].toLowerCase().split('?')[0] : ""; };
  const todayTag = () => { const d = new Date(); return `(${String(d.getFullYear()).slice(2)}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')})`; };

  const looksFile = (u) =>
    !!u && (/\.(pdf|mp4|m4v|mov|mp3|docx?|pptx?|xlsx?|zip|7z|rar|txt|csv|rtf|md|png|jpe?g|gif|svg)(?:[?#].*)?$/i.test(u)
      || /\/pluginfile\.php\//i.test(u)
      || /\/mod\/resource\/view\.php/i.test(u)
      || /\/mod\/folder\/view\.php/i.test(u));

  const isModPage = (u) =>
    /\/mod\/(page|book|label|url)\/view\.php/i.test(u);

  const getParam = (url, key) => { try { return new URL(url, location.href).searchParams.get(key); } catch { return null; } };

  function log(msg) { console.log("[MScraper]", msg); try {
    const el = document.querySelector("#ms-log"); if (!el) return;
    const d = document.createElement("div"); d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    el.appendChild(d); el.scrollTop = el.scrollHeight;
  } catch {} }

  /* ---------------- course detection ---------------- */
  const courseMatch = location.pathname.match(/\/course\/view\.php/);
  const courseId = new URLSearchParams(location.search).get('id');
  
  if (!courseMatch || !courseId) { 
    alert("Please open a Moodle course page (course/view.php?id=...)"); 
    window.__moodle_scraper_active = false; 
    return; 
  }

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
      background: repeating-linear-gradient(45deg,#ff6b35,#ff6b35 8px,#303030 8px,#303030 16px);
      padding:8px 12px;cursor:move;color:#000;display:flex;justify-content:space-between;align-items:center;">
      <b style="color:#FFFFFF">Moodle Scraper V1.0</b>
      <button id="ms-close" title="Close" style="border:0;background:#111;color:#eee;padding:4px 8px;border-radius:6px">✕</button>
    </div>
    <div style="padding:12px">
      <div id="ms-course" style="margin:6px 0;color:#9ca3af">Detecting course…</div>
      <div id="ms-status" style="margin:8px 0;color:#ccc">Idle</div>
      <div id="ms-exts" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px"></div>
      <label style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
        <input type="checkbox" id="ms-deepscan" checked> Deep Scan Pages (find embedded files)
      </label>
      <div style="height:6px;background:#333;border-radius:6px;overflow:hidden;margin-bottom:6px">
        <div id="ms-bar" style="height:6px;width:0;background:#ff6b35"></div>
      </div>
      <div id="ms-count" style="margin:6px 0;color:#bbb;font-size:12px">0/0 downloaded</div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px">
        <button id="ms-start" style="background:#ff6b35;color:#fff;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Scan & Download</button>
        <button id="ms-show" style="background:#333;color:#ccc;border:0;border-radius:6px;padding:6px 10px;cursor:pointer">Show Skipped</button>
      </div>
      <div id="ms-log" style="margin-top:8px;color:#9b9b9b;font-size:12px;max-height:140px;overflow:auto;border-top:1px solid #2a2a2a;padding-top:8px"></div>
    </div>
  `;
  document.body.appendChild(ui);
  const S = (t) => ui.querySelector("#ms-status").textContent = t;
  const B = ui.querySelector("#ms-bar");
  const C = ui.querySelector("#ms-count");
  ui.querySelector("#ms-close").onclick = () => { ui.remove(); window.__moodle_scraper_active = false; };

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
  const exDiv = ui.querySelector("#ms-exts");
  types.forEach(x => {
    const L = document.createElement("label"); L.style="font-size:12px;color:#d1d5db";
    L.innerHTML = `<input type="checkbox" data-ext="${x}" checked> ${x}`;
    exDiv.appendChild(L);
  });
  const allowedSet = () => new Set([...exDiv.querySelectorAll("input:checked")].map(c => c.dataset.ext.toLowerCase()));

  /* ---------------- Extract course structure from DOM ---------------- */
  S("Scanning course structure…");
  
  const courseName = san(document.querySelector('.page-header-headings h1')?.textContent || 
                         document.querySelector('h1.h2')?.textContent || 
                         document.querySelector('.page-context-header h1')?.textContent ||
                         document.querySelector('h1')?.textContent || 
                         `Course_${courseId}`);
  
  ui.querySelector("#ms-course").textContent = courseName;
  log(`Course: ${courseName}`);

  const sections = [];
  
  // Modern Moodle can use various selectors for sections
  const sectionSelectors = [
    'li.section[id^="section-"]',           // Classic format
    'li[data-for="section"]',               // Data attribute format
    '.course-section',                      // Generic course section
    'div[role="region"][aria-label*="opic"]' // Accessibility-based
  ];
  
  let sectionElements = [];
  for (const selector of sectionSelectors) {
    sectionElements = [...document.querySelectorAll(selector)];
    if (sectionElements.length > 0) {
      log(`Using selector: ${selector}`);
      break;
    }
  }
  
  if (sectionElements.length === 0) {
    log("No sections found with standard selectors, scanning entire page...");
    // Fallback: scan the entire main content area
    const mainContent = document.querySelector('#region-main, .course-content, main, [role="main"]');
    if (mainContent) {
      sectionElements = [mainContent];
    }
  }
  
  sectionElements.forEach((sec, idx) => {
    const sectionTitle = san(
      sec.querySelector('.sectionname')?.textContent?.trim() || 
      sec.querySelector('h3')?.textContent?.trim() || 
      sec.querySelector('h2')?.textContent?.trim() ||
      sec.querySelector('[data-for="section_title"]')?.textContent?.trim() ||
      sec.getAttribute('aria-label') ||
      `Section_${idx + 1}`
    );
    
    const activities = [];
    
    // Multiple activity selectors for different Moodle versions
    const activitySelectors = [
      '.activity',
      '.activityinstance',
      'li.modtype_resource',
      'li[class*="modtype_"]',
      '.aalink',
      'a[href*="/mod/"]'
    ];
    
    const activityElements = new Set();
    
    activitySelectors.forEach(selector => {
      sec.querySelectorAll(selector).forEach(el => {
        // Find the actual link element
        const link = el.tagName === 'A' ? el : el.querySelector('a');
        if (link) activityElements.add(link);
      });
    });
    
    activityElements.forEach(link => {
      const href = abs(link.getAttribute('href'));
      if (!href) return;
      
      // Skip course/section navigation links
      if (/\/course\/(view|edit)\.php/i.test(href) || href.includes('#section-')) return;
      
      const title = san(
        link.querySelector('.instancename')?.textContent?.trim() ||
        link.textContent?.trim() || 
        link.getAttribute('title')?.trim() || 
        link.getAttribute('aria-label')?.trim() ||
        'Resource'
      );
      
      // Try to determine activity type from URL or class
      const actType = href.match(/\/mod\/(\w+)\//)?.[1] || 
                     link.closest('[class*="modtype_"]')?.className.match(/modtype_(\w+)/)?.[1] ||
                     'resource';
      
      activities.push({
        title,
        url: href,
        type: actType
      });
    });
    
    if (activities.length > 0) {
      sections.push({
        title: sectionTitle,
        activities
      });
      log(`Section "${sectionTitle}": ${activities.length} activities`);
    }
  });

  log(`Found ${sections.length} sections with ${sections.reduce((sum, s) => sum + s.activities.length, 0)} activities`);

  /* ---------------- Deep scan for embedded files ---------------- */
  async function deepScanPage(url, visited = new Set()) {
    if (!url || visited.has(url)) return [];
    visited.add(url);
    
    const results = [];
    
    try {
      const res = await fetch(url, { credentials: "same-origin" });
      if (!res.ok) return [];
      
      const html = await res.text();
      const d = new DOMParser().parseFromString(html, "text/html");
      
      // Look for all links to files - more comprehensive search
      const nodes = d.querySelectorAll("a[href], source[src], iframe[src], embed[src], object[data], video source, video[src], audio[src]");
      
      for (const n of nodes) {
        const raw = abs(n.getAttribute("href") || n.getAttribute("src") || n.getAttribute("data") || "");
        if (!raw) continue;
        
        if (looksFile(raw) || /\/pluginfile\.php\//i.test(raw)) {
          // Better title extraction
          let title = san((n.textContent || "").trim().slice(0, 80));
          if (!title || title.length < 3) {
            title = san((n.getAttribute('title') || n.getAttribute('alt') || n.getAttribute('aria-label') || "").trim());
          }
          if (!title || title.length < 3) {
            // Try to extract filename from URL
            const urlMatch = raw.match(/\/([^/?]+\.[a-z0-9]{2,6})(?:\?|$)/i);
            title = urlMatch ? urlMatch[1] : "file";
          }
          results.push({ url: raw, title: san(title) });
        }
      }
      
      // Also look for Moodle-specific resource containers
      d.querySelectorAll('.resourcecontent, .fileuploadsubmission').forEach(container => {
        container.querySelectorAll('a[href*="pluginfile"]').forEach(link => {
          const href = abs(link.getAttribute('href'));
          if (href && looksFile(href)) {
            const title = san(link.textContent?.trim() || "file");
            results.push({ url: href, title });
          }
        });
      });
      
      return results;
    } catch (e) {
      log(`Deep scan error at ${url}: ${e.message}`);
      return [];
    }
  }

  /* ---------------- Download helpers ---------------- */
  const skipped = [];
  const downloadedFiles = new Set();
  
  ui.querySelector("#ms-show").onclick = () => {
    if (!skipped.length) return alert("No skipped items yet.");
    const lines = skipped.map(x => `${x.title} → ${x.url} (${x.reason})`);
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
    if (downloadedFiles.has(link)) {
      log(`Already downloaded: ${link}`);
      return true;
    }
    
    try {
      log(`Downloading: ${link}`);
      const r = await fetch(link, { credentials: "same-origin" });
      if (!r.ok || r.status === 403) { 
        log(`Failed to fetch (${r.status}): ${link}`);
        zip.file(`${dir}${san(title)}.url`, `[InternetShortcut]\nURL=${link}\n`); 
        skipped.push({ title, url: link, type: "Shortcut", reason: `HTTP ${r.status}` }); 
        return false; 
      }

      const cd = r.headers.get("content-disposition") || "";
      const hinted = decodeRFC5987(cd);
      const hintedExt = hinted ? (hinted.match(/\.[a-z0-9]{2,6}$/i) || [""])[0].toLowerCase() : "";
      const ctExt = extFromMime(r.headers.get("content-type"));
      const urlExt = extOf(link);
      const chosenExt = hintedExt || urlExt || ctExt || ".bin";

      if (allowSet.size && chosenExt && !allowSet.has(chosenExt.toLowerCase())) {
        log(`Skipped (type filter): ${chosenExt} - ${link}`);
        return false;
      }

      const base = san(hinted ? hinted.replace(/\.[a-z0-9]{2,6}$/i, "") : (title || "file"));
      const blob = await r.blob();
      
      let finalPath = dir + base + chosenExt;
      let counter = 1;
      while (zip.file(finalPath)) {
        finalPath = dir + base + `_${counter}` + chosenExt;
        counter++;
      }
      
      zip.file(finalPath, blob);
      downloadedFiles.add(link);
      log(`✓ Saved: ${finalPath}`);
      return true;
    } catch (e) {
      log(`Error downloading ${link}: ${e.message}`);
      zip.file(`${dir}${san(title)}.url`, `[InternetShortcut]\nURL=${link}\n`); 
      skipped.push({ title, url: link, type: "Shortcut", reason: e.message });
      return false;
    }
  }

  /* ---------------- Main download logic ---------------- */
  ui.querySelector("#ms-start").onclick = async () => {
    try {
      ui.querySelector("#ms-start").disabled = true;
      const allowSet = allowedSet();
      const doDeep = ui.querySelector("#ms-deepscan").checked;

      let done = 0, filesDownloaded = 0;
      const totalActivities = sections.reduce((sum, s) => sum + s.activities.length, 0);
      C.textContent = `0/${totalActivities} activities, 0 files`; 
      B.style.width = "0%";
      S("Scanning & downloading…");

      for (const section of sections) {
        const sectionDir = `${courseName}/${section.title}/`;
        
        for (const activity of section.activities) {
          const filesToDownload = [];
          
          // Check if it's a direct pluginfile link
          if (/\/pluginfile\.php\//i.test(activity.url) || looksFile(activity.url)) {
            filesToDownload.push({ url: activity.url, title: activity.title });
          }
          // Check if it's a resource/folder page
          else if (/\/mod\/(resource|folder|assign|page|book)\/view\.php/i.test(activity.url)) {
            log(`Fetching: ${activity.url}`);
            try {
              const res = await fetch(activity.url, { credentials: "same-origin" });
              if (res.ok) {
                const html = await res.text();
                const d = new DOMParser().parseFromString(html, "text/html");
                
                // Look for various file link patterns in Moodle
                const selectors = [
                  'a[href*="pluginfile.php"]',
                  '.resourceworkaround a',
                  '.resourcecontent a',
                  'object[data*="pluginfile"]',
                  'embed[src*="pluginfile"]',
                  'video source[src*="pluginfile"]',
                  'audio source[src*="pluginfile"]'
                ];
                
                const foundLinks = new Set();
                
                selectors.forEach(sel => {
                  d.querySelectorAll(sel).forEach(el => {
                    const href = abs(el.getAttribute('href') || el.getAttribute('src') || el.getAttribute('data'));
                    if (href && (looksFile(href) || /pluginfile\.php/i.test(href))) {
                      foundLinks.add(href);
                    }
                  });
                });
                
                if (foundLinks.size > 0) {
                  foundLinks.forEach(href => {
                    filesToDownload.push({ url: href, title: activity.title });
                  });
                } else if (doDeep) {
                  // Deep scan if no direct links found
                  const embedded = await deepScanPage(activity.url);
                  filesToDownload.push(...embedded);
                }
              }
            } catch (e) {
              log(`Error fetching ${activity.url}: ${e.message}`);
            }
          }
          // Deep scan for other module types if enabled
          else if (doDeep) {
            log(`Deep scanning: ${activity.url}`);
            const embedded = await deepScanPage(activity.url);
            filesToDownload.push(...embedded);
          }
          
          // Download all found files
          if (filesToDownload.length > 0) {
            for (const file of filesToDownload) {
              const success = await saveLink(sectionDir, file.title, file.url, allowSet);
              if (success) filesDownloaded++;
              await sleep(150); // Slightly longer delay to be respectful
            }
          } else {
            // Create shortcut if no files found
            log(`No files found for: ${activity.title}`);
            zip.file(`${sectionDir}${san(activity.title)}.url`, `[InternetShortcut]\nURL=${activity.url}\n`);
            skipped.push({ title: activity.title, url: activity.url, type: "Shortcut", reason: "No downloadable files found" });
          }
          
          done++;
          B.style.width = ((done / (totalActivities || 1)) * 100).toFixed(1) + "%";
          C.textContent = `${done}/${totalActivities} activities, ${filesDownloaded} files`;
        }
      }

      S("Building ZIP…");
      const name = `Moodle_${courseName}_${todayTag()}.zip`;
      const blob = await zip.generateAsync({ type: "blob" });
      saveAs(blob, name);
      S(`Done! ${filesDownloaded} files downloaded, ${skipped.length} skipped.`);
      log(`✓ Saved ${name} - ${filesDownloaded} files, ${skipped.length} skipped`);
      
      if (skipped.length > 0) {
        log(`Skipped: ${skipped.map(s => `${s.title} (${s.reason})`).join(', ')}`);
      }
    } catch (e) {
      S("Error: " + (e.message || e)); 
      log("ERROR: " + e.stack);
    } finally {
      ui.querySelector("#ms-start").disabled = false;
      window.__moodle_scraper_active = false;
    }
  };

  log("Ready. Click Scan & Download to begin.");
  S("Ready.");
})();
