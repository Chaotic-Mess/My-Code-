/*!
 * ===========================================================
 * Brightspace Scraper v3.8 (Hybrid Full-Fat Edition)
 * by chaotic-mess | https://chaotic-mess.github.io/My-Code-/
 * -----------------------------------------------------------
 * Unified support for:
 *  • /d2l/le/content/ (ToC API)
 *  • /d2l/le/lessons/ (Smart Curriculum iframe)
 *  • Fallback DOM scan
 * 
 * Features:
 *  • Deep HTML link discovery (<a>, <iframe>, <embed>, <source>)
 *  • Smart .url shortcut fallback for locked/unfetchable files
 *  • Smart filename: Brightspace_[CourseName]_(YY-MM-DD).zip
 *  • Draggable hazard-bar UI with ⚙️ theme toggle + ESC close
 *  • Transparent console logging for user trust
 * ===========================================================
 */

(async () => {
'use strict';
if (window.__bs_scraper_active) { alert("Already running."); return; }
window.__bs_scraper_active = true;

//////////////////////////////////////////////////////////////////////
// Utility helpers
//////////////////////////////////////////////////////////////////////
const sleep = ms => new Promise(r=>setTimeout(r,ms));
const abs = u => { try { return new URL(u, location.href).href; } catch { return null; } };
const san = s => (s || "").replace(/[<>:"/\\|?*]+/g,'_').trim();
const extOf = u => { const m = u && u.match(/\.[a-z0-9]{2,5}(?:$|\?)/i); return m ? m[0].toLowerCase() : ""; };
const looksFile = u => /\.(pdf|mp4|docx?|pptx?|xlsx?|zip|txt|csv|rtf|md|epub)(?:[?#].*)?$/i.test(u)
                     || /\/content\/enforced\//i.test(u) || /\/d2l\/common\/viewFile\.d2l/i.test(u);
const today = () => { const d=new Date(); return `(${String(d.getFullYear()).slice(2)}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')})`; };

//////////////////////////////////////////////////////////////////////
// Load deps
//////////////////////////////////////////////////////////////////////
const load = src => new Promise(r=>{const s=document.createElement("script");s.src=src;s.onload=r;document.head.appendChild(s);});
await load("https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js");
await load("https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js");

//////////////////////////////////////////////////////////////////////
// UI Setup
//////////////////////////////////////////////////////////////////////
const ui=document.createElement("div");
ui.innerHTML=`
  <div id="bsHead" style="cursor:move;padding:6px;font-weight:bold;
      background:repeating-linear-gradient(45deg,#444 0 10px,#777 10px 20px);
      display:flex;justify-content:space-between;align-items:center;color:#fff">
    <span>Brightspace Scraper v3.8</span>
    <span>
      <button id="bsToggle" title="Toggle Theme" style="background:none;border:0;color:inherit;font-size:18px;cursor:pointer">⚙️</button>
      <button id="bsClose" title="Close" style="background:none;border:0;color:inherit;font-size:18px;cursor:pointer">❌</button>
    </span>
  </div>
  <div id="bsBody" style="padding:10px;font-size:13px;max-width:300px">
    <label><input type="checkbox" value=".pdf" checked> PDF</label>
    <label><input type="checkbox" value=".mp4" checked> MP4</label>
    <label><input type="checkbox" value=".docx" checked> DOCX</label>
    <label><input type="checkbox" value=".pptx" checked> PPTX</label>
    <label><input type="checkbox" value=".zip" checked> ZIP</label>
    <label><input type="checkbox" value=".txt" checked> TXT</label>
    <div style="margin-top:8px">
      <button id="bsStart" style="width:100%;padding:6px;border:none;
        background:#0b67ff;color:#fff;font-weight:bold;border-radius:4px">
        Start Scraping
      </button>
    </div>
    <div id="bsStatus" style="margin-top:6px;color:#ccc"></div>
    <div style="margin-top:6px;height:8px;background:#222;border-radius:4px;overflow:hidden">
      <div id="bsBar" style="height:8px;width:0;background:#4ade80"></div>
    </div>
    <div id="bsLog" style="margin-top:8px;max-height:150px;overflow:auto;
      font-family:monospace;color:#aaa"></div>
  </div>`;
Object.assign(ui.style,{
  position:'fixed',bottom:'20px',right:'20px',background:'#111',color:'#fff',
  border:'2px solid #333',borderRadius:'10px',zIndex:999999,boxShadow:'0 0 10px #000'
});
document.body.appendChild(ui);

//////////////////////////////////////////////////////////////////////
// Draggable
//////////////////////////////////////////////////////////////////////
(()=>{const drag=ui.querySelector('#bsHead');
let ox=0,oy=0,md=false;
drag.onmousedown=e=>{md=true;ox=e.clientX-ui.offsetLeft;oy=e.clientY-ui.offsetTop;};
window.onmouseup=()=>md=false;
window.onmousemove=e=>{
  if(!md)return;
  ui.style.left=(e.clientX-ox)+'px';
  ui.style.top=(e.clientY-oy)+'px';
  ui.style.right='auto';ui.style.bottom='auto';
};})();

//////////////////////////////////////////////////////////////////////
// Theme toggle
//////////////////////////////////////////////////////////////////////
let theme='industrial';
const applyTheme=()=>{
  if(theme==='industrial'){
    ui.style.background='#222';ui.style.color='#9f9';
    ui.querySelector('#bsHead').style.background='repeating-linear-gradient(45deg,#555 0 10px,#888 10px 20px)';
    ui.querySelector('#bsBar').style.background='#4ade80';
  } else {
    ui.style.background='#111';ui.style.color='#9df';
    ui.querySelector('#bsHead').style.background='#222';
    ui.querySelector('#bsBar').style.background='#4cf';
  }
};
ui.querySelector('#bsToggle').onclick=()=>{theme=theme==='industrial'?'dark':'industrial';applyTheme();};
applyTheme();

//////////////////////////////////////////////////////////////////////
// Logging helpers
//////////////////////////////////////////////////////////////////////
const log = t => {console.log('[BS]',t);const d=document.createElement('div');d.textContent=t;ui.querySelector('#bsLog').appendChild(d);ui.querySelector('#bsLog').scrollTop=9e9;};
const status = t => ui.querySelector('#bsStatus').textContent=t;

//////////////////////////////////////////////////////////////////////
// Fetch ToC or fallback DOM
//////////////////////////////////////////////////////////////////////
const m = location.pathname.match(/\/d2l\/(?:le\/content\/|lessons\/|home\/)(\d+)/);
const courseId = m && m[1];
let toc=null,courseName="";

async function getToC() {
  try {
    const r = await fetch(`/d2l/api/le/1.68/${courseId}/content/toc`, { credentials:"same-origin" });
    if(!r.ok) throw new Error(r.status);
    const j = await r.json();
    if(j && j.Modules) return j;
  } catch(e) { log('ToC fetch failed.'); }
  return null;
}

function scrapeDomLinks() {
  const links=[];
  document.querySelectorAll('a[href]').forEach(a=>{
    const h=abs(a.getAttribute('href'));if(h && looksFile(h))links.push({Title:a.textContent.trim()||'file',Url:h});
  });
  return {Modules:[{Title:'Loose Files',Topics:links}]};
}

async function getSmartCurriculumLinks(){
  const frame=document.querySelector('iframe[src*="smart-curriculum"]');
  if(!frame)return null;
  try{
    const doc=frame.contentDocument||frame.contentWindow.document;
    const as=[...doc.querySelectorAll('a[href]')].filter(a=>looksFile(a.href))
        .map(a=>({Title:a.textContent.trim()||'file',Url:a.href}));
    return {Modules:[{Title:'SmartCurriculum',Topics:as}]};
  }catch{return null;}
}

//////////////////////////////////////////////////////////////////////
// Deep Scan
//////////////////////////////////////////////////////////////////////
async function deepScan(url){
  try{
    const r=await fetch(url,{credentials:'same-origin'});
    if(!r.ok)return[];
    const html=await r.text();
    const d=new DOMParser().parseFromString(html,'text/html');
    const found=[];
    d.querySelectorAll('a[href],source[src],iframe[src],embed[src]').forEach(n=>{
      const u=abs(n.getAttribute('href')||n.getAttribute('src')||'');
      if(u&&looksFile(u))found.push(u);
    });
    return found;
  }catch{return[];}
}

//////////////////////////////////////////////////////////////////////
// Download Worker
//////////////////////////////////////////////////////////////////////
ui.querySelector('#bsStart').onclick=async()=>{
  ui.querySelector('#bsStart').disabled=true;
  const bar=ui.querySelector('#bsBar'); bar.style.width='0';
  status('Collecting topics…');

  toc = await getToC();
  if(!toc || !toc.Modules?.length){
    toc = await getSmartCurriculumLinks() || scrapeDomLinks();
  }
  courseName = san(toc?.Title || document.title || "Course");
  const topics=[]; const walk=m=>{(m.Topics||[]).forEach(t=>topics.push(t));(m.Modules||[]).forEach(walk);}; (toc.Modules||[]).forEach(walk);
  log(`Found ${topics.length} topics.`);

  const allow=[...ui.querySelectorAll('input[type=checkbox]:checked')].map(c=>c.value);
  const zip=new JSZip(); let done=0;
  const addURLShortcut=(dir,title,link)=>{zip.file(`${dir}${san(title)}.url`,`[InternetShortcut]\nURL=${link}\n`);};

  for(const t of topics){
    if(!t.Url){continue;}
    const u=t.Url; const e=extOf(u);
    if(allow.length && !allow.includes('.'+e)){continue;}
    let targets=[u];
    if(/\.html?/i.test(u)){const extra=await deepScan(u);targets.push(...extra);}
    for(const link of targets){
      try{
        const r=await fetch(link,{credentials:'same-origin'});
        if(!r.ok){addURLShortcut('',t.Title||'link',link);continue;}
        const b=await r.blob();
        const fn=san((t.Title||'file')+(extOf(link)||'.bin'));
        zip.file(fn,b); done++;
        bar.style.width=((done/(topics.length||1))*100).toFixed(1)+'%';
        status(`${done}/${topics.length} downloaded`);
      }catch{addURLShortcut('',t.Title||'link',link);}
      await sleep(80);
    }
  }

  status('Building ZIP…');
  const blob=await zip.generateAsync({type:'blob'});
  const zipName=`Brightspace_${courseName}_${today()}.zip`;
  saveAs(blob,zipName);
  status(`✅ Done! ${done} files saved (${zipName})`);
  log('Complete.');
};

//////////////////////////////////////////////////////////////////////
// Exit handlers
//////////////////////////////////////////////////////////////////////
ui.querySelector('#bsClose').onclick=()=>{ui.remove();window.__bs_scraper_active=false;};
window.addEventListener('keydown',e=>{if(e.key==='Escape'){ui.remove();window.__bs_scraper_active=false;}});
})();
