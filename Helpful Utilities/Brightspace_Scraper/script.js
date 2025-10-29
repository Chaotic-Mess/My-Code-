/* ===========================================================
   Brightspace_Scraper v3.7
   by chaotic-mess | https://chaotic-mess.github.io/My-Code-/
   Dark-mode downloader for Brightspace (PDFs, MP4s, DOCXs, etc.)
   Auto-detects course, deep-scans HTML topics, and builds a ZIP.
   =========================================================== */
(async () => {
'use strict';
if (window.__bs_scraper) { console.warn("Already running."); return; }
window.__bs_scraper = { version:"3.7" };

///////////////////////////////////////////////////////////////////////////
// Helper Functions
///////////////////////////////////////////////////////////////////////////
const sleep = ms => new Promise(r=>setTimeout(r,ms));
const sanitize = s => s.replace(/[^\w\-\.]/g,'_');
const abs = (url) => new URL(url,location.href).href;
const extOf = (u)=>u.split(/[#?]/)[0].split('.').pop()?.toLowerCase()||'';
const today = ()=>{const d=new Date();return `(${String(d.getFullYear()).slice(2)}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')})`;};

///////////////////////////////////////////////////////////////////////////
// Load Dependencies (JSZip + FileSaver)
///////////////////////////////////////////////////////////////////////////
const loadScript = src => new Promise(res=>{
  const s=document.createElement('script');
  s.src=src; s.onload=res; document.head.appendChild(s);
});
await loadScript("https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js");
await loadScript("https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js");

///////////////////////////////////////////////////////////////////////////
// UI Creation
///////////////////////////////////////////////////////////////////////////
const ui = document.createElement('div');
ui.innerHTML = `
  <div id="bsHeader" style="
    cursor:move;background:repeating-linear-gradient(45deg,#444 0 10px,#666 10px 20px);
    color:white;padding:6px;font-weight:bold;display:flex;align-items:center;justify-content:space-between">
    <span>Brightspace Scraper v3.7</span>
    <span>
      <button id="bsToggle" title="Toggle Theme" style="background:none;border:none;color:inherit;font-size:18px;cursor:pointer">⚙️</button>
      <button id="bsClose" title="Close" style="background:none;border:none;color:inherit;font-size:18px;cursor:pointer">❌</button>
    </span>
  </div>
  <div id="bsBody" style="padding:10px;font-size:13px;max-width:280px;">
    <label><input type="checkbox" value="pdf" checked> PDF</label>
    <label><input type="checkbox" value="mp4" checked> MP4</label>
    <label><input type="checkbox" value="docx" checked> DOCX</label>
    <label><input type="checkbox" value="pptx" checked> PPTX</label>
    <label><input type="checkbox" value="zip" checked> ZIP</label>
    <div style="margin-top:8px">
      <button id="bsStart" style="width:100%;padding:6px;border:none;background:#0f0;color:#000;font-weight:bold;border-radius:4px">Start Scraping</button>
    </div>
    <div id="bsProgress" style="margin-top:8px;height:8px;background:#222;border-radius:4px;overflow:hidden"><div id="bsBar" style="height:8px;width:0;background:lime"></div></div>
    <div id="bsLog" style="margin-top:8px;max-height:150px;overflow:auto;font-family:monospace"></div>
  </div>`;
Object.assign(ui.style,{
  position:'fixed',bottom:'20px',right:'20px',width:'300px',
  background:'#111',color:'#fff',border:'2px solid #333',
  borderRadius:'10px',zIndex:999999,boxShadow:'0 0 10px #000'
});
document.body.appendChild(ui);

///////////////////////////////////////////////////////////////////////////
// Dragging logic
///////////////////////////////////////////////////////////////////////////
(()=>{const drag=ui.querySelector('#bsHeader');
let ox=0,oy=0,is=false;
drag.onmousedown=e=>{is=true;ox=e.clientX-ui.offsetLeft;oy=e.clientY-ui.offsetTop;};
window.onmouseup=()=>is=false;
window.onmousemove=e=>{if(!is)return;ui.style.left=(e.clientX-ox)+'px';ui.style.top=(e.clientY-oy)+'px';ui.style.right='auto';ui.style.bottom='auto';};
})();

///////////////////////////////////////////////////////////////////////////
// Theme Toggle
///////////////////////////////////////////////////////////////////////////
let theme='industrial';
const applyTheme=()=>{
  if(theme==='industrial'){
    ui.style.background='#333';
    ui.style.color='#0f0';
    ui.querySelector('#bsHeader').style.background='repeating-linear-gradient(45deg,#555 0 10px,#999 10px 20px)';
    ui.querySelector('#bsBar').style.background='lime';
  }else{
    ui.style.background='#111';
    ui.style.color='#9df';
    ui.querySelector('#bsHeader').style.background='#222';
    ui.querySelector('#bsBar').style.background='#4cf';
  }
};
applyTheme();
ui.querySelector('#bsToggle').onclick=()=>{theme=theme==='industrial'?'dark':'industrial';applyTheme();};

///////////////////////////////////////////////////////////////////////////
// Core Scraper
///////////////////////////////////////////////////////////////////////////
const log = t => {
  console.log('[Brightspace]',t);
  const d=document.createElement('div');d.textContent=t;ui.querySelector('#bsLog').appendChild(d);
  ui.querySelector('#bsLog').scrollTop=9999;
};

async function collectLinksAggressive() {
  const selTypes=[...ui.querySelectorAll('input[type=checkbox]:checked')].map(c=>c.value);
  const links=new Set();
  const walker=(el)=>{
    el.querySelectorAll('a[href],iframe[src],embed[src],source[src]').forEach(x=>{
      const u=abs(x.href||x.src);
      const ex=extOf(u);
      if(selTypes.includes(ex)) links.add(u);
    });
  };
  walker(document);
  // Also handle embedded Brightspace HTML blocks
  document.querySelectorAll('d2l-html-block').forEach(b=>{
    try{const html=b.getAttribute('html');if(html){const t=document.createElement('div');t.innerHTML=html;walker(t);}}catch(e){}
  });
  return [...links];
}

///////////////////////////////////////////////////////////////////////////
// Downloader
///////////////////////////////////////////////////////////////////////////
async function fetchAsBlob(url){
  try{
    const r=await fetch(url);
    if(!r.ok) throw new Error(r.status);
    return await r.blob();
  }catch(e){
    log('❌ Failed '+url);
    const blob=new Blob([`[InternetShortcut]\nURL=${url}\n`],{type:'text/plain'});
    return new File([blob],sanitize(url.split('/').pop())+'.url',{type:'text/plain'});
  }
}

///////////////////////////////////////////////////////////////////////////
// 🚀 Main
///////////////////////////////////////////////////////////////////////////
ui.querySelector('#bsStart').onclick = async ()=>{
  const bar=ui.querySelector('#bsBar');
  const links=await collectLinksAggressive();
  log(`Found ${links.length} downloadable items`);
  const zip=new JSZip();
  let i=0;
  for(const url of links){
    i++;
    log(`Fetching (${i}/${links.length}): ${url}`);
    const blob=await fetchAsBlob(url);
    const name=sanitize(url.split('/').pop()||('file_'+i));
    zip.file(name,blob);
    bar.style.width=((i/links.length)*100)+'%';
    await sleep(300);
  }
  const cname=(document.title.match(/([A-Z]{2,}\s?\d{3,})/)||['Course'])[0];
  const zipName=`Brightspace_${sanitize(cname)}_${today()}.zip`;
  log('Zipping…');
  const blob=await zip.generateAsync({type:'blob'});
  saveAs(blob,zipName);
  log('✅ Done! File saved as '+zipName);
};

///////////////////////////////////////////////////////////////////////////
// Close / Reset
///////////////////////////////////////////////////////////////////////////
ui.querySelector('#bsClose').onclick=()=>{
  ui.remove();delete window.__bs_scraper;
};
window.addEventListener('keydown',e=>{
  if(e.key==='Escape'){ui.remove();delete window.__bs_scraper;}
});
})();
