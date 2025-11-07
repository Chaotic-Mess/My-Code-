// Moodle Scraper Landing Page Script
// Generates bookmarklet & handles copy button

const BOOKMARKLET = `javascript:(async()=>{try{
  const srcs=[
    'https://chaotic-mess.github.io/My-Code-/Helpful%20Utilities/Moodle_Scraper/script.js',
    'https://raw.githubusercontent.com/chaotic-mess/My-Code-/main/Helpful%20Utilities/Moodle_Scraper/script.js',
    'https://cdn.jsdelivr.net/gh/chaotic-mess/My-Code-/Helpful%20Utilities/Moodle_Scraper/script.js'
  ];
  let ok=false;
  for(const u of srcs){
    try{
      const r=await fetch(u+'?t='+(+new Date()));
      if(r.ok){(0,eval)(await r.text());ok=true;break;}
    }catch{}
  }
  if(!ok)alert('⚠️ Unable to load Moodle_Scraper.');
}catch(e){alert('Error: '+e.message);}})();`;

document.addEventListener("DOMContentLoaded", () => {
  const link = document.getElementById("bookmarklet");
  link.href = BOOKMARKLET;

  const btn = document.getElementById("copyBtn");
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(BOOKMARKLET);
      btn.textContent = "✅ Copied!";
      setTimeout(() => (btn.textContent = "📋 Copy Bookmarklet"), 2000);
    } catch (err) {
      alert("Failed to copy. Right-click the button and choose 'Copy link address'.");
    }
  });
});
