# ÆEON SERVICES - OPERATOR MANUAL

## ▼ QUICK START

1. **Launch the server:**
   ```powershell
   .\launch.ps1
   ```
   Or manually:
   ```powershell
   python -m http.server 8000
   ```

2. **Open browser:**
   Navigate to `http://localhost:8000`

3. **Experience the boot sequence** (6 seconds)

4. **Navigate the interface:**
   - Right-side panel: Switch between sections
   - Operators: Click cards to view dossiers
   - Black Words: Read the manifesto
   - Terminal: Type `help` for commands
   - Live Ops: **LOCKED** (unlock via puzzle)

---

## ▼ NAVIGATION GUIDE

### **OPERATORS**
- **37 operator cards** with threat levels
- **Click any operator** to open their dossier
- **Dossier contains:**
  - Completion rate, stealth grade, ethical flexibility
  - Specializations
  - Case logs (some will censor themselves after 8-13 seconds)
- **Close dossier:** Click the × in top-right

### **LIVE OPS** (Hidden Access)
**TO UNLOCK:**
1. Click **OPERATORS**
2. Click **BLACK WORDS**
3. Click **TERMINAL** (the ████ option)
4. Click **OPERATORS** again

The lock (■) will fade, and you'll gain access to:
- Real-time mission tracker
- CCTV feeds
- Status updates (ACTIVE/SILENCED)

### **BLACK WORDS**
- Flickering manifesto text
- No interaction needed—just absorb the rage

### **TERMINAL**
Type any of these commands:
- `help` - See all commands
- `list` - List all 37 operators
- `query PHANTOM` - Query a specific operator (replace PHANTOM with any codename)
- `status` - System status
- `missions` - Active mission count
- `clearance` - Your access level
- `trace` - See trace information
- `encrypt hello world` - Encrypt a message
- `decrypt [encrypted]` - Decrypt a message
- `exit` - Close terminal

**HIDDEN COMMANDS:**
- `ghost` - ???
- `aeon` - ???

---

## ▼ KEYBOARD SHORTCUTS

None. This is a mouse-driven experience.

---

## ▼ SOUND DESIGN

The website generates sounds procedurally:
- **Ambient hum:** Low-frequency background
- **Hover static:** Radio-like noise on UI hover
- **Bass drop:** Deep sub-bass on operator selection
- **Glitch:** Digital distortion on special events

**Note:** Some browsers block autoplay audio. Click anywhere to enable sound.

---

## ▼ VISUAL EFFECTS

- **Custom cursor:** Animated diamond with glow
- **Cursor distortion:** Particle trail following mouse
- **Static noise:** During boot sequence
- **Vault doors:** Sliding panels with weight
- **Glitch text:** On navigation hover
- **Scanlines:** On CCTV feeds
- **Flickering:** On manifesto text
- **Censorship:** Time-based log fading

---

## ▼ EASTER EGGS

1. **Hidden terminal commands**
2. **Live Ops puzzle unlock**
3. **Time-based log censorship**
4. **Console messages** (open browser DevTools)
5. **Right-click disabled** (immersion)

---

## ▼ PERFORMANCE TIPS

- Use Chrome/Firefox/Edge for best experience
- Hardware acceleration recommended
- Close unnecessary tabs for smooth animations
- Sound works best on desktop (mobile may have limitations)

---

## ▼ CUSTOMIZATION

Want to modify AEON SERVICES? See `README.md` for:
- Adding operators
- Changing colors
- Modifying terminal commands
- Adjusting unlock sequences
- Custom sounds

---

## ▼ TROUBLESHOOTING

**Q: Audio not playing?**
A: Click anywhere on the page to enable sound (browser autoplay policy).

**Q: Animations stuttering?**
A: Close other tabs, enable hardware acceleration in browser settings.

**Q: Live Ops won't unlock?**
A: Follow the sequence exactly: Operators → Black Words → Terminal → Operators

**Q: Can't see custom cursor?**
A: Make sure you're not on a touchscreen device. Cursor effects are mouse-only.

**Q: Server won't start?**
A: Make sure port 8000 is available. Or change the port:
```powershell
python -m http.server 9000
```

---

## ▼ SHARING

To share your AEON SERVICES instance:

**Option 1: GitHub Pages**
1. Create a GitHub repository
2. Upload all files
3. Enable GitHub Pages in settings
4. Access at `yourusername.github.io/repository-name`

**Option 2: Netlify**
1. Drag folder into Netlify drop zone
2. Get instant live URL

**Option 3: Vercel**
1. Connect GitHub repo
2. Deploy automatically

---

## ▼ CREDITS

**Design & Development:** You (the architect of digital nightmares)
**Inspiration:** Black ops, cyberpunk cinema, surveillance culture
**Tech:** Pure vanilla HTML/CSS/JS—no frameworks needed

---

**ÆEON SERVICES**  
*"Control isn't held by governments or tech companies—*  
*but by those you've never seen."*

Access established. Trace logged. Welcome to the shadow network.
