# ÆEON SERVICES

> *"Visibility is weakness. Legacy is digital. You were never private."*

---

## ▼ CLASSIFIED OVERVIEW

AEON SERVICES is not a website. It's a digital manifestation of power in the shadows—a cinematic, interactive experience showcasing 37 elite operators for hire. Built for those who understand that control doesn't belong to governments or corporations, but to those you've never seen.

## ▼ FEATURES

### ◆ BOOT SEQUENCE
- Distorted static canvas animation
- SVG triangle-eye symbol with glitch effects
- Manifesto text reveal with staggered timing
- Low-frequency ambient hum (Web Audio API)
- 6-second cinematic introduction

### ◆ VAULT DOOR ANIMATION
- Physics-based sliding panels with weight
- Cubic bezier easing for theatrical reveal
- Border lighting effects

### ◆ INTERFACE SYSTEMS
- **Custom Cursor**: Animated diamond cursor with glow and pulse
- **Cursor Distortion**: Real-time particle system following mouse movement
- **Navigation**: Right-side panel system with hover glitch effects
- **No Scrollbars**: Gesture-based content navigation
- **Status Bar**: Real-time UTC clock, connection status, operator count

### ◆ OPERATORS SYSTEM
- **37 Unique Operators**: Procedurally generated with distinct profiles
- **Operator Cards**: Hover animations with weighted transforms
- **Full Dossiers**: 
  - Execution stats (completion rate, stealth grade, ethical flexibility)
  - Specializations: Sabotage, extraction, data destabilization, etc.
  - Case logs with time-based censorship (logs fade after 8-13 seconds)
  - Redacted text effects
- **Sound Design**: Radio static on hover, bass drop on selection

### ◆ LIVE OPS (Hidden Access)
- **Puzzle Unlock**: Navigate in sequence: Operators → Black Words → Terminal → Operators
- **Mission Tracker**: Real-time updating mission feeds
- **CCTV Feeds**: Blurred surveillance with scanline effects
- **Status Tags**: ACTIVE / SILENCED indicators

### ◆ BLACK WORDS
- Flickering manifesto text
- Anti-surveillance poetry
- Rage against data exploitation
- Cinematic pacing with delayed reveals

### ◆ TERMINAL SYSTEM
- Full command-line interface
- **Available Commands**:
  - `help` - Display commands
  - `list` - List all operators
  - `query [CODENAME]` - Query operator details
  - `status` - System status
  - `missions` - Active mission count
  - `clearance` - Check access level
  - `trace` - View trace information
  - `encrypt [MESSAGE]` - Encrypt text
  - `decrypt [ENCRYPTED]` - Decrypt text
  - `exit` - Close terminal
  - **Hidden**: `ghost`, `aeon` (easter eggs)

### ◆ AUDIO SYSTEM
- Procedurally generated sounds via Web Audio API
- Ambient low-frequency hum
- Hover static noise
- Bass drop on interactions
- Glitch sounds for special events

### ◆ ERROR PAGES
- In-universe 403 page with trace logging
- Device fingerprinting display
- Scanline animation effects

## ▼ TECH STACK

- **Pure HTML5 / CSS3 / Vanilla JavaScript** (ES6 Modules)
- **Web Audio API** for procedural sound generation
- **Canvas API** for particle effects and distortion
- **SVG** for vector graphics and filters
- **CSS Animations** with cubic-bezier easing
- **No frameworks, no dependencies** - Raw power only

## ▼ FILE STRUCTURE

```
SickasfWebsite/
├── index.html              # Main entry point
├── 403.html                # Custom error page
├── css/
│   └── main.css           # All styling (modular organization)
└── js/
    ├── main.js            # Core initialization
    └── modules/
        ├── boot.js        # Boot sequence
        ├── cursor.js      # Cursor effects
        ├── navigation.js  # Panel navigation
        ├── operators.js   # Operator generation
        ├── terminal.js    # Terminal system
        ├── audio.js       # Sound generation
        ├── liveops.js     # Live ops unlock
        └── statusbar.js   # Status updates
```

## ▼ USAGE

### Local Development
1. Clone or download this repository
2. Open `index.html` in a modern browser (Chrome/Firefox/Edge recommended)
3. **Important**: Some features require HTTPS or `localhost` for audio autoplay

### Hosting
- Deploy to any static hosting service (GitHub Pages, Netlify, Vercel)
- No build process required
- No server-side code needed

### Browser Compatibility
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+ (some audio features may require user interaction)

## ▼ CUSTOMIZATION

### Adding Operators
Edit `js/modules/operators.js`:
```javascript
const CODENAMES = ['YOUR', 'CUSTOM', 'NAMES'];
```

### Changing Colors
Edit CSS variables in `css/main.css`:
```css
:root {
    --aeon-green: #00ff88;  /* Primary accent */
    --aeon-red: #ff0055;    /* Alerts */
    --aeon-black: #000000;  /* Background */
}
```

### Modifying Terminal Commands
Add commands in `js/modules/terminal.js`:
```javascript
const COMMANDS = {
    yourcommand: {
        description: 'Your description',
        execute: (args) => {
            return 'Your output';
        }
    }
};
```

### Changing Unlock Sequence
Edit `js/modules/liveops.js`:
```javascript
const UNLOCK_SEQUENCE = ['operators', 'black-words', 'terminal', 'operators'];
```

## ▼ PERFORMANCE

- **Initial Load**: ~100KB total (HTML + CSS + JS)
- **No external dependencies**
- **Lazy audio generation**: Sounds created on first interaction
- **Optimized animations**: GPU-accelerated transforms
- **Efficient particle system**: Culls dead particles automatically

## ▼ EASTER EGGS

- Hidden terminal commands: `ghost`, `aeon`
- Live Ops unlock puzzle
- Time-based log censorship
- Console messages for developers
- Right-click disabled (immersion)

## ▼ DESIGN PHILOSOPHY

**"This isn't a website. It's an invitation to disappear."**

Every element is designed to feel:
- **Alive**: Dynamic animations, real-time updates
- **Intelligent**: Reactive, tracking, responsive
- **Dangerous**: Sharp edges, threatening aesthetics
- **Cinematic**: Film-quality timing and composition
- **Immersive**: Sound design, custom cursors, no breaks

## ▼ INSPIRATION

- Black ops briefing folders
- Luxury cyberpunk film UIs (Blade Runner, Ghost in the Shell)
- Psychological thrillers
- Modern surveillance aesthetics
- Digital espionage culture

## ▼ LICENSE

This is art. This is code. This is a statement.

Use it. Modify it. Make something that scares you.

Just remember: **AEON is always watching.**

---

## ▼ DEPLOYMENT

To view this masterpiece:

```bash
# Option 1: Simple HTTP server
python -m http.server 8000

# Option 2: Node.js
npx http-server

# Option 3: PHP
php -S localhost:8000
```

Then navigate to `http://localhost:8000`

---

**ÆEON SERVICES**  
*Est. ████*  
*"We are the silence between the screams."*
