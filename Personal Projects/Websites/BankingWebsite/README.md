# LUNA Banking Website

A futuristic banking website with quantum-themed design, AI features, and light/dark mode support.

## Project Structure

```
BankingWebsite/
├── index.html              # Main landing page
│
├── pages/                  # All additional HTML pages
│   ├── wallets.html        # Digital Vault - Card management
│   ├── transactions.html   # Quantum Transfers - Transaction history
│   ├── services.html       # Banking services and programs
│   ├── support.html        # 24/7 AI & Human support
│   ├── locations.html      # Global locations with map
│   ├── about.html          # Company information
│   ├── privacy.html        # Privacy policy
│   └── terms.html          # Terms of service
│
├── css/
│   ├── style.css           # Global styles & theme variables
│   ├── wallets.css         # Wallet page specific styles
│   ├── transactions.css    # Transaction page styles
│   ├── services.css        # Services page styles
│   ├── support.css         # Support page styles
│   ├── locations.css       # Locations page styles
│   ├── about.css           # About page styles
│   └── legal.css           # Privacy & Terms page styles
│
├── js/
│   ├── theme.js            # Light/Dark mode theme switcher
│   ├── main.js             # Homepage video cycling & animations
│   ├── wallets.js          # Wallet page functionality
│   ├── transactions.js     # Transaction page functionality
│   ├── services.js         # Services page functionality
│   ├── support.js          # Support page functionality
│   └── locations.js        # Interactive map functionality
│
├── video/
│   ├── luna-card-1.mp4     # Background video 1 (15s)
│   └── luna-card-2.mp4     # Background video 2 (10s)
│
└── support/                # Support center subdirectory
    └── [support pages]     # Additional help pages
```

## Features

### Theme System
- **Light/Dark Mode**: Toggle between themes with persistent localStorage
- **Smooth Transitions**: All elements transition smoothly between themes
- **Smart Overlays**: Video overlay adjusts opacity based on theme
- **Cosmic Effects**: Stars and twinkling effects visible only in dark mode

### Video Background
- **Dual Video Cycling**: Alternates between two promo videos
- **Reverse Effect**: Videos rewind 1.5s before transitioning
- **Smooth Crossfade**: 1s fade transition between videos
- **Duration-Aware**: Respects each video's individual duration (15s / 10s)

### Pages

#### Home (index.html)
- Hero section with video background
- Next-Gen Banking features showcase
- Global presence statistics
- Interactive feature cards

#### Vault (wallets.html)
- Card carousel display
- Balance information
- Recent transactions
- Quick actions

#### Transfers (transactions.html)
- Transaction dashboard
- Incoming/Outgoing summaries
- Transfer history with filters
- Pagination

#### Services (services.html)
- Personal banking services
- Business solutions
- Special programs (Seniors, New Parents, Education)
- NOVA AI advisor
- Quantum Credit System

#### Support (support.html)
- AI assistant chat option
- Human expert connection
- Quick help categories
- Knowledge base search
- Global support centers

#### Locations (locations.html)
- Interactive Leaflet.js map
- Location search
- Banking center types
- Featured locations

#### ℹAbout (about.html)
- Company mission
- Journey timeline
- Leadership team
- Future vision

#### Legal (privacy.html, terms.html)
- Privacy policy
- Terms of service
- Contact information

## Technologies Used

- **HTML5**: Semantic markup
- **CSS3**: Custom properties, Grid, Flexbox, Animations
- **JavaScript (ES6+)**: Vanilla JS for all interactions
- **Leaflet.js**: Interactive maps on locations page
- **Font Awesome**: Icons throughout the site
- **Space Grotesk Font**: Modern, tech-forward typography

## Design Philosophy

- **Futuristic Aesthetic**: Space-themed with quantum/cosmic elements
- **Accessibility First**: High contrast, readable fonts, semantic HTML
- **Performance**: Optimized animations, efficient transitions
- **Responsive**: Mobile-first approach (to be expanded)
- **Progressive Enhancement**: Works without JS, enhanced with it

## Getting Started

1. Ensure all video files are in the `video/` directory
2. Open `index.html` in a modern browser
3. Toggle theme using the sun/moon icon in the navbar
4. Explore all pages via the navigation menu

## Notes

- Theme preference is saved to localStorage
- Video autoplay requires user interaction on some browsers
- Leaflet.js CDN required for locations map
- Font Awesome CDN for icons

## Future Enhancements

- [ ] Mobile responsive breakpoints
- [ ] Additional animations and micro-interactions
- [ ] Form validation for support pages
- [ ] Backend integration ready
- [ ] PWA capabilities
- [ ] Performance optimizations
- [ ] Analytics integration

---

**Built with 🌙 for the future of banking**
