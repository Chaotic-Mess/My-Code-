# LUNA Banking Website

A futuristic banking website with quantum-themed design, (Mock) AI features, and light/dark mode support.

## Lore?
Motto: “We don’t own your money. You do.”
The world’s first hybrid banking system — staffed by both humans and AI.
It operates on fairness algorithms designed to balance institutional and personal gain.
Low-interest, high-approval loans; immediate account generation; special programs for students, elders, and new families.
To the public, it’s a bank.
To economists, it’s a paradox: a financial empire that refuses to exploit.

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

## Notes

- Theme preference is saved to localStorage
- Video autoplay requires user interaction on some browsers
- Leaflet.js CDN required for locations map
- Font Awesome CDN for icons

---

**Built with 🌙 for the future of banking**

