// ═══════════════════════════════════════════════════════════
// AEON SERVICES - MAIN CONTROL SYSTEM
// ═══════════════════════════════════════════════════════════

import { initBootSequence } from './modules/boot.js';
import { initCursorEffects } from './modules/cursor.js';
import { initNavigation } from './modules/navigation.js';
import { generateOperators } from './modules/operators.js';
import { initTerminal } from './modules/terminal.js';
import { initAudio } from './modules/audio.js';
import { initLiveOps } from './modules/liveops.js';
import { updateStatusBar } from './modules/statusbar.js';

// INITIALIZATION
document.addEventListener('DOMContentLoaded', async () => {
    console.log('%cAEON SERVICES', 'color: #00ff88; font-size: 24px; font-weight: bold;');
    console.log('%cYou should not be here.', 'color: #ff0055; font-size: 14px;');
    
    // Initialize audio system
    await initAudio();
    
    // Start boot sequence
    await initBootSequence();
    
    // Initialize core systems
    initCursorEffects();
    initNavigation();
    generateOperators();
    initTerminal();
    initLiveOps();
    
    // Status bar clock
    setInterval(updateStatusBar, 1000);
    updateStatusBar();
});

// Custom cursor movement
document.addEventListener('mousemove', (e) => {
    document.body.style.setProperty('--mouse-x', e.clientX + 'px');
    document.body.style.setProperty('--mouse-y', e.clientY + 'px');
});

// Prevent right-click (immersion)
document.addEventListener('contextmenu', (e) => {
    e.preventDefault();
});
