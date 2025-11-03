// ═══════════════════════════════════════════════════════════
// LIVE OPS - Puzzle unlock and mission tracking
// ═══════════════════════════════════════════════════════════

let interactionSequence = [];
const UNLOCK_SEQUENCE = ['operators', 'black-words', 'terminal', 'operators'];

export function initLiveOps() {
    const navItems = document.querySelectorAll('.nav-item');
    const liveOpsNav = document.querySelector('[data-section="live-ops"]');
    const liveOpsSection = document.getElementById('live-ops-section');
    const accessDenied = liveOpsSection.querySelector('.access-denied');
    const liveOpsContent = document.getElementById('live-ops-content');
    
    // Track navigation pattern
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const section = item.getAttribute('data-section');
            if (section !== 'live-ops') {
                interactionSequence.push(section);
                
                // Keep only last 4 interactions
                if (interactionSequence.length > 4) {
                    interactionSequence.shift();
                }
                
                // Check if sequence matches
                if (checkSequence()) {
                    unlockLiveOps(liveOpsNav, accessDenied, liveOpsContent);
                }
            }
        });
    });

    // Animate mission feeds
    setInterval(() => {
        updateMissionFeeds();
    }, 5000);
    
    // Initialize Live Ops controls (after unlock)
    initLiveOpsControls();
}

function initLiveOpsControls() {
    // Mission filter
    const missionFilter = document.querySelector('.mission-filter');
    if (missionFilter) {
        missionFilter.addEventListener('input', (e) => {
            const filter = e.target.value.toLowerCase();
            const feedItems = document.querySelectorAll('.feed-item');
            
            feedItems.forEach(item => {
                const code = item.querySelector('.mission-code').textContent.toLowerCase();
                const location = item.querySelector('.mission-location').textContent.toLowerCase();
                
                if (code.includes(filter) || location.includes(filter)) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }
    
    // Mission refresh
    const missionRefresh = document.querySelector('.mission-refresh');
    if (missionRefresh) {
        missionRefresh.addEventListener('click', () => {
            // Animate refresh
            missionRefresh.style.transform = 'rotate(360deg)';
            missionRefresh.style.transition = 'transform 0.5s ease';
            
            setTimeout(() => {
                missionRefresh.style.transform = 'rotate(0deg)';
                updateMissionFeeds();
            }, 500);
        });
    }
    
    // Feed toggle
    const feedToggle = document.querySelector('.feed-toggle');
    const cctvWindows = document.querySelectorAll('.cctv-window');
    let feedsActive = true;
    
    if (feedToggle) {
        feedToggle.addEventListener('click', () => {
            feedsActive = !feedsActive;
            
            if (feedsActive) {
                feedToggle.textContent = '■ KILL FEED';
                feedToggle.style.borderColor = 'var(--aeon-red)';
                feedToggle.style.color = 'var(--aeon-red)';
                cctvWindows.forEach(w => w.style.opacity = '1');
            } else {
                feedToggle.textContent = '▶ RESTORE FEED';
                feedToggle.style.borderColor = 'var(--aeon-green)';
                feedToggle.style.color = 'var(--aeon-green)';
                cctvWindows.forEach(w => w.style.opacity = '0.3');
            }
        });
    }
    
    // Feed selector
    const feedSelector = document.querySelector('.feed-selector');
    if (feedSelector) {
        feedSelector.addEventListener('change', (e) => {
            const selectedFeed = e.target.value;
            const firstCCTV = document.querySelector('.cctv-window .cctv-header');
            if (firstCCTV) {
                firstCCTV.textContent = selectedFeed;
            }
        });
    }
}

function checkSequence() {
    if (interactionSequence.length < UNLOCK_SEQUENCE.length) {
        return false;
    }
    
    for (let i = 0; i < UNLOCK_SEQUENCE.length; i++) {
        if (interactionSequence[i] !== UNLOCK_SEQUENCE[i]) {
            return false;
        }
    }
    
    return true;
}

function unlockLiveOps(liveOpsNav, accessDenied, liveOpsContent) {
    // Remove lock indicator
    const lockIndicator = liveOpsNav.querySelector('.locked-indicator');
    if (lockIndicator) {
        lockIndicator.style.transition = 'opacity 0.5s ease';
        lockIndicator.style.opacity = '0';
        setTimeout(() => lockIndicator.remove(), 500);
    }
    
    // Hide access denied, show content
    accessDenied.style.transition = 'opacity 0.5s ease';
    accessDenied.style.opacity = '0';
    setTimeout(() => {
        accessDenied.classList.add('hidden');
        liveOpsContent.classList.remove('hidden');
        liveOpsContent.style.opacity = '0';
        setTimeout(() => {
            liveOpsContent.style.transition = 'opacity 1s ease';
            liveOpsContent.style.opacity = '1';
        }, 50);
    }, 500);
    
    // Play unlock sound
    const glitchSound = document.getElementById('glitch-sound');
    if (glitchSound) {
        glitchSound.volume = 0.4;
        glitchSound.play().catch(() => {});
    }
    
    // Add unlock message to terminal if visible
    const terminalOutput = document.getElementById('terminal-output');
    if (terminalOutput) {
        const msg = document.createElement('div');
        msg.className = 'terminal-line';
        msg.textContent = '[SYSTEM] CLEARANCE GRANTED: LIVE OPS ACCESS ENABLED';
        msg.style.color = '#00ff88';
        terminalOutput.appendChild(msg);
    }
}

function updateMissionFeeds() {
    const feedItems = document.querySelectorAll('.feed-item');
    if (feedItems.length === 0) return;
    
    // Randomly update a feed item
    const randomItem = feedItems[Math.floor(Math.random() * feedItems.length)];
    const statusTag = randomItem.querySelector('.status-tag');
    
    if (Math.random() > 0.7) {
        // Change status
        if (statusTag.classList.contains('active')) {
            statusTag.classList.remove('active');
            statusTag.classList.add('silenced');
            statusTag.textContent = 'SILENCED';
        } else {
            statusTag.classList.remove('silenced');
            statusTag.classList.add('active');
            statusTag.textContent = 'ACTIVE';
        }
    }
}
