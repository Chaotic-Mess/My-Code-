// ═══════════════════════════════════════════════════════════
// NAVIGATION SYSTEM - Panel switching with weight
// ═══════════════════════════════════════════════════════════

export function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const contentPanels = document.querySelectorAll('.content-panel');
    const hoverStatic = document.getElementById('hover-static');
    const selectBass = document.getElementById('select-bass');
    
    navItems.forEach(item => {
        item.addEventListener('mouseenter', () => {
            // Play hover sound
            if (hoverStatic) {
                hoverStatic.currentTime = 0;
                hoverStatic.volume = 0.2;
                hoverStatic.play().catch(() => {});
            }
        });
        
        item.addEventListener('click', () => {
            const section = item.getAttribute('data-section');
            
            // Check if locked
            if (item.querySelector('.locked-indicator') && section === 'live-ops') {
                const liveOpsSection = document.getElementById('live-ops-section');
                const accessDenied = liveOpsSection.querySelector('.access-denied');
                
                // Check if unlocked
                if (!accessDenied.classList.contains('hidden')) {
                    // Play denied sound
                    if (selectBass) {
                        selectBass.currentTime = 0;
                        selectBass.volume = 0.3;
                        selectBass.playbackRate = 0.5;
                        selectBass.play().catch(() => {});
                    }
                }
            }
            
            // Play select sound
            if (selectBass) {
                selectBass.currentTime = 0;
                selectBass.volume = 0.4;
                selectBass.playbackRate = 1;
                selectBass.play().catch(() => {});
            }
            
            // Remove active from all
            navItems.forEach(n => n.classList.remove('active'));
            contentPanels.forEach(p => p.classList.remove('active'));
            
            // Activate clicked
            item.classList.add('active');
            const targetPanel = document.getElementById(`${section}-section`);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });
}
