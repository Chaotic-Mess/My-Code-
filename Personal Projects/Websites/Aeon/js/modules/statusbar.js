// ═══════════════════════════════════════════════════════════
// STATUS BAR - Real-time system information
// ═══════════════════════════════════════════════════════════

export function updateStatusBar() {
    const timeElement = document.getElementById('current-time');
    if (!timeElement) return;
    
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    
    timeElement.textContent = `${hours}:${minutes}:${seconds} UTC`;
}
