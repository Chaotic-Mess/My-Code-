// ═══════════════════════════════════════════════════════════
// BOOT SEQUENCE - The awakening
// ═══════════════════════════════════════════════════════════

export async function initBootSequence() {
    return new Promise((resolve) => {
        const bootSequence = document.getElementById('boot-sequence');
        const staticCanvas = document.getElementById('static-canvas');
        const manifesto = document.getElementById('manifesto');
        const vaultDoor = document.getElementById('vault-door');
        const mainInterface = document.getElementById('main-interface');
        
        // Static noise effect
        const ctx = staticCanvas.getContext('2d');
        staticCanvas.width = window.innerWidth;
        staticCanvas.height = window.innerHeight;
        
        let staticInterval = setInterval(() => {
            const imageData = ctx.createImageData(staticCanvas.width, staticCanvas.height);
            for (let i = 0; i < imageData.data.length; i += 4) {
                const color = Math.random() * 255;
                imageData.data[i] = color;
                imageData.data[i + 1] = color;
                imageData.data[i + 2] = color;
                imageData.data[i + 3] = 255;
            }
            ctx.putImageData(imageData, 0, 0);
        }, 50);
        
        // Show manifesto after symbol reveal
        setTimeout(() => {
            manifesto.classList.remove('hidden');
        }, 2500);
        
        // Enable audio on first click
        document.addEventListener('click', () => {
            const ambientHum = document.getElementById('ambient-hum');
            if (ambientHum && ambientHum.paused) {
                ambientHum.volume = 0.3;
                ambientHum.play().catch(() => {});
            }
        }, { once: true });
        
        // Fade out boot sequence
        setTimeout(() => {
            bootSequence.classList.add('fade-out');
            clearInterval(staticInterval);
            
            // Open vault doors
            setTimeout(() => {
                vaultDoor.classList.add('vault-open');
                
                // Show main interface
                setTimeout(() => {
                    mainInterface.classList.remove('hidden');
                    mainInterface.classList.add('visible');
                    bootSequence.style.display = 'none';
                    resolve();
                }, 1000);
            }, 500);
        }, 5500); // Total boot time: ~6 seconds
    });
}
