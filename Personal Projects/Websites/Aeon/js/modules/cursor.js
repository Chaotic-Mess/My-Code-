// ═══════════════════════════════════════════════════════════
// CURSOR EFFECTS - The watching eye
// ═══════════════════════════════════════════════════════════

export function initCursorEffects() {
    const canvas = document.getElementById('cursor-distortion');
    const ctx = canvas.getContext('2d');
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    let mouseX = 0;
    let mouseY = 0;
    const particles = [];
    
    // Custom cursor position tracking
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        
        // Update custom cursor
        const after = document.body;
        after.style.setProperty('--mouse-x', mouseX + 'px');
        after.style.setProperty('--mouse-y', mouseY + 'px');
        
        // Create distortion particles
        if (Math.random() > 0.9) {
            particles.push({
                x: mouseX,
                y: mouseY,
                size: Math.random() * 3 + 1,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                life: 1
            });
        }
    });
    
    // Animate distortion
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw distortion field around cursor
        const gradient = ctx.createRadialGradient(mouseX, mouseY, 0, mouseX, mouseY, 100);
        gradient.addColorStop(0, 'rgba(0, 255, 136, 0.1)');
        gradient.addColorStop(1, 'rgba(0, 255, 136, 0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw and update particles
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            ctx.fillStyle = `rgba(0, 255, 136, ${p.life})`;
            ctx.fillRect(p.x, p.y, p.size, p.size);
            
            p.x += p.vx;
            p.y += p.vy;
            p.life -= 0.02;
            
            if (p.life <= 0) {
                particles.splice(i, 1);
            }
        }
        
        requestAnimationFrame(animate);
    }
    
    animate();
    
    // Resize handler
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}
