// ===================================
// 3D Model Viewer with Scroll Animation
// ===================================

class ScrollDriven3DViewer {
    constructor() {
        this.canvas = document.getElementById('canvas3d');
        this.ctx = this.canvas.getContext('2d');
        this.rotation = { x: 0, y: 0, z: 0 };
        this.position = { x: 0, y: 0, z: 0 };
        this.scale = 1;
        
        this.setupCanvas();
        this.createGeometry();
        this.animate();
        
        window.addEventListener('resize', () => this.setupCanvas());
        window.addEventListener('scroll', () => this.updateFromScroll());
    }
    
    setupCanvas() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        this.centerX = rect.width / 2;
        this.centerY = rect.height / 2;
    }
    
    createGeometry() {
        // Create a stylized museum sculpture-like geometry
        this.vertices = [];
        this.faces = [];
        
        // Create a twisted torus/sculpture shape
        const rings = 20;
        const segments = 16;
        const radius = 100;
        const tubeRadius = 40;
        
        for (let i = 0; i <= rings; i++) {
            const v = i / rings;
            const phi = v * Math.PI * 2;
            
            for (let j = 0; j <= segments; j++) {
                const u = j / segments;
                const theta = u * Math.PI * 2;
                
                // Add twist based on phi
                const twist = phi * 1.5;
                
                const x = (radius + tubeRadius * Math.cos(theta)) * Math.cos(phi);
                const y = (radius + tubeRadius * Math.cos(theta)) * Math.sin(phi);
                const z = tubeRadius * Math.sin(theta) + Math.sin(twist) * 30;
                
                this.vertices.push({ x, y, z });
            }
        }
        
        // Create faces
        for (let i = 0; i < rings; i++) {
            for (let j = 0; j < segments; j++) {
                const a = i * (segments + 1) + j;
                const b = a + segments + 1;
                const c = a + 1;
                const d = b + 1;
                
                this.faces.push([a, b, c]);
                this.faces.push([b, d, c]);
            }
        }
    }
    
    updateFromScroll() {
        const scrollPercent = window.scrollY / (document.documentElement.scrollHeight - window.innerHeight);
        
        // Update rotation based on scroll
        this.rotation.y = scrollPercent * Math.PI * 4;
        this.rotation.x = Math.sin(scrollPercent * Math.PI * 2) * 0.5;
        this.rotation.z = scrollPercent * Math.PI * 0.5;
        
        // Update position
        this.position.x = Math.sin(scrollPercent * Math.PI * 2) * 100;
        this.position.y = (scrollPercent - 0.5) * 200;
        
        // Update scale
        this.scale = 1 + Math.sin(scrollPercent * Math.PI) * 0.3;
    }
    
    rotateVertex(vertex) {
        let x = vertex.x;
        let y = vertex.y;
        let z = vertex.z;
        
        // Rotate around Y axis
        let tempX = x * Math.cos(this.rotation.y) - z * Math.sin(this.rotation.y);
        let tempZ = x * Math.sin(this.rotation.y) + z * Math.cos(this.rotation.y);
        x = tempX;
        z = tempZ;
        
        // Rotate around X axis
        let tempY = y * Math.cos(this.rotation.x) - z * Math.sin(this.rotation.x);
        tempZ = y * Math.sin(this.rotation.x) + z * Math.cos(this.rotation.x);
        y = tempY;
        z = tempZ;
        
        // Rotate around Z axis
        tempX = x * Math.cos(this.rotation.z) - y * Math.sin(this.rotation.z);
        tempY = x * Math.sin(this.rotation.z) + y * Math.cos(this.rotation.z);
        x = tempX;
        y = tempY;
        
        return { x, y, z };
    }
    
    project(vertex) {
        const rotated = this.rotateVertex(vertex);
        const scale = this.scale * 2;
        const perspective = 800;
        const z = rotated.z + perspective;
        
        const x = (rotated.x * perspective) / z * scale + this.centerX + this.position.x;
        const y = (rotated.y * perspective) / z * scale + this.centerY + this.position.y;
        
        return { x, y, z: rotated.z };
    }
    
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Project all vertices
        const projected = this.vertices.map(v => this.project(v));
        
        // Sort faces by average z depth
        const sortedFaces = this.faces.map(face => {
            const avgZ = (projected[face[0]].z + projected[face[1]].z + projected[face[2]].z) / 3;
            return { face, avgZ };
        }).sort((a, b) => a.avgZ - b.avgZ);
        
        // Draw faces
        sortedFaces.forEach(({ face, avgZ }) => {
            const p1 = projected[face[0]];
            const p2 = projected[face[1]];
            const p3 = projected[face[2]];
            
            // Calculate lighting based on depth
            const brightness = Math.max(0, Math.min(1, (avgZ + 200) / 400));
            const alpha = 0.6 * brightness;
            
            this.ctx.beginPath();
            this.ctx.moveTo(p1.x, p1.y);
            this.ctx.lineTo(p2.x, p2.y);
            this.ctx.lineTo(p3.x, p3.y);
            this.ctx.closePath();
            
            // Use colors from palette
            const hue = 330; // Pink/purple range
            this.ctx.fillStyle = `hsla(${hue}, 30%, ${30 + brightness * 40}%, ${alpha})`;
            this.ctx.fill();
            
            this.ctx.strokeStyle = `hsla(${hue}, 40%, ${40 + brightness * 30}%, ${alpha * 0.5})`;
            this.ctx.lineWidth = 0.5;
            this.ctx.stroke();
        });
        
        requestAnimationFrame(() => this.animate());
    }
}

// ===================================
// Scroll Navigation & Active States
// ===================================

class NavigationController {
    constructor() {
        this.sections = document.querySelectorAll('.section');
        this.navLinks = document.querySelectorAll('.nav-link');
        
        this.setupIntersectionObserver();
        this.setupSmoothScroll();
        this.setupScrollAnimations();
    }
    
    setupIntersectionObserver() {
        const options = {
            threshold: 0.3,
            rootMargin: '-100px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Update active nav link
                    const id = entry.target.getAttribute('id');
                    this.updateActiveLink(id);
                    
                    // Add visible class for animations
                    entry.target.classList.add('visible');
                }
            });
        }, options);
        
        this.sections.forEach(section => observer.observe(section));
    }
    
    updateActiveLink(activeId) {
        this.navLinks.forEach(link => {
            const href = link.getAttribute('href').substring(1);
            if (href === activeId) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
    
    setupSmoothScroll() {
        this.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href');
                const targetSection = document.querySelector(targetId);
                
                if (targetSection) {
                    targetSection.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }
    
    setupScrollAnimations() {
        // Add initial visible class to hero
        const hero = document.querySelector('.hero-section');
        if (hero) {
            hero.classList.add('visible');
        }
    }
}

// ===================================
// Particle Background Effect
// ===================================

class ParticleBackground {
    constructor() {
        this.particles = [];
        this.particleCount = 50;
        this.canvas = document.getElementById('canvas3d');
        this.ctx = this.canvas.getContext('2d');
        
        this.createParticles();
        this.animateParticles();
    }
    
    createParticles() {
        for (let i = 0; i < this.particleCount; i++) {
            this.particles.push({
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1
            });
        }
    }
    
    animateParticles() {
        // This would run alongside the 3D viewer
        // Disabled by default to keep focus on the sculpture
    }
}

// ===================================
// Initialize Everything
// ===================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize 3D viewer
    const viewer = new ScrollDriven3DViewer();
    
    // Initialize navigation
    const nav = new NavigationController();
    
    // Add loading animation
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.8s ease-out';
        document.body.style.opacity = '1';
    }, 100);
    
    // Performance optimization: throttle scroll events
    let scrollTimeout;
    let lastScrollY = window.scrollY;
    
    window.addEventListener('scroll', () => {
        const currentScrollY = window.scrollY;
        const scrollDelta = Math.abs(currentScrollY - lastScrollY);
        
        // Only update if significant scroll occurred
        if (scrollDelta > 5) {
            lastScrollY = currentScrollY;
        }
    }, { passive: true });
});

// ===================================
// Utility: Debounce Function
// ===================================

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ===================================
// Easter Eggs & Interactions
// ===================================

// Add subtle parallax to cards
document.addEventListener('mousemove', debounce((e) => {
    const cards = document.querySelectorAll('.project-card, .version-card, .course-card');
    const mouseX = e.clientX / window.innerWidth - 0.5;
    const mouseY = e.clientY / window.innerHeight - 0.5;
    
    cards.forEach(card => {
        const rect = card.getBoundingClientRect();
        const cardX = rect.left + rect.width / 2;
        const cardY = rect.top + rect.height / 2;
        
        const deltaX = (e.clientX - cardX) / 50;
        const deltaY = (e.clientY - cardY) / 50;
        
        card.style.transform = `perspective(1000px) rotateY(${deltaX}deg) rotateX(${-deltaY}deg) translateY(-5px)`;
    });
}, 10));

// Reset card transforms when mouse leaves
document.addEventListener('mouseleave', () => {
    const cards = document.querySelectorAll('.project-card, .version-card, .course-card');
    cards.forEach(card => {
        card.style.transform = '';
    });
});
