// Mute Toggle Functionality
document.addEventListener('DOMContentLoaded', function() {
    const muteToggle = document.getElementById('muteToggle');
    const heroVideo = document.getElementById('heroVideo');
    
    if (muteToggle && heroVideo) {
        muteToggle.addEventListener('click', function() {
            heroVideo.muted = !heroVideo.muted;
            
            // Update icon
            const icon = muteToggle.querySelector('i');
            if (heroVideo.muted) {
                icon.className = 'fas fa-volume-mute';
            } else {
                icon.className = 'fas fa-volume-up';
            }
        });
    }
});

// Video cycling for index page
document.addEventListener('DOMContentLoaded', function() {
    const heroVideo = document.getElementById('heroVideo');
    
    if (heroVideo && (window.location.pathname.includes('index.html') || window.location.pathname === '/' || window.location.pathname === '')) {
        const videos = [
            { src: 'video/luna-card-1.mp4', duration: 15000 }, // 15 seconds
            { src: 'video/luna-card-2.mp4', duration: 10000 },  // 10 seconds
            { src: 'video/luna-card-3.mp4', duration: 10000 }  // 10 seconds
        ];
        
        let currentVideoIndex = 0;
        let videoTimer = null;
        let isTransitioning = false;
        
        // Function to reverse and switch video
        function reverseAndSwitch() {
            if (isTransitioning) return;
            isTransitioning = true;
            
            // Pause the video and get current time
            heroVideo.pause();
            const currentTime = heroVideo.currentTime;
            
            // Create reverse effect by seeking backwards
            let reverseTime = currentTime;
            const reverseInterval = setInterval(() => {
                reverseTime -= 0.1; // Go back 0.1 seconds per frame
                if (reverseTime <= Math.max(0, currentTime - 1)) {
                    clearInterval(reverseInterval);
                    startTransition();
                } else {
                    heroVideo.currentTime = reverseTime;
                }
            }, 50);
        }
        
        function startTransition() {
            // Start fading out
            heroVideo.style.transition = 'opacity 1s ease-in-out';
            heroVideo.style.opacity = '0';
            
            setTimeout(() => {
                // Switch to NEXT video
                currentVideoIndex = (currentVideoIndex + 1) % videos.length;
                
                console.log('Switching to video:', videos[currentVideoIndex].src, 'index:', currentVideoIndex);
                
                // Change source
                const source = heroVideo.querySelector('source');
                source.src = videos[currentVideoIndex].src;
                heroVideo.load();
                
                // Wait for video to load
                heroVideo.onloadeddata = function() {
                    isTransitioning = false;
                    
                    // Play and fade in
                    heroVideo.play().then(() => {
                        heroVideo.style.opacity = '1';
                        
                        // Set timer for next video based on current video's duration
                        clearTimeout(videoTimer);
                        videoTimer = setTimeout(reverseAndSwitch, videos[currentVideoIndex].duration);
                    }).catch(error => {
                        console.log('Video play error:', error);
                        heroVideo.style.opacity = '1';
                    });
                };
            }, 1000); // Wait for fade out
        }
        
        // Start the cycle with the first video's duration
        videoTimer = setTimeout(reverseAndSwitch, videos[currentVideoIndex].duration);
        
        // Ensure video plays after initial loading
        heroVideo.addEventListener('loadeddata', function() {
            if (!isTransitioning) {
                heroVideo.play().catch(error => {
                    console.log('Video autoplay prevented:', error);
                });
            }
        });
    }
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add animation to feature cards on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe feature cards
document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(card);
    });
});