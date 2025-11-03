// 404 Page Animations
document.addEventListener('DOMContentLoaded', function() {
    // Add entrance animation
    const errorContainer = document.querySelector('.error-container');
    if (errorContainer) {
        errorContainer.style.opacity = '0';
        errorContainer.style.transform = 'translateY(30px)';
        
        setTimeout(() => {
            errorContainer.style.transition = 'opacity 1s ease, transform 1s ease';
            errorContainer.style.opacity = '1';
            errorContainer.style.transform = 'translateY(0)';
        }, 100);
    }

    // Animate quick links on scroll
    const quickLinks = document.querySelectorAll('.quick-link');
    quickLinks.forEach((link, index) => {
        link.style.opacity = '0';
        link.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            link.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            link.style.opacity = '1';
            link.style.transform = 'translateY(0)';
        }, 1000 + (index * 100));
    });

    // Log 404 error (could be sent to analytics)
    console.log('404 Error - Page not found:', window.location.href);
});