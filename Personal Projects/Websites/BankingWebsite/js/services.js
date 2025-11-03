// Services page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Service category filtering
    const selectorBtns = document.querySelectorAll('.selector-btn');
    const serviceCategories = document.querySelectorAll('.service-category');

    selectorBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons
            selectorBtns.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            btn.classList.add('active');

            const category = btn.dataset.category;
            
            // Show/hide appropriate categories
            if (category === 'all') {
                serviceCategories.forEach(cat => cat.style.display = 'grid');
            } else {
                serviceCategories.forEach(cat => {
                    if (cat.classList.contains(category)) {
                        cat.style.display = 'grid';
                    } else {
                        cat.style.display = 'none';
                    }
                });
            }
        });
    });

    // Apply button functionality
    const applyBtns = document.querySelectorAll('.apply-btn');
    applyBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const serviceName = btn.closest('.service-card').querySelector('h3').textContent;
            alert(`Application process for ${serviceName} will be implemented here`);
        });
    });

    // Chat with NOVA button
    const chatBtn = document.querySelector('.chat-btn');
    chatBtn.addEventListener('click', () => {
        alert('AI chat interface with NOVA will be implemented here');
    });

    // Add hover effect for service cards
    const serviceCards = document.querySelectorAll('.service-card');
    serviceCards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const glow = card.querySelector('.card-glow');
            glow.style.background = `radial-gradient(circle at ${x}px ${y}px, 
                rgba(138, 180, 255, 0.15),
                transparent 70%)`;
        });
    });
});