// Wallets page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Card carousel functionality
    const cardsContainer = document.querySelector('.cards-container');
    const prevBtn = document.querySelector('.carousel-btn.prev');
    const nextBtn = document.querySelector('.carousel-btn.next');
    const cardWidth = 320; // card width + gap

    prevBtn.addEventListener('click', () => {
        cardsContainer.scrollLeft -= cardWidth;
    });

    nextBtn.addEventListener('click', () => {
        cardsContainer.scrollLeft += cardWidth;
    });

    // Add new card button functionality
    const addCardBtn = document.querySelector('.add-card-btn');
    addCardBtn.addEventListener('click', () => {
        alert('Add new card functionality will be implemented here');
    });

    // Quick action buttons
    const quickActionBtns = document.querySelectorAll('.quick-actions button');
    quickActionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            alert(`${btn.textContent} functionality will be implemented here`);
        });
    });
});