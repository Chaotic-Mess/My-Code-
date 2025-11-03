// Transactions page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Filter buttons functionality
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons
            filterBtns.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            btn.classList.add('active');
            // Filter functionality would go here
        });
    });

    // Action buttons functionality
    const actionBtns = document.querySelectorAll('.action-btn');
    actionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            alert(`${btn.textContent} functionality will be implemented here`);
        });
    });

    // Pagination functionality
    const pageNumbers = document.querySelectorAll('.page-number');
    pageNumbers.forEach(num => {
        num.addEventListener('click', () => {
            // Remove active class from all numbers
            pageNumbers.forEach(n => n.classList.remove('active'));
            // Add active class to clicked number
            num.classList.add('active');
            // Pagination functionality would go here
        });
    });

    const prevBtn = document.querySelector('.page-btn.prev');
    const nextBtn = document.querySelector('.page-btn.next');

    prevBtn.addEventListener('click', () => {
        // Previous page functionality would go here
        alert('Previous page functionality will be implemented here');
    });

    nextBtn.addEventListener('click', () => {
        // Next page functionality would go here
        alert('Next page functionality will be implemented here');
    });
});