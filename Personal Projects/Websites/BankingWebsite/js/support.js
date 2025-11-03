// Support page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Support option buttons
    const chatBtn = document.querySelector('.support-btn.chat');
    const connectBtn = document.querySelector('.support-btn.connect');

    chatBtn.addEventListener('click', () => {
        alert('AI chat interface with NOVA will be implemented here');
    });

    connectBtn.addEventListener('click', () => {
        alert('Connection to human expert will be implemented here');
    });

    // Smart search functionality
    const searchBtn = document.querySelector('.search-btn');
    const searchInput = document.querySelector('.search-box input');
    const topicBtns = document.querySelectorAll('.topic-btn');

    searchBtn.addEventListener('click', () => {
        if (searchInput.value.trim()) {
            alert(`Searching knowledge base for: ${searchInput.value}`);
        }
    });

    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && searchInput.value.trim()) {
            alert(`Searching knowledge base for: ${searchInput.value}`);
        }
    });

    topicBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            searchInput.value = btn.textContent;
            searchBtn.click();
        });
    });

    // Location finder
    const locateBtn = document.querySelector('.locate-btn');
    locateBtn.addEventListener('click', () => {
        //alert('Location finder will be implemented here');
        window.location.href = 'locations.html';
    });

    // Quick help links
    // const helpLinks = document.querySelectorAll('.help-card ul li a');
    // helpLinks.forEach(link => {
    //     link.addEventListener('click', (e) => {
    //         e.preventDefault();
    //         alert(`Help article for "${link.textContent}" will be displayed here`);
    //     });
    // });

    // Animated map overlay
    const mapOverlay = document.querySelector('.map-overlay');
    let points = [];
    
    // Create random points for the map
    for (let i = 0; i < 20; i++) {
        const point = document.createElement('div');
        point.className = 'map-point';
        point.style.left = `${Math.random() * 100}%`;
        point.style.top = `${Math.random() * 100}%`;
        point.style.animation = `pulse ${2 + Math.random() * 2}s infinite`;
        points.push(point);
        document.querySelector('.location-map').appendChild(point);
    }
});