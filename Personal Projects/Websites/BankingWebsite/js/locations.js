document.addEventListener('DOMContentLoaded', function() {
    // Initialize the map
    const map = L.map('map').setView([37.7749, -122.4194], 3);

    // Add a custom dark theme tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    // Custom icon for markers
    const lunaIcon = L.divIcon({
        className: 'luna-marker',
        html: `<div class="marker-pulse"></div>`,
        iconSize: [20, 20]
    });

    // Sample locations data (replace with actual data)
    const locations = [
        { name: 'LUNA HQ', lat: 37.7749, lng: -122.4194, type: 'hq' },
        { name: 'London Center', lat: 51.5074, lng: -0.1278, type: 'center' },
        { name: 'Singapore Hub', lat: 1.3521, lng: 103.8198, type: 'hub' },
        { name: 'Tokyo Branch', lat: 35.6762, lng: 139.6503, type: 'branch' },
        { name: 'Dubai Office', lat: 25.2048, lng: 55.2708, type: 'office' },
        { name: 'Sydney Center', lat: -33.8688, lng: 151.2093, type: 'center' }
    ];

    // Add markers to the map
    locations.forEach(location => {
        const marker = L.marker([location.lat, location.lng], { icon: lunaIcon })
            .bindPopup(`
                <div class="location-popup">
                    <h3>${location.name}</h3>
                    <p>Type: ${location.type}</p>
                    <button onclick="getDirections(${location.lat}, ${location.lng})">Get Directions</button>
                </div>
            `);
        marker.addTo(map);
    });

    // Search functionality
    const searchBtn = document.querySelector('.search-btn');
    const locationInput = document.getElementById('location-input');

    searchBtn.addEventListener('click', function() {
        const searchQuery = locationInput.value;
        if (searchQuery) {
            // In a real application, you would:
            // 1. Geocode the search query to get coordinates
            // 2. Find the nearest LUNA location
            // 3. Pan and zoom the map to show results
            alert('Search functionality would search for: ' + searchQuery);
        }
    });

    // Add custom CSS for the markers
    const style = document.createElement('style');
    style.textContent = `
        .luna-marker {
            position: relative;
        }

        .marker-pulse {
            width: 20px;
            height: 20px;
            background: rgba(138, 180, 255, 0.6);
            border-radius: 50%;
            position: relative;
            animation: pulse 2s infinite;
        }

        .marker-pulse::after {
            content: '';
            position: absolute;
            width: 10px;
            height: 10px;
            background: #8ab4ff;
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .location-popup {
            text-align: center;
            padding: 10px;
        }

        .location-popup h3 {
            color: #333;
            margin-bottom: 5px;
        }

        .location-popup button {
            background: #8ab4ff;
            border: none;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            margin-top: 10px;
            cursor: pointer;
        }

        @keyframes pulse {
            0% {
                transform: scale(1);
                opacity: 0.6;
            }
            50% {
                transform: scale(1.5);
                opacity: 0;
            }
            100% {
                transform: scale(1);
                opacity: 0.6;
            }
        }
    `;
    document.head.appendChild(style);
});

// Function to handle getting directions
function getDirections(lat, lng) {
    // In a real application, you would:
    // 1. Get user's current location
    // 2. Open in maps application or show directions
    window.open(`https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}`);
}