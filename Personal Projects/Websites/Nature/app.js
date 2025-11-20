// Global variables
let scene, camera, renderer, earth, cloudLayer;
let pins = [];
let raycaster, mouse;
let isRotating = true;
let animationId;
let isDragging = false;
let previousMousePosition = { x: 0, y: 0 };
let hasDragged = false;
let clusterMenu = null;

// Initialize the scene
function init() {
    // Create scene
    scene = new THREE.Scene();
    
    // Create camera
    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.z = 2.5;
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Clear existing canvas if any
    const container = document.getElementById('canvas-container');
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(5, 3, 5);
    scene.add(directionalLight);
    
    // Create Earth
    createEarth();
    
    // Create location pins
    createPins();
    
    // Setup raycaster for click detection
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    
    // Event listeners
    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('click', onCanvasClick);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('mousedown', onMouseDown);
    renderer.domElement.addEventListener('mouseup', onMouseUp);
    renderer.domElement.addEventListener('mouseleave', onMouseUp);
    
    const rotateBtn = document.getElementById('rotate-toggle');
    if (rotateBtn) rotateBtn.addEventListener('click', toggleRotation);
    
    const resetBtn = document.getElementById('reset-view');
    if (resetBtn) resetBtn.addEventListener('click', resetView);
    
    // Start animation
    animate();
}

// Create Earth with texture
function createEarth() {
    const geometry = new THREE.SphereGeometry(1, 64, 64);
    
    const textureLoader = new THREE.TextureLoader();
    
    // Earth material with real textures
    const material = new THREE.MeshPhongMaterial({
        map: textureLoader.load('https://unpkg.com/three-globe@2.31.0/example/img/earth-blue-marble.jpg'),
        bumpMap: textureLoader.load('https://unpkg.com/three-globe@2.31.0/example/img/earth-topology.png'),
        bumpScale: 0.01,
        specularMap: textureLoader.load('https://unpkg.com/three-globe@2.31.0/example/img/earth-water.png'),
        specular: new THREE.Color(0x444444),
        shininess: 15
    });
    
    earth = new THREE.Mesh(geometry, material);
    scene.add(earth);
    
    // Add clouds layer
    createCloudLayer(textureLoader);
    
    // Add country borders
    createCountryBorders();
}

// Create cloud layer
function createCloudLayer(textureLoader) {
    const geometry = new THREE.SphereGeometry(1.01, 64, 64);
    const material = new THREE.MeshPhongMaterial({
        map: textureLoader.load('https://unpkg.com/three-globe@2.31.0/example/img/earth-clouds.png'),
        transparent: true,
        opacity: 0.4,
        depthWrite: false,
        side: THREE.DoubleSide
    });
    
    cloudLayer = new THREE.Mesh(geometry, material);
    earth.add(cloudLayer);
}

// Create country borders as lines
function createCountryBorders() {
    const borderGroup = new THREE.Group();
    
    // Simple border lines (latitude and longitude lines to show country divisions)
    const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x666666,
        transparent: true,
        opacity: 0.2
    });
    
    // Latitude lines
    for (let lat = -80; lat <= 80; lat += 10) {
        const points = [];
        for (let lon = 0; lon <= 360; lon += 5) {
            const phi = (90 - lat) * Math.PI / 180;
            const theta = lon * Math.PI / 180;
            const x = 1.002 * Math.sin(phi) * Math.cos(theta);
            const y = 1.002 * Math.cos(phi);
            const z = 1.002 * Math.sin(phi) * Math.sin(theta);
            points.push(new THREE.Vector3(x, y, z));
        }
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, lineMaterial);
        borderGroup.add(line);
    }
    
    // Longitude lines
    for (let lon = 0; lon < 360; lon += 10) {
        const points = [];
        for (let lat = -90; lat <= 90; lat += 5) {
            const phi = (90 - lat) * Math.PI / 180;
            const theta = lon * Math.PI / 180;
            const x = 1.002 * Math.sin(phi) * Math.cos(theta);
            const y = 1.002 * Math.cos(phi);
            const z = 1.002 * Math.sin(phi) * Math.sin(theta);
            points.push(new THREE.Vector3(x, y, z));
        }
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, lineMaterial);
        borderGroup.add(line);
    }
    
    earth.add(borderGroup);
}

// Convert lat/lon to 3D position
function latLonToVector3(lat, lon, radius = 1.02) {
    const phi = (90 - lat) * Math.PI / 180;
    const theta = (lon + 180) * Math.PI / 180;
    
    const x = -radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.cos(phi);
    const z = radius * Math.sin(phi) * Math.sin(theta);
    
    return new THREE.Vector3(x, y, z);
}

// Create pins for locations
function createPins() {
    if (typeof locations === 'undefined') {
        console.error('Locations data not found. Make sure data.js is loaded.');
        return;
    }

    locations.forEach(location => {
        const pinGroup = new THREE.Group();
        
        // Pin marker (cone)
        const pinGeometry = new THREE.ConeGeometry(0.015, 0.06, 8);
        const pinMaterial = new THREE.MeshPhongMaterial({
            color: location.type === 'volcano' ? 0xff4444 : 0x44ff44,
            emissive: location.type === 'volcano' ? 0xff0000 : 0x00ff00,
            emissiveIntensity: 0.5
        });
        const pin = new THREE.Mesh(pinGeometry, pinMaterial);
        
        // Position pin at location
        const position = latLonToVector3(location.lat, location.lon);
        pinGroup.position.copy(position);
        
        // Orient pin to point outward from Earth
        pinGroup.lookAt(position.clone().multiplyScalar(2));
        pin.rotation.x = Math.PI;
        
        pinGroup.add(pin);
        
        // Add glow sphere
        const glowGeometry = new THREE.SphereGeometry(0.025, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: location.type === 'volcano' ? 0xff4444 : 0x44ff44,
            transparent: true,
            opacity: 0.6
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        pinGroup.add(glow);
        
        // Store location data
        pinGroup.userData = location;
        pinGroup.userData.glow = glow;
        pinGroup.userData.originalScale = pinGroup.scale.clone();
        
        earth.add(pinGroup);
        pins.push(pinGroup);
    });
}

// Animation loop
function animate() {
    animationId = requestAnimationFrame(animate);
    
    // Rotate Earth
    if (isRotating && earth) {
        earth.rotation.y += 0.0005;
    }
    
    // Animate cloud layer
    if (cloudLayer) {
        cloudLayer.rotation.y += 0.0007;
    }
    
    // Animate pins (blinking effect)
    pins.forEach((pin, index) => {
        const time = Date.now() * 0.001;
        const offset = index * 0.5;
        const pulse = Math.sin(time * 3 + offset) * 0.5 + 0.5;
        
        if (pin.userData.glow) {
            pin.userData.glow.material.opacity = 0.3 + pulse * 0.5;
            pin.userData.glow.scale.setScalar(0.8 + pulse * 0.4);
        }
    });
    
    renderer.render(scene, camera);
}

// Handle window resize
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Handle canvas click
function onCanvasClick(event) {
    if (hasDragged) return;

    // Close existing cluster menu if clicking elsewhere
    if (clusterMenu) {
        closeClusterMenu();
    }

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(pins, true);
    
    if (intersects.length > 0) {
        let clickedPin = intersects[0].object;
        while (clickedPin.parent && !clickedPin.userData.name) {
            clickedPin = clickedPin.parent;
        }
        
        if (clickedPin.userData.name) {
            // Check for nearby pins (cluster detection)
            const nearbyPins = findNearbyPins(clickedPin, 0.15);
            
            if (nearbyPins.length > 1) {
                // Show cluster menu
                showClusterMenu(nearbyPins, event.clientX, event.clientY);
            } else {
                // Single pin - show info directly
                displayLocationInfo(clickedPin.userData);
                
                // Highlight effect
                pins.forEach(p => p.scale.copy(p.userData.originalScale));
                clickedPin.scale.multiplyScalar(1.5);
            }
        }
    }
}

// Find nearby pins within a certain distance
function findNearbyPins(centerPin, maxDistance) {
    const nearbyPins = [];
    const centerPos = centerPin.position;
    
    pins.forEach(pin => {
        const distance = centerPos.distanceTo(pin.position);
        if (distance <= maxDistance) {
            nearbyPins.push(pin);
        }
    });
    
    return nearbyPins;
}

// Show cluster menu with radial layout
function showClusterMenu(nearbyPins, x, y) {
    // Create menu container
    clusterMenu = document.createElement('div');
    clusterMenu.className = 'cluster-menu';
    clusterMenu.style.left = x + 'px';
    clusterMenu.style.top = y + 'px';
    
    // Add title
    const title = document.createElement('div');
    title.className = 'cluster-menu-title';
    title.textContent = `${nearbyPins.length} Locations Nearby`;
    clusterMenu.appendChild(title);
    
    // Add location items
    nearbyPins.forEach((pin, index) => {
        const item = document.createElement('div');
        item.className = 'cluster-menu-item';
        item.style.animationDelay = `${index * 0.05}s`;
        
        const icon = pin.userData.type === 'volcano' ? '🌋' : '⛰️';
        item.innerHTML = `
            <span class="cluster-icon">${icon}</span>
            <span class="cluster-name">${pin.userData.name}</span>
            <span class="cluster-country">${pin.userData.country}</span>
        `;
        
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            displayLocationInfo(pin.userData);
            pins.forEach(p => p.scale.copy(p.userData.originalScale));
            pin.scale.multiplyScalar(1.5);
            closeClusterMenu();
        });
        
        clusterMenu.appendChild(item);
    });
    
    document.body.appendChild(clusterMenu);
    
    // Adjust position if menu goes off-screen
    setTimeout(() => {
        const rect = clusterMenu.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            clusterMenu.style.left = (x - rect.width) + 'px';
        }
        if (rect.bottom > window.innerHeight) {
            clusterMenu.style.top = (y - rect.height) + 'px';
        }
    }, 10);
}

// Close cluster menu
function closeClusterMenu() {
    if (clusterMenu) {
        clusterMenu.remove();
        clusterMenu = null;
    }
}

// Handle mouse down for dragging
function onMouseDown(event) {
    isDragging = true;
    hasDragged = false;
    previousMousePosition = {
        x: event.clientX,
        y: event.clientY
    };
    document.body.style.cursor = 'grabbing';
}

// Handle mouse up to stop dragging
function onMouseUp(event) {
    isDragging = false;
    document.body.style.cursor = 'default';
}

// Handle mouse move for hover effect and dragging
function onMouseMove(event) {
    // Handle rotation if dragging
    if (isDragging) {
        const deltaMove = {
            x: event.clientX - previousMousePosition.x,
            y: event.clientY - previousMousePosition.y
        };
        
        if (Math.abs(deltaMove.x) > 2 || Math.abs(deltaMove.y) > 2) {
            hasDragged = true;
        }

        const rotateSpeed = 0.005;
        
        earth.rotation.y += deltaMove.x * rotateSpeed;
        earth.rotation.x += deltaMove.y * rotateSpeed;
        
        // Limit vertical rotation to avoid flipping
        earth.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, earth.rotation.x));

        previousMousePosition = {
            x: event.clientX,
            y: event.clientY
        };
    }

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(pins, true);
    
    if (intersects.length > 0) {
        document.body.style.cursor = 'pointer';
    } else {
        document.body.style.cursor = isDragging ? 'grabbing' : 'default';
    }
}

// Display location information
function displayLocationInfo(location) {
    const infoDiv = document.getElementById('location-info');
    if (!infoDiv) return;
    
    infoDiv.innerHTML = `
        <div class="location-detail">
            <h3>${location.name}</h3>
            <p><strong>Type:</strong> ${location.type.charAt(0).toUpperCase() + location.type.slice(1)}</p>
            <p><strong>Country:</strong> ${location.country}</p>
            <p><strong>Elevation:</strong> ${location.elevation}</p>
            <p class="coordinates">
                <strong>Coordinates:</strong><br>
                Latitude: ${location.lat.toFixed(4)}°<br>
                Longitude: ${location.lon.toFixed(4)}°
            </p>
            ${location.description ? `<p style="margin-top: 10px;">${location.description}</p>` : ''}
            ${location.imageUrl ? `<img src="${location.imageUrl}" alt="${location.name}" class="location-image" loading="lazy">` : ''}
            ${location.wikiUrl ? `<a href="${location.wikiUrl}" ${location.wikiUrl.startsWith('locations/') ? '' : 'target="_blank"'} class="learn-more-btn">🔍 Discover More</a>` : ''}
        </div>
    `;
}

// Toggle rotation
function toggleRotation() {
    isRotating = !isRotating;
    const btn = document.getElementById('rotate-toggle');
    if (btn) {
        btn.textContent = isRotating ? '⏸️ Pause Rotation' : '▶️ Resume Rotation';
    }
}

// Reset view
function resetView() {
    camera.position.set(0, 0, 2.5);
    camera.lookAt(0, 0, 0);
    if (earth) earth.rotation.set(0, 0, 0);
    pins.forEach(p => p.scale.copy(p.userData.originalScale));
    const infoDiv = document.getElementById('location-info');
    if (infoDiv) infoDiv.innerHTML = '';
}

// Start the application
init();
