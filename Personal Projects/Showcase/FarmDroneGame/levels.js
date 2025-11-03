// Level definitions for Farm Drone Automation

const levels = [
    {
        id: 1,
        title: "Level 1: First Steps",
        objective: "Plant 3 seeds and harvest 3 crops. Learn basic movement and actions.",
        gridSize: 8,
        setup: (game) => {
            game.resources.seeds = 5;
            game.drone.x = 0;
            game.drone.y = 0;
        },
        checkComplete: (game) => {
            return game.resources.harvested >= 3;
        },
        starterCode: `// Welcome to Farm Drone Automation!
// Goal: Plant and harvest 3 crops

void loop() {
    // Scan the current cell
    if (scan() == EMPTY) {
        plant();
    }
    
    // Check if crop is ready to harvest
    if (scan() == GROWN) {
        harvest();
    }
    
    // Move forward
    moveForward();
}`
    },
    
    {
        id: 2,
        title: "Level 2: Grid Navigation",
        objective: "Plant seeds in a 3x3 pattern. Learn to turn and navigate efficiently.",
        gridSize: 8,
        setup: (game) => {
            game.resources.seeds = 15;
            game.drone.x = 2;
            game.drone.y = 2;
        },
        checkComplete: (game) => {
            let plantedCount = 0;
            for (let y = 2; y < 5; y++) {
                for (let x = 2; x < 5; x++) {
                    const cell = game.grid[y][x];
                    if (cell.state === 'PLANTED' || cell.state === 'GROWING' || cell.state === 'GROWN') {
                        plantedCount++;
                    }
                }
            }
            return plantedCount >= 9;
        },
        starterCode: `// Goal: Plant a 3x3 grid of seeds
// Hint: Use turnRight() to change direction

void loop() {
    if (scan() == EMPTY) {
        plant();
    }
    
    // Add your navigation logic here
    moveForward();
    turnRight();
}`
    },
    
    {
        id: 3,
        title: "Level 3: Conditional Harvesting",
        objective: "Harvest 5 crops while avoiding obstacles. Use scanAhead() to check before moving.",
        gridSize: 10,
        setup: (game) => {
            game.resources.seeds = 8;
            game.drone.x = 0;
            game.drone.y = 0;
            
            // Add some pre-planted crops
            game.grid[2][3].state = 'GROWN';
            game.grid[3][5].state = 'GROWN';
            game.grid[5][2].state = 'GROWN';
            game.grid[7][6].state = 'GROWN';
            game.grid[8][8].state = 'GROWN';
            
            // Add obstacles
            game.grid[1][1].state = 'OBSTACLE';
            game.grid[4][4].state = 'OBSTACLE';
            game.grid[6][7].state = 'OBSTACLE';
        },
        checkComplete: (game) => {
            return game.resources.harvested >= 5;
        },
        starterCode: `// Goal: Harvest 5 crops while avoiding obstacles
// Use scanAhead() to check before moving

void loop() {
    // Check current cell
    if (scan() == GROWN) {
        harvest();
    }
    
    // Check ahead before moving
    if (scanAhead() != OBSTACLE) {
        moveForward();
    } else {
        turnRight();
    }
}`
    },
    
    {
        id: 4,
        title: "Level 4: Efficient Farming",
        objective: "Plant, water, and harvest 8 crops. Watered crops grow faster!",
        gridSize: 10,
        setup: (game) => {
            game.resources.seeds = 12;
            game.drone.x = 1;
            game.drone.y = 1;
        },
        checkComplete: (game) => {
            return game.resources.harvested >= 8;
        },
        starterCode: `// Goal: Plant, water, and harvest 8 crops
// Watered crops grow 50% faster!

void loop() {
    if (scan() == EMPTY) {
        plant();
        water(); // Water immediately after planting
    }
    
    if (scan() == PLANTED || scan() == GROWING) {
        water(); // Keep watering
    }
    
    if (scan() == GROWN) {
        harvest();
    }
    
    moveForward();
    
    // Add navigation logic
}`
    },
    
    {
        id: 5,
        title: "Level 5: Advanced Patterns",
        objective: "Create a checkerboard pattern of crops. Harvest 10 crops total.",
        gridSize: 10,
        setup: (game) => {
            game.resources.seeds = 25;
            game.drone.x = 0;
            game.drone.y = 0;
        },
        checkComplete: (game) => {
            return game.resources.harvested >= 10;
        },
        starterCode: `// Goal: Create a checkerboard pattern
// Plant only on cells where (x + y) is even

void loop() {
    if (scan() == EMPTY) {
        // Your pattern logic here
        plant();
    }
    
    if (scan() == GROWN) {
        harvest();
    }
    
    moveForward();
    
    // Add smart navigation
}`
    },
    
    {
        id: 6,
        title: "Level 6: Optimization Challenge",
        objective: "Harvest 15 crops with limited energy (150 units). Plan efficiently!",
        gridSize: 12,
        setup: (game) => {
            game.resources.seeds = 20;
            game.resources.energy = 150;
            game.drone.x = 0;
            game.drone.y = 0;
            
            // Add some pre-planted crops at various growth stages
            for (let i = 0; i < 8; i++) {
                const x = Math.floor(Math.random() * 12);
                const y = Math.floor(Math.random() * 12);
                game.grid[y][x].state = ['PLANTED', 'GROWING', 'GROWN'][Math.floor(Math.random() * 3)];
            }
        },
        checkComplete: (game) => {
            return game.resources.harvested >= 15;
        },
        starterCode: `// Goal: Harvest 15 crops with limited energy
// Every action costs energy!
// Movement: 1 energy
// Plant/Harvest: 2 energy
// Water: 1 energy
// Turn: 0.5 energy

void loop() {
    // Write an efficient algorithm
    if (scan() == GROWN) {
        harvest();
    }
    
    if (scan() == EMPTY) {
        plant();
    }
    
    moveForward();
}`
    }
];

// Export levels
if (typeof module !== 'undefined' && module.exports) {
    module.exports = levels;
}
