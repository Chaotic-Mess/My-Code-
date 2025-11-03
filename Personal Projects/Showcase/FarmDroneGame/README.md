# 🚜 Farm Drone Automation - Learn C++ Through Fun?

An interactive browser-based game that teaches C++ programming concepts through farm automation! Control farming drones by writing C++ code to plant, water, and harvest crops efficiently.

## 🎮 Game Overview

You play as an engineer tasked with automating a farm after the last human farmer has retired. Instead of manually farming, you write C++ programs that control farming drones to manage fields — planting seeds, watering crops, harvesting produce, and optimizing farm efficiency.

### Key Features

-  **Code-Controlled Drones** - Write real C++ syntax to control farming robots
-  **Progressive Levels** - 6 challenging levels teaching different programming concepts
-  **Visual Feedback** - Watch your code execute in real-time on a grid-based farm
-  **Built-in API Reference** - Complete documentation of available functions
-  **Adjustable Speed** - Control execution speed (1x, 2x, 4x, 8x)
-  **Syntax Highlighting** - Full C++ code editor with syntax highlighting
-  **Goal-Oriented Challenges** - Complete objectives to advance

## 📖 How to Play

### Writing Your First Program

Every drone program must have a `void loop()` function that contains your automation logic:

```cpp
void loop() {
    // Your code here
    if (scan() == EMPTY) {
        plant();
    }
    moveForward();
}
```

### Available Functions

#### Movement
- `moveForward()` - Move one cell forward
- `turnRight()` - Rotate 90° clockwise
- `turnLeft()` - Rotate 90° counter-clockwise

#### Sensing
- `scan()` - Check current cell state
- `scanAhead()` - Check cell ahead without moving

#### Actions
- `plant()` - Plant a seed (requires EMPTY cell)
- `harvest()` - Harvest crops (requires GROWN cell)
- `water()` - Water crops to speed growth

#### Cell States
- `EMPTY` - Empty farmland
- `PLANTED` - Seed planted, not yet growing
- `GROWING` - Crop is growing
- `GROWN` - Ready to harvest
- `OBSTACLE` - Cannot pass

### Controls

- **Run Program** - Execute your code (or press `Ctrl+Enter`)
- **Stop** - Stop execution
- **Reset Level** - Restart the level (or press `Ctrl+R`)
- **Speed** - Adjust execution speed
- **API Reference** - View complete API documentation

## Level Progression

### Level 1: First Steps
Learn basic movement, planting, and harvesting. Plant and harvest 3 crops.

### Level 2: Grid Navigation
Master turning and navigation. Plant seeds in a 3x3 pattern.

### Level 3: Conditional Harvesting
Use `scanAhead()` to avoid obstacles. Harvest 5 pre-planted crops.

### Level 4: Efficient Farming
Learn about watering crops. Plant, water, and harvest 8 crops.

### Level 5: Advanced Patterns
Create a checkerboard pattern. Harvest 10 crops.

### Level 6: Optimization Challenge
Limited energy! Plan efficiently to harvest 15 crops.

## 🛠️ Technical Details

### Project Structure

```
LearnToProgramGame/
├── index.html          # Main HTML structure
├── styles.css          # Styling and layout
├── parser.js           # C++ code parser
├── game.js            # Game engine and logic
├── renderer.js        # Canvas rendering
├── levels.js          # Level definitions
├── main.js            # Application controller
└── README.md          # This file
```

### Technologies Used

- **HTML5 Canvas** - Graphics rendering
- **CodeMirror** - Code editor with syntax highlighting
- **Vanilla JavaScript** - No frameworks, pure JS
- **CSS3** - Modern styling with gradients and animations

### Game Architecture

1. **Parser** (`parser.js`) - Parses simplified C++ into an Abstract Syntax Tree (AST)
2. **Game Engine** (`game.js`) - Manages game state, grid, drone, and resources
3. **Renderer** (`renderer.js`) - Draws the game grid, drone, and crops on canvas
4. **Levels** (`levels.js`) - Defines level objectives, setup, and completion criteria
5. **Main Controller** (`main.js`) - Coordinates everything together

## Learning Outcomes

By playing this game, you'll learn:

- C++ syntax fundamentals
- Control flow (if/else, while loops)
- Function calls and parameters
- Boolean conditions and comparisons
- Algorithm design and optimization
- Problem-solving and debugging
- Resource management

## Known Issues

- Parser supports simplified C++ only (no pointers, classes, etc.)
- Loop detection is basic - complex infinite loops may not be caught
- Mobile support could be improved

## Future Enhancements

- [ ] More levels with increasing difficulty
- [ ] Achievement system
- [ ] Code challenges and leaderboards
- [ ] Save/load programs
- [ ] Step-by-step debugging mode
- [ ] Multiplayer cooperative farming

- [ ] Custom level editor

