// Game Engine for Farm Drone Automation

class Game {
    constructor(gridSize = 10) {
        this.gridSize = gridSize;
        this.grid = [];
        this.drone = {
            x: 0,
            y: 0,
            direction: 0 // 0=North, 1=East, 2=South, 3=West
        };
        this.resources = {
            seeds: 10,
            harvested: 0,
            energy: 100
        };
        this.cycleCount = 0;
        this.running = false;
        this.speed = 1; // Execution speed multiplier
        this.maxCycles = 1000; // Prevent infinite loops
        this.currentLevel = null;

        // Cell state constants
        this.CELL_STATES = {
            EMPTY: 'EMPTY',
            PLANTED: 'PLANTED',
            GROWING: 'GROWING',
            GROWN: 'GROWN',
            OBSTACLE: 'OBSTACLE'
        };

        this.initGrid();
    }

    initGrid() {
        this.grid = [];
        for (let y = 0; y < this.gridSize; y++) {
            this.grid[y] = [];
            for (let x = 0; x < this.gridSize; x++) {
                this.grid[y][x] = {
                    state: this.CELL_STATES.EMPTY,
                    growthTimer: 0,
                    watered: false
                };
            }
        }
    }

    reset() {
        this.initGrid();
        this.drone = {
            x: Math.floor(this.gridSize / 2),
            y: Math.floor(this.gridSize / 2),
            direction: 0
        };
        this.resources = {
            seeds: 10,
            harvested: 0,
            energy: 100
        };
        this.cycleCount = 0;
        this.running = false;

        // Apply level-specific setup
        if (this.currentLevel && this.currentLevel.setup) {
            this.currentLevel.setup(this);
        }
    }

    loadLevel(level) {
        this.currentLevel = level;
        this.gridSize = level.gridSize || 10;
        this.reset();
    }

    // Drone movement functions
    moveForward() {
        if (this.resources.energy <= 0) {
            throw new Error("Out of energy!");
        }

        const directions = [
            { dx: 0, dy: -1 }, // North
            { dx: 1, dy: 0 },  // East
            { dx: 0, dy: 1 },  // South
            { dx: -1, dy: 0 }  // West
        ];

        const dir = directions[this.drone.direction];
        const newX = this.drone.x + dir.dx;
        const newY = this.drone.y + dir.dy;

        // Check bounds
        if (newX >= 0 && newX < this.gridSize && newY >= 0 && newY < this.gridSize) {
            const cell = this.grid[newY][newX];
            if (cell.state !== this.CELL_STATES.OBSTACLE) {
                this.drone.x = newX;
                this.drone.y = newY;
                this.resources.energy -= 1;
                return true;
            } else {
                throw new Error("Cannot move into obstacle!");
            }
        } else {
            throw new Error("Cannot move outside the farm boundary!");
        }
    }

    turnRight() {
        this.drone.direction = (this.drone.direction + 1) % 4;
        this.resources.energy -= 0.5;
    }

    turnLeft() {
        this.drone.direction = (this.drone.direction + 3) % 4;
        this.resources.energy -= 0.5;
    }

    // Sensing functions
    scan() {
        const cell = this.grid[this.drone.y][this.drone.x];
        return cell.state;
    }

    scanAhead() {
        const directions = [
            { dx: 0, dy: -1 },
            { dx: 1, dy: 0 },
            { dx: 0, dy: 1 },
            { dx: -1, dy: 0 }
        ];

        const dir = directions[this.drone.direction];
        const newX = this.drone.x + dir.dx;
        const newY = this.drone.y + dir.dy;

        if (newX >= 0 && newX < this.gridSize && newY >= 0 && newY < this.gridSize) {
            return this.grid[newY][newX].state;
        }
        return this.CELL_STATES.OBSTACLE;
    }

    // Action functions
    plant() {
        const cell = this.grid[this.drone.y][this.drone.x];
        
        if (this.resources.seeds <= 0) {
            throw new Error("Out of seeds!");
        }
        
        if (cell.state === this.CELL_STATES.EMPTY) {
            cell.state = this.CELL_STATES.PLANTED;
            cell.growthTimer = 0;
            this.resources.seeds--;
            this.resources.energy -= 2;
            return true;
        } else {
            throw new Error("Cannot plant on non-empty cell!");
        }
    }

    harvest() {
        const cell = this.grid[this.drone.y][this.drone.x];
        
        if (cell.state === this.CELL_STATES.GROWN) {
            cell.state = this.CELL_STATES.EMPTY;
            cell.growthTimer = 0;
            cell.watered = false;
            this.resources.harvested++;
            this.resources.seeds += 2; // Get seeds back plus bonus
            this.resources.energy -= 2;
            return true;
        } else {
            throw new Error("Cannot harvest non-grown crops!");
        }
    }

    water() {
        const cell = this.grid[this.drone.y][this.drone.x];
        
        if (cell.state === this.CELL_STATES.PLANTED || cell.state === this.CELL_STATES.GROWING) {
            cell.watered = true;
            this.resources.energy -= 1;
            return true;
        } else {
            throw new Error("Can only water planted or growing crops!");
        }
    }

    // Update crop growth
    updateCropGrowth() {
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                const cell = this.grid[y][x];
                
                if (cell.state === this.CELL_STATES.PLANTED) {
                    cell.growthTimer++;
                    const growthSpeed = cell.watered ? 2 : 3;
                    if (cell.growthTimer >= growthSpeed) {
                        cell.state = this.CELL_STATES.GROWING;
                        cell.growthTimer = 0;
                    }
                } else if (cell.state === this.CELL_STATES.GROWING) {
                    cell.growthTimer++;
                    const growthSpeed = cell.watered ? 2 : 3;
                    if (cell.growthTimer >= growthSpeed) {
                        cell.state = this.CELL_STATES.GROWN;
                        cell.growthTimer = 0;
                    }
                }
            }
        }
    }

    // Execute parsed AST
    async executeAST(ast, maxIterations = 100) {
        let iterations = 0;

        const executeBlock = async (block) => {
            for (const instruction of block) {
                if (!this.running) return;
                
                iterations++;
                if (iterations > maxIterations) {
                    throw new Error("Maximum iterations exceeded! Check for infinite loops.");
                }

                try {
                    if (instruction.type === 'call') {
                        await this.executeFunction(instruction.function);
                    } else if (instruction.type === 'if') {
                        const conditionResult = this.evaluateCondition(instruction.condition);
                        if (conditionResult) {
                            await executeBlock(instruction.ifBlock);
                        } else if (instruction.elseBlock) {
                            await executeBlock(instruction.elseBlock);
                        }
                    } else if (instruction.type === 'while') {
                        let whileIterations = 0;
                        while (this.evaluateCondition(instruction.condition) && this.running) {
                            whileIterations++;
                            if (whileIterations > 50) {
                                throw new Error("While loop exceeded 50 iterations!");
                            }
                            await executeBlock(instruction.block);
                        }
                    }
                } catch (error) {
                    throw error;
                }
            }
        };

        await executeBlock(ast);
    }

    async executeFunction(functionName) {
        const delay = 500 / this.speed;
        
        switch (functionName) {
            case 'moveForward':
                this.moveForward();
                break;
            case 'turnRight':
                this.turnRight();
                break;
            case 'turnLeft':
                this.turnLeft();
                break;
            case 'plant':
                this.plant();
                break;
            case 'harvest':
                this.harvest();
                break;
            case 'water':
                this.water();
                break;
            default:
                throw new Error(`Unknown function: ${functionName}`);
        }

        // Wait for animation
        await new Promise(resolve => setTimeout(resolve, delay));
    }

    evaluateCondition(condition) {
        let leftValue, rightValue;

        // Evaluate left side
        if (condition.left.type === 'call') {
            if (condition.left.function === 'scan') {
                leftValue = this.scan();
            } else if (condition.left.function === 'scanAhead') {
                leftValue = this.scanAhead();
            }
        }

        // Evaluate right side
        if (condition.right.type === 'constant') {
            rightValue = condition.right.value;
        } else if (condition.right.type === 'number') {
            rightValue = condition.right.value;
        }

        // Apply operator
        switch (condition.operator) {
            case '==':
                return leftValue === rightValue;
            case '!=':
                return leftValue !== rightValue;
            case '<':
                return leftValue < rightValue;
            case '>':
                return leftValue > rightValue;
            case '<=':
                return leftValue <= rightValue;
            case '>=':
                return leftValue >= rightValue;
            default:
                return false;
        }
    }

    // Check level completion
    checkLevelComplete() {
        if (this.currentLevel && this.currentLevel.checkComplete) {
            return this.currentLevel.checkComplete(this);
        }
        return false;
    }

    getStats() {
        return {
            seeds: this.resources.seeds,
            harvested: this.resources.harvested,
            energy: Math.max(0, Math.floor(this.resources.energy)),
            cycles: this.cycleCount
        };
    }
}
