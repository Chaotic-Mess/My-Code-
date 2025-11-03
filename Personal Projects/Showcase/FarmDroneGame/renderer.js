// Renderer for Farm Drone Automation

class Renderer {
    constructor(canvas, game) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.game = game;
        this.cellSize = 0;
        this.updateCellSize();

        // Colors
        this.colors = {
            empty: '#d4e4d4',
            planted: '#8b7355',
            growing: '#9fc15f',
            grown: '#f4d03f',
            obstacle: '#5a5a5a',
            grid: '#999',
            drone: '#3498db',
            droneHighlight: '#2980b9'
        };

        // Emoji assets
        this.emojis = {
            seed: '🌱',
            growing: '🌿',
            grown: '🌾',
            obstacle: '🪨',
            drone: '🤖'
        };
    }

    updateCellSize() {
        this.cellSize = this.canvas.width / this.game.gridSize;
    }

    clear() {
        this.ctx.fillStyle = '#f0f4f0';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    drawGrid() {
        this.ctx.strokeStyle = this.colors.grid;
        this.ctx.lineWidth = 1;

        // Draw vertical lines
        for (let x = 0; x <= this.game.gridSize; x++) {
            const xPos = x * this.cellSize;
            this.ctx.beginPath();
            this.ctx.moveTo(xPos, 0);
            this.ctx.lineTo(xPos, this.canvas.height);
            this.ctx.stroke();
        }

        // Draw horizontal lines
        for (let y = 0; y <= this.game.gridSize; y++) {
            const yPos = y * this.cellSize;
            this.ctx.beginPath();
            this.ctx.moveTo(0, yPos);
            this.ctx.lineTo(this.canvas.width, yPos);
            this.ctx.stroke();
        }
    }

    drawCell(x, y, state) {
        const xPos = x * this.cellSize;
        const yPos = y * this.cellSize;

        // Draw cell background
        switch (state) {
            case 'EMPTY':
                this.ctx.fillStyle = this.colors.empty;
                break;
            case 'PLANTED':
                this.ctx.fillStyle = this.colors.planted;
                break;
            case 'GROWING':
                this.ctx.fillStyle = this.colors.growing;
                break;
            case 'GROWN':
                this.ctx.fillStyle = this.colors.grown;
                break;
            case 'OBSTACLE':
                this.ctx.fillStyle = this.colors.obstacle;
                break;
        }

        this.ctx.fillRect(xPos + 1, yPos + 1, this.cellSize - 2, this.cellSize - 2);

        // Draw emoji
        const centerX = xPos + this.cellSize / 2;
        const centerY = yPos + this.cellSize / 2;
        const fontSize = this.cellSize * 0.6;

        this.ctx.font = `${fontSize}px Arial`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        switch (state) {
            case 'PLANTED':
                this.ctx.fillText(this.emojis.seed, centerX, centerY);
                break;
            case 'GROWING':
                this.ctx.fillText(this.emojis.growing, centerX, centerY);
                break;
            case 'GROWN':
                this.ctx.fillText(this.emojis.grown, centerX, centerY);
                break;
            case 'OBSTACLE':
                this.ctx.fillText(this.emojis.obstacle, centerX, centerY);
                break;
        }
    }

    drawDrone() {
        const x = this.game.drone.x;
        const y = this.game.drone.y;
        const xPos = x * this.cellSize;
        const yPos = y * this.cellSize;
        const centerX = xPos + this.cellSize / 2;
        const centerY = yPos + this.cellSize / 2;

        // Draw highlight circle
        this.ctx.fillStyle = 'rgba(52, 152, 219, 0.3)';
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, this.cellSize * 0.45, 0, Math.PI * 2);
        this.ctx.fill();

        // Draw drone emoji
        const fontSize = this.cellSize * 0.7;
        this.ctx.font = `${fontSize}px Arial`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        // Rotate based on direction
        this.ctx.save();
        this.ctx.translate(centerX, centerY);
        this.ctx.rotate((this.game.drone.direction * Math.PI) / 2);
        this.ctx.fillText(this.emojis.drone, 0, 0);
        this.ctx.restore();

        // Draw direction indicator
        this.drawDirectionIndicator(centerX, centerY);
    }

    drawDirectionIndicator(centerX, centerY) {
        const directions = [
            { dx: 0, dy: -1 }, // North
            { dx: 1, dy: 0 },  // East
            { dx: 0, dy: 1 },  // South
            { dx: -1, dy: 0 }  // West
        ];

        const dir = directions[this.game.drone.direction];
        const arrowLength = this.cellSize * 0.3;
        const endX = centerX + dir.dx * arrowLength;
        const endY = centerY + dir.dy * arrowLength;

        this.ctx.strokeStyle = '#e74c3c';
        this.ctx.lineWidth = 3;
        this.ctx.lineCap = 'round';

        // Draw arrow line
        this.ctx.beginPath();
        this.ctx.moveTo(centerX, centerY);
        this.ctx.lineTo(endX, endY);
        this.ctx.stroke();

        // Draw arrowhead
        const arrowHeadSize = 8;
        const angle = Math.atan2(dir.dy, dir.dx);

        this.ctx.beginPath();
        this.ctx.moveTo(endX, endY);
        this.ctx.lineTo(
            endX - arrowHeadSize * Math.cos(angle - Math.PI / 6),
            endY - arrowHeadSize * Math.sin(angle - Math.PI / 6)
        );
        this.ctx.moveTo(endX, endY);
        this.ctx.lineTo(
            endX - arrowHeadSize * Math.cos(angle + Math.PI / 6),
            endY - arrowHeadSize * Math.sin(angle + Math.PI / 6)
        );
        this.ctx.stroke();
    }

    render() {
        this.updateCellSize();
        this.clear();
        
        // Draw all cells
        for (let y = 0; y < this.game.gridSize; y++) {
            for (let x = 0; x < this.game.gridSize; x++) {
                const cell = this.game.grid[y][x];
                this.drawCell(x, y, cell.state);
            }
        }

        this.drawGrid();
        this.drawDrone();
    }

    // Animation helper
    async animateAction(action) {
        // You can add specific animations for different actions here
        this.render();
    }
}
