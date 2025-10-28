// Main application logic

let game;
let renderer;
let parser;
let editor;
let currentLevelIndex = 0;
let executionInterval;
let speedMultiplier = 1;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeGame();
    initializeEditor();
    initializeControls();
    initializeModal();
    loadLevel(0);
});

function initializeGame() {
    const canvas = document.getElementById('gameCanvas');
    game = new Game(10);
    renderer = new Renderer(canvas, game);
    parser = new CppParser();
    
    renderer.render();
}

function initializeEditor() {
    const textarea = document.getElementById('codeEditor');
    editor = CodeMirror.fromTextArea(textarea, {
        mode: 'text/x-c++src',
        theme: 'monokai',
        lineNumbers: true,
        indentUnit: 4,
        tabSize: 4,
        indentWithTabs: false,
        lineWrapping: true,
        matchBrackets: true,
        autoCloseBrackets: true
    });
    
    editor.setSize(null, '400px');
}

function initializeControls() {
    document.getElementById('runBtn').addEventListener('click', runProgram);
    document.getElementById('stopBtn').addEventListener('click', stopProgram);
    document.getElementById('resetBtn').addEventListener('click', resetLevel);
    document.getElementById('speedBtn').addEventListener('click', toggleSpeed);
}

function initializeModal() {
    const modal = document.getElementById('apiModal');
    const btn = document.getElementById('showApiBtn');
    const span = document.getElementsByClassName('close')[0];

    btn.onclick = () => {
        modal.style.display = 'block';
    };

    span.onclick = () => {
        modal.style.display = 'none';
    };

    window.onclick = (event) => {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    };
}

function loadLevel(levelIndex) {
    if (levelIndex < 0 || levelIndex >= levels.length) {
        return;
    }

    currentLevelIndex = levelIndex;
    const level = levels[levelIndex];
    
    // Update UI
    document.getElementById('levelTitle').textContent = level.title;
    document.getElementById('levelObjective').textContent = level.objective;
    
    // Load level into game
    game.loadLevel(level);
    
    // Set starter code
    if (level.starterCode) {
        editor.setValue(level.starterCode);
    }
    
    // Render
    renderer.render();
    updateStats();
    
    logToConsole(`Loaded ${level.title}`, 'success');
    logToConsole(`Objective: ${level.objective}`);
}

async function runProgram() {
    const code = editor.getValue();
    
    // Clear console
    clearConsole();
    logToConsole('Parsing program...');
    
    // Parse the code
    const parseResult = parser.parse(code);
    
    if (!parseResult.success) {
        parseResult.errors.forEach(error => {
            logToConsole(error, 'error');
        });
        return;
    }
    
    logToConsole('Program parsed successfully!', 'success');
    logToConsole('Starting execution...');
    
    // Disable run button, enable stop button
    document.getElementById('runBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    
    game.running = true;
    
    try {
        // Run the main loop
        while (game.running && game.cycleCount < game.maxCycles) {
            game.cycleCount++;
            
            // Execute one iteration of the loop
            await game.executeAST(parseResult.instructions, 100);
            
            // Update crops
            if (game.cycleCount % 5 === 0) {
                game.updateCropGrowth();
            }
            
            // Render
            renderer.render();
            updateStats();
            
            // Check for completion
            if (game.checkLevelComplete()) {
                logToConsole('🎉 Level Complete! Well done!', 'success');
                game.running = false;
                
                // Show completion message
                setTimeout(() => {
                    if (confirm('Level completed! Move to next level?')) {
                        loadLevel(currentLevelIndex + 1);
                    }
                }, 500);
                break;
            }
            
            // Check if out of energy
            if (game.resources.energy <= 0) {
                logToConsole('⚠️ Out of energy! Stopping execution.', 'warning');
                game.running = false;
                break;
            }
        }
        
        if (game.cycleCount >= game.maxCycles) {
            logToConsole('⚠️ Maximum cycles reached. Stopping execution.', 'warning');
        }
        
        if (game.running) {
            logToConsole('Program execution stopped.', 'warning');
        }
        
    } catch (error) {
        logToConsole(`❌ Runtime Error: ${error.message}`, 'error');
        game.running = false;
    }
    
    // Re-enable buttons
    document.getElementById('runBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
}

function stopProgram() {
    game.running = false;
    logToConsole('Program stopped by user.', 'warning');
    document.getElementById('runBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
}

function resetLevel() {
    stopProgram();
    game.reset();
    renderer.render();
    updateStats();
    clearConsole();
    logToConsole('Level reset.', 'success');
}

function toggleSpeed() {
    const speeds = [1, 2, 4, 8];
    const currentIndex = speeds.indexOf(speedMultiplier);
    const nextIndex = (currentIndex + 1) % speeds.length;
    speedMultiplier = speeds[nextIndex];
    game.speed = speedMultiplier;
    
    document.getElementById('speedBtn').textContent = `⏩ Speed: ${speedMultiplier}x`;
    logToConsole(`Speed set to ${speedMultiplier}x`);
}

function updateStats() {
    const stats = game.getStats();
    document.getElementById('seedCount').textContent = stats.seeds;
    document.getElementById('harvestCount').textContent = stats.harvested;
    document.getElementById('energyCount').textContent = stats.energy;
    document.getElementById('cycleCount').textContent = stats.cycles;
}

function logToConsole(message, type = 'normal') {
    const consoleOutput = document.getElementById('consoleOutput');
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    line.textContent = `[${timestamp}] ${message}`;
    
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function clearConsole() {
    const consoleOutput = document.getElementById('consoleOutput');
    consoleOutput.innerHTML = '';
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to run
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!game.running) {
            runProgram();
        }
    }
    
    // Ctrl/Cmd + R to reset
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        resetLevel();
    }
});
