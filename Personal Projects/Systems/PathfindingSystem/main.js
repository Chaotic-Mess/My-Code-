        const ROWS = 20;
        const COLS = 40;
        const BASE_STEP_MS = 25; // 1× playback ≈ old 25ms per step

        const grid = document.getElementById('grid');
        grid.style.gridTemplateColumns = `repeat(${COLS}, 25px)`;
        
        let nodes = [];
        let startNode = null;
        let endNode = null;
        let isMouseDown = false;
        let isRunning = false;
        
        // Descriptions for each algorithm (chatty + clear)
        const ALGO_DESCRIPTIONS = {
            astar: `<strong>A* Search:</strong> the overachiever with a compass. It mixes what we’ve explored with a guess of how far is left (a heuristic), so it usually beelines to the goal while still guaranteeing the shortest path when the heuristic doesn’t overestimate.`,
            dijkstra: `<strong>Dijkstra:</strong> the patient optimizer. It expands outward in all directions with perfectly fair costs, guaranteeing the shortest path—but without any “directional” hint it can explore more nodes than A*.`,
            bfs: `<strong>Breadth-First Search (BFS):</strong> the crowd favorite for uniform costs. It ripples out layer by layer and finds the shortest path when every move costs the same. Simple, reliable, and great for grids like this.`,
            dfs: `<strong>Depth-First Search (DFS):</strong> the spelunker. It dives deep down one corridor before backtracking. Fast to run and fun to watch, but it doesn’t guarantee the shortest path and can wander a lot.`
        };

        // Helpers for timescale-driven animation
        function getTimeScale() {
            return parseFloat(document.getElementById('timescale').value);
        }
        function getMsPerStep() {
            const ms = BASE_STEP_MS / Math.max(getTimeScale(), 0.0001);
            return Math.max(0, ms);
        }

        // Initialize grid
        function initGrid() {
            nodes = [];
            grid.innerHTML = '';
            
            for (let row = 0; row < ROWS; row++) {
                nodes[row] = [];
                for (let col = 0; col < COLS; col++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    cell.dataset.row = row;
                    cell.dataset.col = col;
                    
                    cell.addEventListener('mousedown', () => handleMouseDown(row, col));
                    cell.addEventListener('mouseenter', () => handleMouseEnter(row, col));
                    cell.addEventListener('mouseup', handleMouseUp);
                    
                    grid.appendChild(cell);
                    
                    nodes[row][col] = {
                        row,
                        col,
                        isWall: false,
                        isStart: false,
                        isEnd: false,
                        isVisited: false,
                        distance: Infinity,
                        heuristic: 0,
                        parent: null,
                        element: cell
                    };
                }
            }
            
            // Set default start and end
            startNode = nodes[10][5];
            endNode = nodes[10][35];
            startNode.isStart = true;
            endNode.isEnd = true;
            startNode.element.classList.add('start');
            endNode.element.classList.add('end');
        }
        
        function handleMouseDown(row, col) {
            if (isRunning) return;
            isMouseDown = true;
            toggleNode(row, col);
        }
        
        function handleMouseEnter(row, col) {
            if (!isMouseDown || isRunning) return;
            toggleWall(row, col);
        }
        
        function handleMouseUp() {
            isMouseDown = false;
        }
        
        function toggleNode(row, col) {
            const node = nodes[row][col];
            
            if (!startNode) {
                startNode = node;
                node.isStart = true;
                node.element.classList.add('start');
            } else if (!endNode && node !== startNode) {
                endNode = node;
                node.isEnd = true;
                node.element.classList.add('end');
            } else if (node === startNode) {
                node.isStart = false;
                node.element.classList.remove('start');
                startNode = null;
            } else if (node === endNode) {
                node.isEnd = false;
                node.element.classList.remove('end');
                endNode = null;
            } else {
                toggleWall(row, col);
            }
        }
        
        function toggleWall(row, col) {
            const node = nodes[row][col];
            if (node.isStart || node.isEnd) return;
            
            node.isWall = !node.isWall;
            node.element.classList.toggle('wall');
        }
        
        function clearPath() {
            for (let row = 0; row < ROWS; row++) {
                for (let col = 0; col < COLS; col++) {
                    const node = nodes[row][col];
                    node.isVisited = false;
                    node.distance = Infinity;
                    node.parent = null;
                    node.element.classList.remove('visited', 'path');
                }
            }
            updateStats(0, 0, 0, 0);
        }
        
        function clearAll() {
            startNode = null;
            endNode = null;
            initGrid();
            updateStats(0, 0, 0, 0);
        }
        
        function generateMaze() {
            clearAll();
            
            for (let row = 0; row < ROWS; row++) {
                for (let col = 0; col < COLS; col++) {
                    const node = nodes[row][col];
                    if (node.isStart || node.isEnd) continue;
                    
                    if (Math.random() < 0.3) {
                        node.isWall = true;
                        node.element.classList.add('wall');
                    }
                }
            }
        }
        
        function getNeighbors(node) {
            const neighbors = [];
            const { row, col } = node;
            
            if (row > 0) neighbors.push(nodes[row - 1][col]);
            if (row < ROWS - 1) neighbors.push(nodes[row + 1][col]);
            if (col > 0) neighbors.push(nodes[row][col - 1]);
            if (col < COLS - 1) neighbors.push(nodes[row][col + 1]);
            
            return neighbors.filter(n => !n.isWall);
        }
        
        function heuristic(node1, node2) {
            return Math.abs(node1.row - node2.row) + Math.abs(node1.col - node2.col);
        }

        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        function countAnimationSteps(visitedNodes, path) {
            const visitedSteps = visitedNodes.filter(n => !n.isStart && !n.isEnd).length;
            const pathSteps = path.filter(n => !n.isStart && !n.isEnd).length;
            return { visitedSteps, pathSteps, total: visitedSteps + pathSteps };
        }

        function updateStats(visited, pathLength, algoMs, animMs) {
            document.getElementById('nodesVisited').textContent = visited;
            document.getElementById('pathLength').textContent = pathLength;
            document.getElementById('algoTime').textContent = Math.round(algoMs) + 'ms';
            document.getElementById('animationTime').textContent = Math.round(animMs) + 'ms';
        }

        // Dynamic algorithm description
        function updateAlgoBlurb() {
            const key = document.getElementById('algorithm').value;
            document.getElementById('algoBlurb').innerHTML = ALGO_DESCRIPTIONS[key];
        }
        
        async function visualize() {
            if (!startNode || !endNode || isRunning) return;
            
            isRunning = true;
            clearPath();

            const algorithm = document.getElementById('algorithm').value;
            const msPerStep = getMsPerStep();

            const animStart = performance.now();

            let visitedNodes = [];
            let path = [];
            
            switch (algorithm) {
                case 'astar':
                    ({ visitedNodes, path } = await aStar(msPerStep));
                    break;
                case 'dijkstra':
                    ({ visitedNodes, path } = await dijkstra(msPerStep));
                    break;
                case 'bfs':
                    ({ visitedNodes, path } = await bfs(msPerStep));
                    break;
                case 'dfs':
                    ({ visitedNodes, path } = await dfs(msPerStep));
                    break;
            }

            const animEnd = performance.now();
            const animationTime = animEnd - animStart;

            // Estimate "real" algorithm time by subtracting intentional sleeps
            const { visitedSteps, pathSteps, total } = countAnimationSteps(visitedNodes, path);
            const sleepTime = total * msPerStep;
            const algoTime = Math.max(0, animationTime - sleepTime);

            updateStats(visitedNodes.length, path.length, algoTime, animationTime);
            isRunning = false;
        }
        
        async function aStar(msPerStep) {
            const openSet = [startNode];
            const visitedNodes = [];
            startNode.distance = 0;
            startNode.heuristic = heuristic(startNode, endNode);
            
            while (openSet.length > 0) {
                openSet.sort((a, b) => (a.distance + a.heuristic) - (b.distance + b.heuristic));
                const current = openSet.shift();
                
                if (current.isVisited) continue;
                current.isVisited = true;
                visitedNodes.push(current);
                
                if (!current.isStart && !current.isEnd) {
                    current.element.classList.add('visited');
                    await sleep(msPerStep);
                }
                
                if (current === endNode) {
                    const path = await reconstructPath(endNode, msPerStep);
                    return { visitedNodes, path };
                }
                
                const neighbors = getNeighbors(current);
                for (const neighbor of neighbors) {
                    const newDistance = current.distance + 1;
                    
                    if (newDistance < neighbor.distance) {
                        neighbor.distance = newDistance;
                        neighbor.heuristic = heuristic(neighbor, endNode);
                        neighbor.parent = current;
                        openSet.push(neighbor);
                    }
                }
            }
            
            return { visitedNodes, path: [] };
        }
        
        async function dijkstra(msPerStep) {
            const unvisited = [];
            startNode.distance = 0;
            
            for (let row = 0; row < ROWS; row++) {
                for (let col = 0; col < COLS; col++) {
                    unvisited.push(nodes[row][col]);
                }
            }
            
            const visitedNodes = [];
            
            while (unvisited.length > 0) {
                unvisited.sort((a, b) => a.distance - b.distance);
                const current = unvisited.shift();
                
                if (current.distance === Infinity) break;
                
                current.isVisited = true;
                visitedNodes.push(current);
                
                if (!current.isStart && !current.isEnd) {
                    current.element.classList.add('visited');
                    await sleep(msPerStep);
                }
                
                if (current === endNode) {
                    const path = await reconstructPath(endNode, msPerStep);
                    return { visitedNodes, path };
                }
                
                const neighbors = getNeighbors(current);
                for (const neighbor of neighbors) {
                    const newDistance = current.distance + 1;
                    
                    if (newDistance < neighbor.distance) {
                        neighbor.distance = newDistance;
                        neighbor.parent = current;
                    }
                }
            }
            
            return { visitedNodes, path: [] };
        }
        
        async function bfs(msPerStep) {
            const queue = [startNode];
            const visitedNodes = [];
            startNode.isVisited = true;
            
            while (queue.length > 0) {
                const current = queue.shift();
                visitedNodes.push(current);
                
                if (!current.isStart && !current.isEnd) {
                    current.element.classList.add('visited');
                    await sleep(msPerStep);
                }
                
                if (current === endNode) {
                    const path = await reconstructPath(endNode, msPerStep);
                    return { visitedNodes, path };
                }
                
                const neighbors = getNeighbors(current);
                for (const neighbor of neighbors) {
                    if (!neighbor.isVisited) {
                        neighbor.isVisited = true;
                        neighbor.parent = current;
                        queue.push(neighbor);
                    }
                }
            }
            
            return { visitedNodes, path: [] };
        }
        
        async function dfs(msPerStep) {
            const stack = [startNode];
            const visitedNodes = [];
            
            while (stack.length > 0) {
                const current = stack.pop();
                
                if (current.isVisited) continue;
                
                current.isVisited = true;
                visitedNodes.push(current);
                
                if (!current.isStart && !current.isEnd) {
                    current.element.classList.add('visited');
                    await sleep(msPerStep);
                }
                
                if (current === endNode) {
                    const path = await reconstructPath(endNode, msPerStep);
                    return { visitedNodes, path };
                }
                
                const neighbors = getNeighbors(current);
                for (const neighbor of neighbors) {
                    if (!neighbor.isVisited) {
                        neighbor.parent = current;
                        stack.push(neighbor);
                    }
                }
            }
            
            return { visitedNodes, path: [] };
        }
        
        async function reconstructPath(endNode, msPerStep) {
            const path = [];
            let current = endNode;
            
            while (current !== null) {
                path.unshift(current);
                current = current.parent;
            }
            
            for (const node of path) {
                if (!node.isStart && !node.isEnd) {
                    node.element.classList.add('path');
                    await sleep(msPerStep);
                }
            }
            return path;
        }
        
        // Event listeners
        document.getElementById('visualize').addEventListener('click', visualize);
        document.getElementById('generateMaze').addEventListener('click', generateMaze);
        document.getElementById('clearPath').addEventListener('click', clearPath);
        document.getElementById('clearAll').addEventListener('click', clearAll);
        
        document.getElementById('timescale').addEventListener('input', (e) => {
            const scale = parseFloat(e.target.value);
            const approxMs = Math.round(BASE_STEP_MS / Math.max(scale, 0.0001));
            document.getElementById('speedValue').textContent = `${scale.toFixed(1)}× (~${approxMs}ms/step)`;
        });

        document.getElementById('algorithm').addEventListener('change', updateAlgoBlurb);
        document.addEventListener('mouseup', handleMouseUp);
        
        // Initialize
        initGrid();
        updateAlgoBlurb();
