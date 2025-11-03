        let array = [];
        let sorting = false;
        let comparisons = 0;
        let accesses = 0;
        let startTime = 0;
        let animationSpeed = 50;

        const arrayContainer = document.getElementById('arrayContainer');
        const algorithmSelect = document.getElementById('algorithm');
        const arraySizeInput = document.getElementById('arraySize');
        const speedInput = document.getElementById('speed');
        const generateBtn = document.getElementById('generate');
        const sortBtn = document.getElementById('sort');
        const sizeValueSpan = document.getElementById('sizeValue');
        const speedValueSpan = document.getElementById('speedValue');
        const comparisonsSpan = document.getElementById('comparisons');
        const accessesSpan = document.getElementById('accesses');
        const timeSpan = document.getElementById('time');
        const algorithmInfo = document.getElementById('algorithmInfo');

        const algorithmDescriptions = {
            bubble: "Bubble Sort: Repeatedly steps through the list, compares adjacent elements and swaps them if they're in the wrong order. Time Complexity: O(n²)",
            selection: "Selection Sort: Divides the array into sorted and unsorted regions, repeatedly selecting the minimum element from unsorted region. Time Complexity: O(n²)",
            insertion: "Insertion Sort: Builds the final sorted array one item at a time, inserting each element into its proper position. Time Complexity: O(n²)",
            quick: "Quick Sort: Divides array into partitions around a pivot element, recursively sorting each partition. Time Complexity: O(n log n) average",
            merge: "Merge Sort: Divides array into halves, recursively sorts them, and merges the sorted halves. Time Complexity: O(n log n)"
        };

        arraySizeInput.addEventListener('input', (e) => {
            sizeValueSpan.textContent = e.target.value;
        });

        speedInput.addEventListener('input', (e) => {
            animationSpeed = 201 - e.target.value;
            speedValueSpan.textContent = animationSpeed;
        });

        algorithmSelect.addEventListener('change', (e) => {
            algorithmInfo.textContent = algorithmDescriptions[e.target.value];
        });

        generateBtn.addEventListener('click', generateArray);
        sortBtn.addEventListener('click', startSorting);

        function generateArray() {
            if (sorting) return;
            
            const size = parseInt(arraySizeInput.value);
            array = [];
            for (let i = 0; i < size; i++) {
                array.push(Math.floor(Math.random() * 350) + 20);
            }
            
            resetStats();
            renderArray();
        }

        function renderArray(comparing = [], swapping = [], sorted = []) {
            arrayContainer.innerHTML = '';
            const barWidth = Math.max(2, (arrayContainer.offsetWidth - array.length * 2) / array.length);
            
            array.forEach((value, idx) => {
                const bar = document.createElement('div');
                bar.className = 'array-bar';
                bar.style.height = `${value}px`;
                bar.style.width = `${barWidth}px`;
                
                if (sorted.includes(idx)) {
                    bar.classList.add('sorted');
                } else if (comparing.includes(idx)) {
                    bar.classList.add('comparing');
                } else if (swapping.includes(idx)) {
                    bar.classList.add('swapping');
                }
                
                arrayContainer.appendChild(bar);
            });
        }

        function resetStats() {
            comparisons = 0;
            accesses = 0;
            comparisonsSpan.textContent = '0';
            accessesSpan.textContent = '0';
            timeSpan.textContent = '0s';
        }

        function updateStats() {
            comparisonsSpan.textContent = comparisons;
            accessesSpan.textContent = accesses;
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            timeSpan.textContent = `${elapsed}s`;
        }

        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        async function startSorting() {
            if (sorting) return;
            
            sorting = true;
            sortBtn.disabled = true;
            generateBtn.disabled = true;
            algorithmSelect.disabled = true;
            arraySizeInput.disabled = true;
            
            resetStats();
            startTime = Date.now();
            
            const algorithm = algorithmSelect.value;
            
            switch(algorithm) {
                case 'bubble':
                    await bubbleSort();
                    break;
                case 'selection':
                    await selectionSort();
                    break;
                case 'insertion':
                    await insertionSort();
                    break;
                case 'quick':
                    await quickSort(0, array.length - 1);
                    break;
                case 'merge':
                    await mergeSort(0, array.length - 1);
                    break;
            }
            
            await finishAnimation();
            
            sorting = false;
            sortBtn.disabled = false;
            generateBtn.disabled = false;
            algorithmSelect.disabled = false;
            arraySizeInput.disabled = false;
        }

        async function bubbleSort() {
            for (let i = 0; i < array.length; i++) {
                for (let j = 0; j < array.length - i - 1; j++) {
                    comparisons++;
                    accesses += 2;
                    updateStats();
                    
                    renderArray([j, j + 1], [], []);
                    await sleep(animationSpeed);
                    
                    if (array[j] > array[j + 1]) {
                        [array[j], array[j + 1]] = [array[j + 1], array[j]];
                        accesses += 4;
                        renderArray([], [j, j + 1], []);
                        await sleep(animationSpeed);
                    }
                }
            }
        }

        async function selectionSort() {
            for (let i = 0; i < array.length; i++) {
                let minIdx = i;
                
                for (let j = i + 1; j < array.length; j++) {
                    comparisons++;
                    accesses += 2;
                    updateStats();
                    
                    renderArray([minIdx, j], [], Array.from({length: i}, (_, k) => k));
                    await sleep(animationSpeed);
                    
                    if (array[j] < array[minIdx]) {
                        minIdx = j;
                    }
                }
                
                if (minIdx !== i) {
                    [array[i], array[minIdx]] = [array[minIdx], array[i]];
                    accesses += 4;
                    renderArray([], [i, minIdx], Array.from({length: i}, (_, k) => k));
                    await sleep(animationSpeed);
                }
            }
        }

        async function insertionSort() {
            for (let i = 1; i < array.length; i++) {
                let key = array[i];
                let j = i - 1;
                accesses++;
                
                while (j >= 0 && array[j] > key) {
                    comparisons++;
                    accesses += 2;
                    updateStats();
                    
                    array[j + 1] = array[j];
                    accesses += 2;
                    
                    renderArray([j, j + 1], [], Array.from({length: i}, (_, k) => k));
                    await sleep(animationSpeed);
                    
                    j--;
                }
                
                array[j + 1] = key;
                accesses++;
            }
        }

        async function quickSort(low, high) {
            if (low < high) {
                const pi = await partition(low, high);
                await quickSort(low, pi - 1);
                await quickSort(pi + 1, high);
            }
        }

        async function partition(low, high) {
            const pivot = array[high];
            accesses++;
            let i = low - 1;
            
            for (let j = low; j < high; j++) {
                comparisons++;
                accesses += 2;
                updateStats();
                
                renderArray([j, high], [], []);
                await sleep(animationSpeed);
                
                if (array[j] < pivot) {
                    i++;
                    [array[i], array[j]] = [array[j], array[i]];
                    accesses += 4;
                    renderArray([], [i, j], []);
                    await sleep(animationSpeed);
                }
            }
            
            [array[i + 1], array[high]] = [array[high], array[i + 1]];
            accesses += 4;
            renderArray([], [i + 1, high], []);
            await sleep(animationSpeed);
            
            return i + 1;
        }

        async function mergeSort(left, right) {
            if (left < right) {
                const mid = Math.floor((left + right) / 2);
                await mergeSort(left, mid);
                await mergeSort(mid + 1, right);
                await merge(left, mid, right);
            }
        }

        async function merge(left, mid, right) {
            const leftArr = array.slice(left, mid + 1);
            const rightArr = array.slice(mid + 1, right + 1);
            accesses += (mid - left + 1) + (right - mid);
            
            let i = 0, j = 0, k = left;
            
            while (i < leftArr.length && j < rightArr.length) {
                comparisons++;
                accesses += 2;
                updateStats();
                
                renderArray([left + i, mid + 1 + j], [], []);
                await sleep(animationSpeed);
                
                if (leftArr[i] <= rightArr[j]) {
                    array[k] = leftArr[i];
                    accesses++;
                    i++;
                } else {
                    array[k] = rightArr[j];
                    accesses++;
                    j++;
                }
                k++;
                renderArray([], [k - 1], []);
                await sleep(animationSpeed);
            }
            
            while (i < leftArr.length) {
                array[k] = leftArr[i];
                accesses++;
                renderArray([], [k], []);
                await sleep(animationSpeed);
                i++;
                k++;
            }
            
            while (j < rightArr.length) {
                array[k] = rightArr[j];
                accesses++;
                renderArray([], [k], []);
                await sleep(animationSpeed);
                j++;
                k++;
            }
        }

        async function finishAnimation() {
            for (let i = 0; i < array.length; i++) {
                renderArray([], [], Array.from({length: i + 1}, (_, k) => k));
                await sleep(10);
            }
        }

        generateArray();
        algorithmInfo.textContent = algorithmDescriptions[algorithmSelect.value];
