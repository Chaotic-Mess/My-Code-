// ═══════════════════════════════════════════════════════════
// OPERATORS GENERATOR - 37 Ghosts
// ═══════════════════════════════════════════════════════════

const CODENAMES = [
    'PHANTOM', 'ECLIPSE', 'GHOST', 'SHADOW', 'VIPER',
    'RAVEN', 'COBRA', 'WRAITH', 'REAPER', 'SPECTRE',
    'CIPHER', 'NOMAD', 'RAZOR', 'VOID', 'NEXUS',
    'ONYX', 'SABLE', 'OBSIDIAN', 'SMOKE', 'ASH',
    'FROST', 'BLAZE', 'STORM', 'THUNDER', 'LIGHTNING',
    'TITAN', 'ATLAS', 'ZEUS', 'HADES', 'ARES',
    'APEX', 'OMEGA', 'ALPHA', 'DELTA', 'SIGMA',
    'KILO', 'ZULU'
];

const CLASSES = [
    'INFILTRATOR', 'SABOTEUR', 'EXTRACTOR', 'CLEANER',
    'GHOST PROTOCOL', 'DATA THIEF', 'PSYOP SPECIALIST',
    'URBAN PHANTOM', 'NARRATIVE ASSASSIN', 'SIGNAL HUNTER'
];

const SPECIALIZATIONS = [
    'SABOTAGE', 'EXTRACTION', 'DATA DESTABILIZATION',
    'DIGITAL NARRATIVE COLLAPSE', 'WETWORK', 'CYBER WARFARE',
    'SOCIAL ENGINEERING', 'COUNTER-INTELLIGENCE', 
    'ASSET LIQUIDATION', 'DEEP COVER', 'PSYCHOLOGICAL OPERATIONS',
    'SURVEILLANCE EVASION', 'IDENTITY FABRICATION', 'SIGNAL MANIPULATION'
];

const THREAT_LEVELS = ['EXTREME', 'CRITICAL', 'HIGH', 'SEVERE'];

function generateOperatorData(codename, index) {
    const numSpecs = Math.floor(Math.random() * 3) + 2;
    const specs = [];
    const tempSpecs = [...SPECIALIZATIONS];
    
    for (let i = 0; i < numSpecs; i++) {
        const randIndex = Math.floor(Math.random() * tempSpecs.length);
        specs.push(tempSpecs.splice(randIndex, 1)[0]);
    }
    
    return {
        codename,
        class: CLASSES[Math.floor(Math.random() * CLASSES.length)],
        threat: THREAT_LEVELS[Math.floor(Math.random() * THREAT_LEVELS.length)],
        completion: Math.floor(Math.random() * 20) + 80, // 80-100%
        stealth: Math.floor(Math.random() * 30) + 70, // 70-100%
        ethics: Math.floor(Math.random() * 60) + 20, // 20-80% (lower is more flexible)
        specializations: specs,
        status: Math.random() > 0.3 ? 'ACTIVE' : 'DORMANT',
        caseLogs: generateCaseLogs(3 + Math.floor(Math.random() * 3))
    };
}

function generateCaseLogs(count) {
    const templates = [
        'Asset neutralized in █████. No witnesses. Clean extraction.',
        'Data breach at ████████ Corp. 47TB exfiltrated. Trail cold.',
        'Target eliminated. Method: ████████. Client satisfied.',
        'Operation ████████ complete. Collateral: minimal.',
        'Deep cover maintained for ███ days. Identity intact.',
        'Signal intercepted from ████████. Contents classified.',
        'Psychological operation successful. Target behavior modified.',
        'Counter-surveillance detected. Evasion protocol executed.',
        'Social engineering exploit: ████████ credentials obtained.',
        'Infrastructure sabotaged. Attribution: impossible.'
    ];
    
    const logs = [];
    for (let i = 0; i < count; i++) {
        logs.push({
            id: `LOG-${Math.floor(Math.random() * 90000) + 10000}`,
            date: generateRandomDate(),
            text: templates[Math.floor(Math.random() * templates.length)],
            classified: Math.random() > 0.6
        });
    }
    return logs;
}

function generateRandomDate() {
    const year = 2024 + Math.floor(Math.random() * 2);
    const month = String(Math.floor(Math.random() * 12) + 1).padStart(2, '0');
    const day = String(Math.floor(Math.random() * 28) + 1).padStart(2, '0');
    return `${year}.${month}.${day}`;
}

export function generateOperators() {
    const grid = document.getElementById('operators-grid');
    const detailPanel = document.getElementById('operator-detail-panel');
    const operators = CODENAMES.map((name, index) => generateOperatorData(name, index));
    
    operators.forEach(operator => {
        const card = document.createElement('div');
        card.className = 'operator-card';
        card.innerHTML = `
            <div class="operator-card-content">
                <div class="operator-info">
                    <div class="operator-codename">${operator.codename}</div>
                    <div class="operator-class">${operator.class}</div>
                </div>
            </div>
            <div class="operator-threat">${operator.threat}</div>
        `;
        
        card.addEventListener('click', () => {
            // Remove selected from all cards
            document.querySelectorAll('.operator-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            
            // Show detail panel
            showOperatorDetail(operator, detailPanel);
        });
        
        // Hover sound
        card.addEventListener('mouseenter', () => {
            const hoverStatic = document.getElementById('hover-static');
            if (hoverStatic) {
                hoverStatic.currentTime = 0;
                hoverStatic.volume = 0.15;
                hoverStatic.play().catch(() => {});
            }
        });
        
        grid.appendChild(card);
    });
}

function showOperatorDetail(operator, detailPanel) {
    const selectBass = document.getElementById('select-bass');
    
    // Play bass drop
    if (selectBass) {
        selectBass.currentTime = 0;
        selectBass.volume = 0.4;
        selectBass.playbackRate = 0.8;
        selectBass.play().catch(() => {});
    }
    
    // Build detail HTML
    detailPanel.innerHTML = `
        <div class="dossier-header">
            <h2 class="operator-codename">${operator.codename}</h2>
            <span class="operator-status">${operator.status}</span>
        </div>
        <div class="dossier-stats">
            <div class="stat-item">
                <span class="stat-label">COMPLETION RATE</span>
                <div class="stat-bar">
                    <div class="stat-fill" style="width: 0%"></div>
                </div>
                <span class="stat-value">${operator.completion}%</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">STEALTH GRADE</span>
                <div class="stat-bar">
                    <div class="stat-fill" style="width: 0%"></div>
                </div>
                <span class="stat-value">${operator.stealth}%</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">ETHICAL FLEXIBILITY</span>
                <div class="stat-bar">
                    <div class="stat-fill" style="width: 0%"></div>
                </div>
                <span class="stat-value">${operator.ethics}%</span>
            </div>
        </div>
        <div class="dossier-specializations">
            <h3>SPECIALIZATIONS</h3>
            <div class="specialization-tags">
                ${operator.specializations.map(spec => `<span class="spec-tag">${spec}</span>`).join('')}
            </div>
        </div>
        <div class="dossier-logs">
            <h3>CASE LOGS</h3>
            <div class="case-logs-container">
                ${operator.caseLogs.map(log => `
                    <div class="case-log ${log.classified ? 'will-censor' : ''}">
                        <div class="case-log-header">
                            <span class="case-log-id">${log.id}</span>
                            <span class="case-log-date">${log.date}</span>
                        </div>
                        <div class="case-log-text">${log.text}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    // Animate stats
    setTimeout(() => {
        const statFills = detailPanel.querySelectorAll('.stat-fill');
        statFills[0].style.width = operator.completion + '%';
        statFills[1].style.width = operator.stealth + '%';
        statFills[2].style.width = operator.ethics + '%';
    }, 100);
    
    // Time-based censorship
    detailPanel.querySelectorAll('.case-log.will-censor').forEach(log => {
        setTimeout(() => {
            log.classList.add('censored');
        }, 8000 + Math.random() * 5000);
    });
    
    // Show panel
    detailPanel.classList.add('active');
}

function openOperatorDossier(operator) {
    const modal = document.getElementById('operator-modal');
    const selectBass = document.getElementById('select-bass');
    
    // Play deep bass
    if (selectBass) {
        selectBass.currentTime = 0;
        selectBass.volume = 0.5;
        selectBass.playbackRate = 0.7;
        selectBass.play().catch(() => {});
    }
    
    // Populate dossier
    modal.querySelector('.operator-codename').textContent = operator.codename;
    modal.querySelector('.operator-status').textContent = operator.status;
    
    // Stats with animation
    const statFills = modal.querySelectorAll('.stat-fill');
    const statValues = modal.querySelectorAll('.stat-value');
    
    statFills[0].style.width = '0%';
    statFills[1].style.width = '0%';
    statFills[2].style.width = '0%';
    
    setTimeout(() => {
        statFills[0].style.width = operator.completion + '%';
        statValues[0].textContent = operator.completion + '%';
        
        statFills[1].style.width = operator.stealth + '%';
        statValues[1].textContent = operator.stealth + '%';
        
        statFills[2].style.width = operator.ethics + '%';
        statValues[2].textContent = operator.ethics + '%';
    }, 100);
    
    // Specializations
    const specsContainer = modal.querySelector('.specialization-tags');
    specsContainer.innerHTML = '';
    operator.specializations.forEach(spec => {
        const tag = document.createElement('span');
        tag.className = 'spec-tag';
        tag.textContent = spec;
        specsContainer.appendChild(tag);
    });
    
    // Case logs
    const logsContainer = modal.querySelector('.case-logs-container');
    logsContainer.innerHTML = '';
    operator.caseLogs.forEach(log => {
        const logDiv = document.createElement('div');
        logDiv.className = 'case-log';
        logDiv.innerHTML = `
            <div class="case-log-header">
                <span class="case-log-id">${log.id}</span>
                <span class="case-log-date">${log.date}</span>
            </div>
            <div class="case-log-text ${log.classified ? 'redacted' : ''}">
                ${log.text}
            </div>
        `;
        logsContainer.appendChild(logDiv);
        
        // Time-based censorship for classified logs
        if (log.classified) {
            setTimeout(() => {
                logDiv.classList.add('censored');
            }, 8000 + Math.random() * 5000);
        }
    });
    
    // Show modal
    modal.classList.remove('hidden');
    
    // Close handler
    const closeBtn = modal.querySelector('.modal-close');
    const closeHandler = () => {
        modal.classList.add('hidden');
        closeBtn.removeEventListener('click', closeHandler);
    };
    closeBtn.addEventListener('click', closeHandler);
}
