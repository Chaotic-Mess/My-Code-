// ═══════════════════════════════════════════════════════════
// TERMINAL SYSTEM - Deep access interface
// ═══════════════════════════════════════════════════════════

const COMMANDS = {
    help: {
        description: 'Display available commands',
        execute: () => {
            return `Available commands:
  help       - Display this message
  list       - List all operators
  query      - Query operator by codename
  status     - System status
  missions   - Active mission count
  clearance  - Check clearance level
  trace      - Initialize trace protocol
  encrypt    - Encrypt message
  decrypt    - Decrypt message
  exit       - Close terminal`;
        }
    },
    list: {
        description: 'List all operators',
        execute: () => {
            const names = [
                'PHANTOM', 'ECLIPSE', 'GHOST', 'SHADOW', 'VIPER',
                'RAVEN', 'COBRA', 'WRAITH', 'REAPER', 'SPECTRE',
                'CIPHER', 'NOMAD', 'RAZOR', 'VOID', 'NEXUS',
                'ONYX', 'SABLE', 'OBSIDIAN', 'SMOKE', 'ASH',
                'FROST', 'BLAZE', 'STORM', 'THUNDER', 'LIGHTNING',
                'TITAN', 'ATLAS', 'ZEUS', 'HADES', 'ARES',
                'APEX', 'OMEGA', 'ALPHA', 'DELTA', 'SIGMA',
                'KILO', 'ZULU'
            ];
            return `37 OPERATORS REGISTERED:\n${names.join(' | ')}`;
        }
    },
    query: {
        description: 'Query operator by codename',
        execute: (args) => {
            if (!args[0]) {
                return 'ERROR: Codename required. Usage: query [CODENAME]';
            }
            const codename = args[0].toUpperCase();
            return `QUERYING: ${codename}
STATUS: ${Math.random() > 0.5 ? 'ACTIVE' : 'DORMANT'}
LOCATION: ████████
CLEARANCE: OMEGA-${Math.floor(Math.random() * 9) + 1}
LAST CONTACT: ${Math.floor(Math.random() * 72)}H AGO`;
        }
    },
    status: {
        description: 'System status',
        execute: () => {
            return `AEON MAINFRAME STATUS:
CONNECTION: SECURE
ENCRYPTION: AES-512-QUANTUM
OPERATORS ACTIVE: 37
MISSIONS IN PROGRESS: ${Math.floor(Math.random() * 15) + 10}
THREAT LEVEL: ${['LOW', 'MODERATE', 'HIGH', 'CRITICAL'][Math.floor(Math.random() * 4)]}
LAST BREACH ATTEMPT: ${Math.floor(Math.random() * 168)}H AGO`;
        }
    },
    missions: {
        description: 'Active mission count',
        execute: () => {
            const count = Math.floor(Math.random() * 15) + 10;
            return `${count} ACTIVE MISSIONS
  - ${Math.floor(count * 0.6)} IN PROGRESS
  - ${Math.floor(count * 0.3)} NEAR COMPLETION
  - ${Math.floor(count * 0.1)} COMPROMISED`;
        }
    },
    clearance: {
        description: 'Check clearance level',
        execute: () => {
            return `YOUR CLEARANCE: GUEST
ACCESS LEVEL: 1/10
RESTRICTIONS: ACTIVE
Note: Elevated access requires authorization code.`;
        }
    },
    trace: {
        description: 'Initialize trace protocol',
        execute: () => {
            return `INITIALIZING TRACE PROTOCOL...
IP: ${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}
GEOLOCATION: ████████, ████████
ISP: ████████████
DEVICE: ████████
WARNING: This connection is being monitored.`;
        }
    },
    encrypt: {
        description: 'Encrypt message',
        execute: (args) => {
            if (!args[0]) {
                return 'ERROR: Message required. Usage: encrypt [MESSAGE]';
            }
            const encrypted = btoa(args.join(' '));
            return `ENCRYPTED: ${encrypted}`;
        }
    },
    decrypt: {
        description: 'Decrypt message',
        execute: (args) => {
            if (!args[0]) {
                return 'ERROR: Encrypted message required. Usage: decrypt [ENCRYPTED]';
            }
            try {
                const decrypted = atob(args[0]);
                return `DECRYPTED: ${decrypted}`;
            } catch {
                return 'ERROR: Invalid encrypted message.';
            }
        }
    },
    exit: {
        description: 'Close terminal',
        execute: () => {
            return 'CONNECTION TERMINATED.';
        }
    },
    // Hidden easter egg commands
    ghost: {
        hidden: true,
        execute: () => {
            return `You've found the ghost protocol.
"In the age of surveillance, invisibility is power."`;
        }
    },
    aeon: {
        hidden: true,
        execute: () => {
            return `AEON SERVICES
Est. ████
"We are the silence between the screams."`;
        }
    }
};

export function initTerminal() {
    const input = document.getElementById('terminal-input');
    const output = document.getElementById('terminal-output');
    
    if (!input || !output) return;
    
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const command = input.value.trim();
            if (command) {
                executeCommand(command, output);
                input.value = '';
            }
        }
    });
    
    // Focus on terminal when section is active
    const terminalNav = document.querySelector('[data-section="terminal"]');
    if (terminalNav) {
        terminalNav.addEventListener('click', () => {
            setTimeout(() => input.focus(), 100);
        });
    }
}

function executeCommand(commandString, output) {
    // Add command to output
    addLine(output, `AEON> ${commandString}`, 'terminal-line');
    
    const parts = commandString.split(' ');
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);
    
    if (COMMANDS[cmd]) {
        const result = COMMANDS[cmd].execute(args);
        addLine(output, result, 'terminal-line');
    } else {
        addLine(output, `ERROR: Unknown command '${cmd}'. Type 'help' for available commands.`, 'terminal-error');
    }
    
    // Scroll to bottom
    output.scrollTop = output.scrollHeight;
}

function addLine(output, text, className) {
    const lines = text.split('\n');
    lines.forEach(line => {
        const div = document.createElement('div');
        div.className = className;
        div.textContent = line;
        output.appendChild(div);
    });
}
