// ═══════════════════════════════════════════════════════════
// AUDIO SYSTEM - Sound design for immersion
// ═══════════════════════════════════════════════════════════

export async function initAudio() {
    // Create audio context for Web Audio API
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContext();
    
    // Generate ambient hum
    const ambientHum = document.getElementById('ambient-hum');
    if (ambientHum) {
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(60, audioContext.currentTime); // Low frequency hum
        gainNode.gain.setValueAtTime(0.05, audioContext.currentTime);
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        // Start on user interaction (due to autoplay policies)
        document.addEventListener('click', () => {
            if (audioContext.state === 'suspended') {
                audioContext.resume();
                oscillator.start();
            }
        }, { once: true });
    }
    
    // Generate hover static sound
    const hoverStatic = document.getElementById('hover-static');
    if (hoverStatic) {
        hoverStatic.src = generateStaticNoise(audioContext, 0.1);
    }
    
    // Generate select bass drop
    const selectBass = document.getElementById('select-bass');
    if (selectBass) {
        selectBass.src = generateBassDrop(audioContext);
    }
    
    // Generate glitch sound
    const glitchSound = document.getElementById('glitch-sound');
    if (glitchSound) {
        glitchSound.src = generateGlitchNoise(audioContext);
    }
}

function generateStaticNoise(audioContext, duration) {
    const sampleRate = audioContext.sampleRate;
    const buffer = audioContext.createBuffer(1, sampleRate * duration, sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < buffer.length; i++) {
        data[i] = Math.random() * 0.2 - 0.1;
    }
    
    return bufferToWave(buffer);
}

function generateBassDrop(audioContext) {
    const sampleRate = audioContext.sampleRate;
    const duration = 0.5;
    const buffer = audioContext.createBuffer(1, sampleRate * duration, sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < buffer.length; i++) {
        const t = i / sampleRate;
        const frequency = 80 - (t * 60); // Drop from 80Hz to 20Hz
        data[i] = Math.sin(2 * Math.PI * frequency * t) * Math.exp(-t * 3);
    }
    
    return bufferToWave(buffer);
}

function generateGlitchNoise(audioContext) {
    const sampleRate = audioContext.sampleRate;
    const duration = 0.2;
    const buffer = audioContext.createBuffer(1, sampleRate * duration, sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < buffer.length; i++) {
        const t = i / sampleRate;
        if (Math.random() > 0.95) {
            data[i] = (Math.random() * 2 - 1) * 0.5;
        } else {
            data[i] = 0;
        }
    }
    
    return bufferToWave(buffer);
}

function bufferToWave(buffer) {
    const length = buffer.length * buffer.numberOfChannels * 2 + 44;
    const arrayBuffer = new ArrayBuffer(length);
    const view = new DataView(arrayBuffer);
    const channels = [];
    let offset = 0;
    let pos = 0;
    
    // Write WAV header
    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8); // file length - 8
    setUint32(0x45564157); // "WAVE"
    
    setUint32(0x20746d66); // "fmt " chunk
    setUint32(16); // length = 16
    setUint16(1); // PCM (uncompressed)
    setUint16(buffer.numberOfChannels);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * buffer.numberOfChannels);
    setUint16(buffer.numberOfChannels * 2);
    setUint16(16);
    
    setUint32(0x61746164); // "data" - chunk
    setUint32(length - pos - 4);
    
    // Write interleaved data
    for (let i = 0; i < buffer.numberOfChannels; i++) {
        channels.push(buffer.getChannelData(i));
    }
    
    while (pos < length) {
        for (let i = 0; i < buffer.numberOfChannels; i++) {
            let sample = Math.max(-1, Math.min(1, channels[i][offset]));
            sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }
    
    const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
    return URL.createObjectURL(blob);
    
    function setUint16(data) {
        view.setUint16(pos, data, true);
        pos += 2;
    }
    
    function setUint32(data) {
        view.setUint32(pos, data, true);
        pos += 4;
    }
}
