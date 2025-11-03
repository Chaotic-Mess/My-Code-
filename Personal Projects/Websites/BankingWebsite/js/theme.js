// Theme management
const themeToggle = document.getElementById('theme-toggle');
const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

// Theme colors and values
const themes = {
    dark: {
        primary: '#8ab4ff',
        primaryLight: 'rgba(138, 180, 255, 0.1)',
        primaryMedium: 'rgba(138, 180, 255, 0.3)',
        background: '#0a0a1a',
        surface: 'rgba(10, 10, 26, 0.8)',
        surfaceLight: 'rgba(10, 10, 26, 0.6)',
        textPrimary: '#e2e8f0',
        textSecondary: 'rgba(138, 180, 255, 0.8)',
        textMuted: 'rgba(138, 180, 255, 0.6)',
        overlay: 'linear-gradient(45deg, rgba(10, 10, 26, 0.85), rgba(10, 10, 26, 0.7))',
        cardBg: 'rgba(10, 10, 26, 0.6)',
        cardBorder: 'rgba(138, 180, 255, 0.2)',
        navBg: 'rgba(10, 10, 26, 0.8)',
        buttonBg: 'linear-gradient(45deg, rgba(138, 180, 255, 0.1), rgba(138, 180, 255, 0.2))',
        buttonBorder: 'rgba(138, 180, 255, 0.3)',
        buttonHover: 'rgba(138, 180, 255, 0.3)',
        shadowColor: 'rgba(138, 180, 255, 0.3)',
        footerBg: 'rgba(10, 10, 26, 0.8)',
        footerBorder: 'rgba(138, 180, 255, 0.1)'
    },
    light: {
        primary: '#2563eb',
        primaryLight: 'rgba(37, 99, 235, 0.1)',
        primaryMedium: 'rgba(37, 99, 235, 0.3)',
        background: '#ffffff',
        surface: 'rgba(255, 255, 255, 0.9)',
        surfaceLight: 'rgba(255, 255, 255, 0.8)',
        textPrimary: '#1e293b',
        textSecondary: 'rgba(37, 99, 235, 0.8)',
        textMuted: 'rgba(37, 99, 235, 0.6)',
        overlay: 'linear-gradient(45deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.86))',
        cardBg: 'rgba(255, 255, 255, 0.9)',
        cardBorder: 'rgba(37, 99, 235, 0.1)',
        navBg: 'rgba(255, 255, 255, 0.9)',
        buttonBg: 'linear-gradient(45deg, rgba(37, 99, 235, 0.1), rgba(37, 99, 235, 0.2))',
        buttonBorder: 'rgba(37, 99, 235, 0.3)',
        buttonHover: 'rgba(37, 99, 235, 0.3)',
        shadowColor: 'rgba(37, 99, 235, 0.2)',
        footerBg: 'rgba(255, 255, 255, 0.9)',
        footerBorder: 'rgba(37, 99, 235, 0.1)'
    }
};

// Initialize theme
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const systemTheme = prefersDarkScheme.matches ? 'dark' : 'light';
    const currentTheme = savedTheme || systemTheme;
    
    document.documentElement.setAttribute('data-theme', currentTheme);
    if (themeToggle) {
        themeToggle.checked = currentTheme === 'dark';
    }
    
    updateThemeColors(currentTheme);
}

// Update theme colors
function updateThemeColors(theme) {
    const colors = themes[theme];
    const root = document.documentElement;
    
    Object.entries(colors).forEach(([key, value]) => {
        root.style.setProperty(`--${key}`, value);
    });
    
    // Update video overlay with better contrast
    const videoOverlay = document.querySelector('.video-overlay');
    if (videoOverlay) {
        if (theme === 'dark') {
            videoOverlay.style.background = 'linear-gradient(45deg, rgba(10, 10, 26, 0.75), rgba(10, 10, 26, 0.85))';
        } else {
            videoOverlay.style.background = 'linear-gradient(45deg, rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.92))';
        }
    }
    
    // Update stars visibility
    // const stars = document.querySelector('.stars');
    // const twinkling = document.querySelector('.twinkling');
    // if (stars && twinkling) {
    //     stars.style.opacity = theme === 'dark' ? '1' : '0';
    //     twinkling.style.opacity = theme === 'dark' ? '1' : '0';
    // }
}

// Toggle theme
function toggleTheme(event) {
    const newTheme = event.target.checked ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeColors(newTheme);
}

// Event listeners
if (themeToggle) {
    themeToggle.addEventListener('change', toggleTheme);
}

prefersDarkScheme.addListener((e) => {
    if (!localStorage.getItem('theme')) {
        const newTheme = e.matches ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        updateThemeColors(newTheme);
    }
});

// Initialize on load
initializeTheme();
