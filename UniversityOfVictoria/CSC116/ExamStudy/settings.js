// settings.js
// This script applies the selected theme to all pages by reading from localStorage and updating CSS variables.
(function() {
    const STORAGE_KEY = 'cppExamTheme';
    const DURATION_KEY = 'cppExamTransitionDuration';
    const defaultTheme = {
        bg: '#0b0f14',
        panel: '#11161d',
        text: '#e6e6e6',
        muted: '#9aa3af',
        accent: '#4f8cff',
        border: '#1f2937',
        syntaxKeyword: '#ff7e94',
        syntaxType: '#78d4ff',
        syntaxString: '#a8e6a1',
        syntaxNumber: '#f5c28d',
        syntaxComment: '#707070',
        syntaxFunction: '#e6b3ff'
    };
    function applyThemeVars(colors) {
        if (!colors) return;
        Object.entries(colors).forEach(([key, value]) => {
            const cssVar = '--' + key.replace(/([A-Z])/g, '-$1').toLowerCase();
            document.documentElement.style.setProperty(cssVar, value);
        });
    }
    function getStoredTheme() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                const parsed = JSON.parse(stored);
                if (parsed.colors) return parsed.colors;
                return parsed; // fallback for old format
            }
        } catch {}
        return defaultTheme;
    }
    function applyTransitionDuration() {
        try {
            const stored = localStorage.getItem(DURATION_KEY);
            const ms = stored ? parseInt(stored, 10) : 300;
            const seconds = (ms / 1000).toFixed(2);
            document.documentElement.style.setProperty('--theme-transition-duration', `${seconds}s`);
        } catch {
            document.documentElement.style.setProperty('--theme-transition-duration', '0.3s');
        }
    }
    applyThemeVars(getStoredTheme());
    applyTransitionDuration();
})();
