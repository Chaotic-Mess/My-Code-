// settings.js
// Enhanced theme management with improved error handling and performance
(function () {
    'use strict';

    const STORAGE_KEY = 'cppExamTheme';
    const DURATION_KEY = 'cppExamTransitionDuration';

    const defaultTheme = {
        bg: '#0b0f14',
        panel: '#11161d',
        text: '#e6e6e6',
        muted: '#9aa3af',
        accent: '#4f8cff',
        logo: '#4f8cff',
        border: '#1f2937',
        syntaxKeyword: '#ff7e94',
        syntaxType: '#78d4ff',
        syntaxString: '#a8e6a1',
        syntaxNumber: '#f5c28d',
        syntaxComment: '#707070',
        syntaxFunction: '#e6b3ff'
    };

    /**
     * Convert color object keys to CSS variable names
     * @param {string} key - Camel case color key
     * @returns {string} CSS variable name
     */
    function toCSSVar(key) {
        return '--' + key.replace(/([A-Z])/g, '-$1').toLowerCase();
    }

    /**
     * Apply theme colors to CSS variables
     * @param {Object} colors - Color object
     */
    function applyThemeVars(colors) {
        if (!colors || typeof colors !== 'object') {
            console.warn('Invalid theme colors, using defaults');
            colors = defaultTheme;
        }

        const root = document.documentElement;
        Object.entries(colors).forEach(([key, value]) => {
            if (typeof value === 'string' && value.match(/^#[0-9a-fA-F]{6}$/)) {
                root.style.setProperty(toCSSVar(key), value);
            }
        });

        // Always provide a logo color, falling back to accent or text
        const logoColor = colors.logo || colors.accent || colors.text || '#ffffff';
        root.style.setProperty('--logo', logoColor);
    }

    /**
     * Get stored theme from localStorage
     * @returns {Object} Theme colors
     */
    function getStoredTheme() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                const parsed = JSON.parse(stored);
                // Handle both old and new format
                return parsed.colors || parsed;
            }
        } catch (error) {
            console.warn('Error reading stored theme:', error);
        }
        return defaultTheme;
    }

    /**
     * Apply transition duration from storage
     */
    function applyTransitionDuration() {
        try {
            const stored = localStorage.getItem(DURATION_KEY);
            const ms = stored ? parseInt(stored, 10) : 300;

            // Validate duration is within reasonable bounds
            const validMs = Math.max(0, Math.min(2000, ms));
            const seconds = (validMs / 1000).toFixed(2);

            document.documentElement.style.setProperty(
                '--theme-transition-duration',
                `${seconds}s`
            );
        } catch (error) {
            console.warn('Error applying transition duration:', error);
            document.documentElement.style.setProperty(
                '--theme-transition-duration',
                '0.3s'
            );
        }
    }

    /**
     * Initialize theme on page load
     */
    function initTheme() {
        // Apply transition duration first for smooth initial render
        applyTransitionDuration();

        // Then apply theme colors
        const theme = getStoredTheme();
        applyThemeVars(theme);

        // Mark document as theme-ready
        document.documentElement.classList.add('theme-loaded');
    }

    // Initialize immediately
    initTheme();

    // Re-initialize on page show (handles back/forward cache)
    window.addEventListener('pageshow', function (event) {
        if (event.persisted) {
            initTheme();
        }
    });

    // Expose utilities for settings page
    if (typeof window !== 'undefined') {
        window.themeUtils = {
            applyThemeVars,
            getStoredTheme,
            applyTransitionDuration,
            toCSSVar,
            defaultTheme
        };
    }
})();
