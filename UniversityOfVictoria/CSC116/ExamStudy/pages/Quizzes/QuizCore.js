/**
 * QuizCore.js - Shared quiz functionality
 * Provides reusable functions for all quiz pages
 */

// Quiz state (to be initialized by each quiz)
let selections = {};
let selectedButtons = {};

// Get a normalized value for an option button (supports data-text, data-value, or visible text)
function getOptionValue(btn) {
    // Prefer data-text first, then data-value, then normalize the visible button text
    if (btn.dataset.text) return btn.dataset.text.trim();
    if (btn.dataset.value) return btn.dataset.value.trim();
    
    // For plain text buttons, normalize by removing extra whitespace and line breaks
    return btn.textContent
        .replace(/\s+/g, ' ')  // collapse multiple whitespace/newlines into single spaces
        .trim();
}

/**
 * Shuffle an array in-place using Fisher-Yates algorithm
 */
function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

/**
 * Shuffle the order of option buttons within each question
 */
function shuffleOptions() {
    document.querySelectorAll('.quiz-options').forEach(container => {
        const opts = Array.from(container.querySelectorAll('.option'));
        shuffleArray(opts);
        opts.forEach(opt => container.appendChild(opt));
    });
}

/**
 * Shuffle the order of quiz questions
 */
function shuffleQuestions() {
    const form = document.getElementById('quiz-form');
    const blocks = Array.from(form.querySelectorAll('.quiz-question'));
    shuffleArray(blocks);
    blocks.forEach(block => form.appendChild(block));
}

/**
 * Re-number question headings after shuffle
 * Matches pattern "N. Question text"
 */
function renumberQuestions() {
    document.querySelectorAll('.quiz-question h3').forEach((h3, idx) => {
        const text = h3.textContent.replace(/^\d+\.\s*/, '');
        h3.textContent = `${idx + 1}. ${text}`;
    });
}

/**
 * Save the best quiz score to localStorage
 * Only updates if the new score is better than the previous best
 * @param {number} percent - Score as percentage (0-100)
 */
function saveBestScore(percent) {
    try {
        const key = `bestScore-${quizId}`;
        const prev = Number(localStorage.getItem(key)) || 0;
        if (percent > prev) {
            localStorage.setItem(key, String(percent));
        }
    } catch (err) {
        // Ignore storage errors (e.g., in private browsing)
    }
}

/**
 * Setup quiz event listeners
 * Must be called after DOM is fully loaded and shuffle is complete
 * Expects: answers, explanations, quizId to be defined in quiz HTML
 */
function setupQuizListeners() {
    document.querySelectorAll('.option').forEach(btn => {
        btn.addEventListener('click', function (e) {
            e.preventDefault();
            const question = this.dataset.question;
            const text = getOptionValue(this);

            // Store selection
            selections[question] = text;
            selectedButtons[question] = this;

            // Update UI - remove selected from all buttons for this question
            document.querySelectorAll(`[data-question="${question}"]`).forEach(b => {
                b.classList.remove('selected');
            });
            // Add selected to clicked button
            this.classList.add('selected');
        });
    });
}

/**
 * Submit the quiz and display results
 * Expects: answers, explanations, quizId to be defined in quiz HTML
 */
function submitQuiz() {
    let score = 0;
    const total = Object.keys(answers).length;

    for (const [question, correctAnswer] of Object.entries(answers)) {
        const buttons = document.querySelectorAll(`[data-question="${question}"]`);
        const feedbackEl = document.getElementById(`feedback${question.substring(1)}`);
            const selected = selections[question];
            const selectedBtn = selectedButtons[question];

        // Clear previous styling
        buttons.forEach(btn => btn.classList.remove('correct', 'incorrect'));

        if (selected) {
            const isCorrect = selected === correctAnswer;

            if (isCorrect) {
                score++;
                feedbackEl.className = 'feedback correct show';
                feedbackEl.textContent = '✓ Correct! ' + explanations[question];
                if (selectedBtn) selectedBtn.classList.add('correct');
            } else {
                feedbackEl.className = 'feedback incorrect show';
                feedbackEl.textContent = '✗ Incorrect. ' + explanations[question];
                if (selectedBtn) selectedBtn.classList.add('incorrect');

                // Find and highlight the correct button
                for (const btn of buttons) {
                    if (getOptionValue(btn) === correctAnswer) {
                        btn.classList.add('correct');
                        break;
                    }
                }
            }
        } else {
            feedbackEl.className = 'feedback incorrect show';
            feedbackEl.textContent = '⚠ Not answered. ' + explanations[question];
        }
    }

    // Display results
    const percentage = Math.round((score / total) * 100);
    saveBestScore(percentage);
    const resultsEl = document.getElementById('results');
    const scoreDisplay = document.getElementById('score-display');
    const scoreMessage = document.getElementById('score-message');

    scoreDisplay.textContent = `${score}/${total} (${percentage}%)`;
    scoreDisplay.className = 'score ' + (percentage >= 70 ? 'high' : 'low');

    // Personalized message based on score
    if (percentage >= 90) {
        scoreMessage.textContent = 'Excellent! You have a strong understanding of the material.';
    } else if (percentage >= 70) {
        scoreMessage.textContent = 'Good job! Review the incorrect answers to strengthen your knowledge.';
    } else if (percentage >= 50) {
        scoreMessage.textContent = 'You\'re on the right track. Review the content and try again.';
    } else {
        scoreMessage.textContent = 'Keep studying! Review the fundamentals.';
    }

    resultsEl.classList.add('show');
    window.scrollTo(0, 0);
}

/**
 * Reset the quiz to initial state
 */
function resetQuiz() {
    selections = {};
    selectedButtons = {};
    document.getElementById('results').classList.remove('show');
    document.querySelectorAll('.feedback').forEach(el => el.classList.remove('show'));
    document.querySelectorAll('.option').forEach(el => {
        el.classList.remove('correct', 'incorrect', 'selected');
    });
}

/**
 * Initialize quiz on page load
 * Call this in DOMContentLoaded event
 */
function initializeQuiz() {
    shuffleQuestions();
    shuffleOptions();
    renumberQuestions();
    setupQuizListeners();
}
