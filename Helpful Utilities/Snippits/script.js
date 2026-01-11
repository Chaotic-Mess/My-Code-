document.addEventListener('DOMContentLoaded', () => {
    // Copy to clipboard functionality
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', () => {
            const codeBox = button.parentElement;
            const code = codeBox.querySelector('code').innerText;
            
            navigator.clipboard.writeText(code).then(() => {
                const originalText = button.innerText;
                button.innerText = 'Copied!';
                button.classList.add('copied');
                
                setTimeout(() => {
                    button.innerText = originalText;
                    button.classList.remove('copied');
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy: ', err);
            });
        });
    });

    // Simple scroll animation trigger
    const observerOptions = {
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
});
