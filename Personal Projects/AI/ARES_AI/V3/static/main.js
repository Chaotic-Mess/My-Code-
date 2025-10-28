const messagesContainer = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const typing = document.getElementById('typing');
let hasMessages = false;

function clearWelcome() {
    if (!hasMessages) {
        messagesContainer.innerHTML = '';
        hasMessages = true;
    }
}

function addMessage(text, isUser) {
    clearWelcome();

    const message = document.createElement('div');
    message.className = `message ${isUser ? 'user' : 'bot'}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? 'U' : 'A';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = text;

    message.appendChild(avatar);
    message.appendChild(content);
    messagesContainer.appendChild(message);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function send() {
    const message = input.value.trim();
    if (!message) return;

    addMessage(message, true);
    input.value = '';
    sendBtn.disabled = true;
    typing.classList.add('active');

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await res.json();
        typing.classList.remove('active');
        addMessage(data.response, false);
    } catch (err) {
        typing.classList.remove('active');
        addMessage('Error: Could not reach server', false);
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
    }
});

input.focus();