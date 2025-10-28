// ---- UI Elements ----
const messagesContainer = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const typing = document.getElementById('typing');
let hasMessages = false;
let pyodideReady = false;
let pyodide;

// ---- Load Pyodide ----
async function loadPy() {
  pyodide = await loadPyodide();
  await pyodide.runPythonAsync(`
def respond(msg: str) -> str:
    # Simple placeholder model that imitates AI behavior
    msg = msg.strip()
    if not msg:
        return "ARES_AI: I'm here, but you didn't say anything!"
    if "hello" in msg.lower():
        return "ARES_AI: Hello there! Running fully client-side on GitHub Pages."
    if "how" in msg.lower() and "work" in msg.lower():
        return "ARES_AI: I run entirely in your browser using Pyodide (Python in WebAssembly)."
    return f"ARES_AI: You said '{msg}', and I processed it locally in Python!"
  `);
  pyodideReady = true;
}
loadPy();

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
    let response = "ARES_AI: Python runtime not ready yet.";
    if (pyodideReady) {
      response = await pyodide.runPythonAsync(`respond("${message}")`);
    }
    typing.classList.remove('active');
    addMessage(response, false);
  } catch (err) {
    typing.classList.remove('active');
    addMessage('Error: Could not run local Python model.', false);
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
