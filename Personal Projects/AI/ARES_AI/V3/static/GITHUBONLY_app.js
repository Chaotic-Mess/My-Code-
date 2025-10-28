<script type="module">
  // ---- UI Elements ----
  const messagesContainer = document.getElementById('messages');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('sendBtn');
  const typing = document.getElementById('typing');
  let hasMessages = false;
  let pyodideReady = false;
  let pyodideInstance;

  // ---- Load Pyodide ----
  async function loadPy() {
    try {
      // ✅ Avoid naming conflict with loadPyodide() global
      pyodideInstance = await loadPyodide();
      await pyodideInstance.runPythonAsync(`
def respond(msg: str) -> str:
    msg = msg.strip()
    if not msg:
        return "ARES_AI: I'm here, but you didn't say anything!"
    lower = msg.lower()
    if "hello" in lower:
        return "ARES_AI: Hello there! Running fully client-side on GitHub Pages."
    if "how" in lower and "work" in lower:
        return "ARES_AI: I run entirely in your browser using Pyodide (Python in WebAssembly)."
    return f"ARES_AI: You said '{msg}'. However, ARES cannot access server AI due to GitHub deployment restrictions."
`);
      pyodideReady = true;
    } catch (err) {
      console.error('Failed to load Pyodide:', err);
      addMessage('Error loading Python runtime. Please refresh the page.', false);
    }
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
      let response = 'ARES_AI: Python runtime not ready yet.';
      if (pyodideReady) {
        // ✅ Escape quotes properly to avoid breaking Python string
        const safeMessage = message.replace(/(["\\])/g, '\\$1');
        response = await pyodideInstance.runPythonAsync(`respond("${safeMessage}")`);
      }
      addMessage(response, false);
    } catch (err) {
      console.error(err);
      addMessage('Error: Could not run local Python model.', false);
    } finally {
      typing.classList.remove('active');
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
</script>
