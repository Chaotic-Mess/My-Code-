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
    import random, re

    msg = msg.strip()
    if not msg:
        return "ARES_AI: You didn’t say anything! Try asking me something — I’m listening."

    text = msg.lower()

    greetings = ["hello", "hi", "hey", "yo"]
    farewells = ["bye", "goodbye", "see you", "later"]
    thanks = ["thanks", "thank you"]
    question = "?" in text

    # Greeting
    if any(word in text for word in greetings):
        responses = [
            "ARES_AI: Hello there! Running fully client-side with Pyodide",
            "ARES_AI: Hi! Nice to see you — all powered by your browser.",
            "ARES_AI: Hey! I’m ARES_AI, a lightweight local Python assistant."
        ]
        return random.choice(responses)

    # Farewell
    if any(word in text for word in farewells):
        return "ARES_AI: Goodbye! Until next time, stay curious."

    # Gratitude
    if any(word in text for word in thanks):
        return "ARES_AI: You’re welcome! Glad to help — even without a server."

    # How it works
    if "how" in text and "work" in text:
        return ("ARES_AI: I’m running Python directly in your browser "
                "through Pyodide (WebAssembly). No servers, no tracking — just pure client-side code.")

    # Help requests
    if "help" in text:
        return ("ARES_AI: I can respond to simple text messages here in your browser. "
                "Try saying hello, asking how I work, or just chat casually!")

    # Questions
    if question:
        return f"ARES_AI: Hmm, that’s an interesting question — \"{msg}\". I can’t query the web yet, but I can think with you!"

    # Random fun responses
    playful = [
        "ARES_AI: That’s fascinating. Tell me more!",
        "ARES_AI: I see — processing that locally",
        "ARES_AI: Haha, good one!",
        "ARES_AI: Running Python in your browser never felt this chatty.",
    ]

    return f"ARES_AI: You said '{msg}'. {random.choice(playful)}"

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
