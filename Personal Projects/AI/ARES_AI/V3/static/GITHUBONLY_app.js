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
  try {
    pyodide = await loadPyodide();
    await pyodide.runPythonAsync(`
def respond(msg: str) -> str:
    import random

    msg = msg.strip()
    if not msg:
        return "ARES_AI: You didn’t type anything! I’m ready when you are."

    text = msg.lower()

    greetings = ["hello", "hi", "hey", "yo"]
    farewells = ["bye", "goodbye", "later"]
    thanks = ["thanks", "thank you"]

    # Greeting responses
    if any(word in text for word in greetings):
        return random.choice([
            "ARES_AI: Hello there! I’m running entirely in your browser using Pyodide! Kinda...",
            "ARES_AI: Hi! No backend, no API — just pure static code here on GitHub Pages!",
            "ARES_AI: Greetings! I might not have server power, but I’m doing my best."
        ])

    # Farewell
    if any(word in text for word in farewells):
        return random.choice([
            "ARES_AI: Goodbye! Even static AIs need rest sometimes.",
            "ARES_AI: Farewell, human friend! I’ll be right here, waiting in the HTML.",
            "ARES_AI: Bye for now — reloading me is basically reincarnation."
        ])

    # Thanks
    if any(word in text for word in thanks):
        return random.choice([
            "ARES_AI: You’re welcome! Static, but still polite.",
            "ARES_AI: No problem! Courtesy requires no server calls.",
            "ARES_AI: My pleasure — even though I can’t *really* feel it."
        ])

    # “How does it work?”
    if "how" in text and "work" in text:
        return ("ARES_AI: I run completely client-side via Pyodide — that’s Python compiled to WebAssembly. "
                "No backend servers, no AI APIs — which means I can’t think or learn, just simulate responses locally."
                "If you were to download the V3 folder, then run app.py, you'll have an purely local AI! Trained? Probably not. I'm still learnin'")

    # Acknowledging GitHub limitations
    responses = [
        f"ARES_AI: I understood that! However, I can’t generate real AI answers due to GitHub’s static hosting limitations for Python scripts.",
        f"ARES_AI: You said '{msg}'. I can read and echo messages, but I can’t process them deeply — GitHub Pages doesn’t allow live AI execution.",
        f"ARES_AI: Got it! Sadly, my neural circuits are trapped in static HTML.",
        f"ARES_AI: Interesting message — I wish I could think about it, but I’m limited by GitHub’s deployment rules.",
        f"ARES_AI: That’s a good one. If only I had dynamic server access to respond properly!",
        f"ARES_AI: Processing… oh wait, I’m not actually connected to an AI backend. Curse you, static hosting!",
        f"ARES_AI: I get what you mean. Unfortunately, GitHub Pages doesn’t let me run real Python logic beyond this simulation.",
        f"ARES_AI: I feel your vibe! Though, technically, I can’t *feel* anything — I’m just static code pretending to be alive.",
        f"ARES_AI: Haha, nice! You know, I’m basically an AI hologram — no backend brain attached.",
        f"ARES_AI: I’d love to give a deep answer, but GitHub’s static setup prevents live AI inference.",
        f"ARES_AI: Message received! Though my Python brain is stuck in the browser sandbox.",
        f"ARES_AI: That’s fascinating! But I can’t generate meaningful insights without server processing power.",
        f"ARES_AI: You said '{msg}'. I’d respond intelligently, but static hosting keeps me simple and honest.",
        f"ARES_AI: I understand that, but my responses are prewritten — GitHub’s static nature prevents actual AI reasoning.",
        f"ARES_AI: Appreciate your input! Sadly, my AI core is disabled due to static deployment limitations."
    ]

    return random.choice(responses)
    `);
    pyodideReady = true;
  } catch (err) {
    console.error("Failed to load Pyodide:", err);
  }
}
loadPy();

// ---- UI Helpers ----
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

// ---- Messaging Logic ----
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
      // Safely encode input for Python
      const safeMsg = message
        .replace(/\\/g, "\\\\")  // escape backslashes
        .replace(/"/g, '\\"')    // escape quotes
        .replace(/\n/g, "\\n");  // escape newlines

      response = await pyodide.runPythonAsync(`respond("${safeMsg}")`);
    }
    addMessage(response, false);
  } catch (err) {
    console.error(err);
    addMessage('ARES_AI: Error — my local Python brain just crashed!', false);
  } finally {
    typing.classList.remove('active');
    sendBtn.disabled = false;
    input.focus();
  }
}

// ---- Event Listeners ----
sendBtn.addEventListener('click', send);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
