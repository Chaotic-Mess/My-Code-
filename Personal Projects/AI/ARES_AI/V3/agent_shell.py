import os, re, webbrowser, subprocess
from model.model import TinyCharRNN
from model.tokenizer import CharTokenizer

# Load model + tokenizer (same as chat mode)
tok = CharTokenizer(open("data/tiny_shakespeare.txt", encoding="utf-8").read())
model = TinyCharRNN.load("weights/model.json")

def interpret(command: str):
    command = command.lower().strip()

    if "search google for" in command:
        query = command.split("search google for", 1)[1].strip()
        os.system(f'start "" "https://www.google.com/search?q={query.replace(" ", "+")}"')
        return f"🧠 Searching Google for: {query}"

    elif "open vs code" in command or "launch vscode" in command:
        subprocess.Popen(["code"])
        return "🧠 Launching Visual Studio Code"

    elif "send email" in command or "draft email" in command:
        # placeholder (later you can integrate with SMTP or a dummy text file)
        return "📨 Email draft simulated (no network access for safety)."

    elif "exit" in command:
        return "Goodbye, Commander."

    else:
        return "ARES doesn’t recognize that command yet."

def chat_loop():
    print("ARES Agent Shell ready. Type your commands.")
    while True:
        user = input("\nYou: ")
        if user.lower() in {"exit", "quit"}:
            print("Exiting shell.")
            break

        # Get ARES's interpretation (optional; or just skip straight to interpret())
        seed = f"[USER]: {user}\n[ARES]:"
        reply = model.generate(tok, seed=seed, max_new=120, temperature=0.8, top_k=50)
        print(f"\nARES: {reply.split('[ARES]:')[-1].strip()}\n")

        # Interpret + execute safe actions
        print(interpret(reply))

if __name__ == "__main__":
    chat_loop()
