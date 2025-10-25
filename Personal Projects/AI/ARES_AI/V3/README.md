# ARES_AI — Pure-Python LLM (homegrown)

A from-scratch character-level language model you can train and chat with using **only the Python standard library**.  
No NumPy. No PyTorch. No Flask. Just Me, Python, and grit. 

SO much nicer to run, still takes time to train but still nice. No imports aside from Python which I obv needed anyway.

> Trains on `data/tiny_shakespeare.txt`, serves a minimal web UI at `http://localhost:8000`, and chats via a tiny `/chat` endpoint.

---

## ✨ Features

- **Pure Python**: no third-party packages or native deps
- **TinyCharRNN**: one-layer Elman RNN with embeddings (all list-based math in `mymath.py`)
- **Tokenizer**: simple character tokenizer (`model/tokenizer.py`)
- **Training**: ETA, moving-average step timing, LR decay, periodic previews
- **Sampling**: temperature + top-k decoding, seed priming (e.g. `ROMEO:\n`)
- **Checkpoints**: atomic `model_step_XXXX.json` + `ckpt.json` for auto-resume
- **Progress logs**: `progress_latest.txt` (for live panel) + `progress_history.txt`
- **Web UI**: static HTML/JS/CSS served by a tiny stdlib server (`app.py`)
- **Zero-config**: `python app.py` and `python train.py`—that’s it

---

## 📦 Project Layout

V3/
├─ app.py # tiny HTTP server (stdlib) + /chat endpoint
├─ train.py # trainer with ETA, checkpoints, previews
├─ mymath.py # pure-Python math ops (lists, not numpy)
├─ data/
│ └─ tiny_shakespeare.txt # training corpus
├─ model/
│ ├─ model.py # TinyCharRNN (temperature + top-k + atomic save)
│ ├─ tokenizer.py # CharTokenizer
│ └─ transformer.py # (optional/experimental; not required for RNN)
├─ static/
│ ├─ index.html # chat UI (+ optional training status panel)
│ ├─ main.js
│ └─ style.css
├─ weights/
│ ├─ model.json # latest copy (atomic)
│ ├─ ckpt.json # pointer to latest step file (atomic)
│ └─ model_step_XXXX.json # per-step checkpoints (atomic)
├─ progress_latest.txt # overwritten each preview
├─ progress_history.txt # append-only run history
├─ .gitignore
├─ README.md
└─ requirements.txt # (empty or comment-only; no pip deps)

---

## 🚀 Quickstart

### 1) Run the web UI
```bash
python app.py
# → ARES_AI running → http://localhost:8000
2) Train the model (new terminal)
python train.py

```
The trainer prints throughput (steps/sec), ETA, loss, and previews.

It writes checkpoints to weights/ and progress logs to progress_*.txt.

Auto-resume: if you interrupt (Ctrl+C), just run python train.py again.

It reloads weights/ckpt.json (or falls back to the newest model_step_*.json).

3) Chat in your browser
```
Open http://localhost:8000, type into the chat box, hit Send.
```
The server calls model.generate(..., temperature=0.8, top_k=50) by default.
Adjust those in app.py to change style/creativity.

⚙️ Configuration (trainer)

Edit the top of train.py:
```
Setting	Default	Meaning
BLOCK_LEN	128	Truncated BPTT context window
TOTAL_STEPS	20000	Training iterations
SAMPLE_EVERY	1000	Preview cadence (higher = faster training)
SAVE_EVERY	1000	Checkpoint cadence
BASE_LR	0.03	Base learning rate (halves every 10k steps)
PREVIEW_TEMP	0.8	Temperature used for previews
PREVIEW_TOPK	50	Top-k cutoff for previews (None to disable)
```
Where to change creativity
```
Training previews: edit PREVIEW_TEMP / PREVIEW_TOPK in train.py

Browser chat: edit the model.generate(...) call in app.py
```
🧪 Generation tips
```
Temperature:

0.6–0.9 → safer, more coherent

1.0–1.3 → creative/chaotic

Top-k: keeps only the k most likely tokens before sampling.

top_k=40–80 usually feels good for char models.
```
Priming / role prompts
Give it structure in the seed:
```
[SYSTEM]: You are ARES, an artificial mind born of pure code.
[USER]: Hello.
[ARES]:
```

Then call:

```
model.generate(tok, seed=that_text, max_new=200, temperature=0.8, top_k=50)
```
💾 Checkpoints & Resume (atomic)

The trainer saves atomically to avoid corrupt files if interrupted mid-write:

weights/model_step_XXXX.json — per-step snapshots (atomic)

weights/model.json — latest copy (atomic)

weights/ckpt.json — pointer { "path": "...", "step": N } (atomic)

Resume logic:

Try ckpt.json (if valid)

Else pick the highest model_step_*.json

Else start from scratch

If you ever suspect a partial file (rare now):

Delete the bad model_step_XXXX.json

Ensure ckpt.json points to the last good step file

  

🧩 Requirements

Python 3.10+ (tested on 3.12)

No pip installs required 

🔧 Troubleshooting

404 for style.css/main.js
Ensure the HTML links are /static/style.css and /static/main.js, and that app.py serves static/.

KeyboardInterrupt during save
Writes are atomic now. If you killed it twice mid-save and a file looks broken, delete that one model_step_*.json and re-run; the trainer will fall back gracefully.

Server says 501 Unsupported method ('GET')
That happens if you GET a POST-only route (like /chat). Load / or /static/index.html in your browser; let the UI handle POSTs.

📜 License

Idk, just dont steal ig

🙏 Acknowledgements

Shakespeare corpus: tiny_shakespeare.txt (public domain compilation)

Lots of inspiration from classic char-RNNs—this repo keeps it homegrown and stdlib-only.

🗺️ Roadmap (ideas)

Optional Transformer path (use model/transformer.py)

Mixed datasets (add a small chat corpus for better conversational flow)

Longer context (BLOCK_LEN 256/512) if your CPU can handle it

Simple export script to freeze only weights + vocab for distribution


Happy hacking. 🛠️
