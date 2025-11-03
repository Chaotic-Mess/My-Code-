# Ares • Odin • Nova — My AI Portfolio

Three different takes on personal AI — from a **homegrown LLM built with only the Python standard library** to a **voice assistant** and a **desktop companion**.

> Quick jumps: [ARES](#ares-homegrown-llm) · [ODIN](#odin-voice-assistant) · [NOVA](#nova-desktop-companion) · [Comparison](#comparison) · [Setup](#setup--run) · [Screenshots](#screenshots) · [Roadmap](#roadmap)

---
## tl;dr

- **ARES** — a from-scratch, stdlib-only character LLM: trains, checkpoints, and serves a web chat.  
- **ODIN** — a voice assistant pipeline (wake word → STT → LLM → TTS → actions).  
- **NOVA** — an Electron desktop companion with voice/overlay, memory, and hooks for vision.
```
---
```
## Repository Layout
 
```
.
├─ ARES_AI/V3               # Homegrown LLM (pure Python)
│  ├─ app.py                # tiny HTTP server (stdlib)
│  ├─ train.py              # trainer with ETA/checkpoints/temp/top-k
│  ├─ mymath.py             # list-based math (no numpy)
│  ├─ data/                 # corpus (tiny_shakespeare.txt)
│  ├─ model/                # TinyCharRNN + tokenizer
│  ├─ static/               # index.html / main.js / style.css
│  └─ weights/              # checkpoints (atomic JSON)
│
├─ ODIN_AI/                 # Voice assistant
│  ├─ src/                  # assistant code
│  ├─ requirements.txt      # Python deps (STT/TTS/etc.)
│  └─ README.md
│
├─ NOVA_AI/                 # Desktop companion (Electron)
│  ├─ electron/             # overlay app (Node/Electron)
│  ├─ backend/              # local hooks/models
│  └─ README.md
│
└─ README.md                # you are here (portfolio overview)

```

---

## ARES (homegrown LLM)

**What it is:** a character-level RNN language model implemented end-to-end with only Python’s standard library. Trains on Shakespeare (or any text) and serves a tiny chat web UI.

**Highlights**
- Pure stdlib: **no NumPy, no PyTorch, no Flask**
- Checkpoints (atomic JSON) + auto-resume
- ETA & throughput, live preview samples, progress panel
- Temperature & top-k decoding

**Run it**
```bash
cd ARES_AI
python app.py           # → http://localhost:8000  (chat UI)
# in another terminal:
python train.py         # trains; writes weights/ + progress_*.txt
````

**Tune it**

* In `train.py`: `TOTAL_STEPS`, `BLOCK_LEN`, `BASE_LR`, `SAMPLE_EVERY`, `SAVE_EVERY`,
  `PREVIEW_TEMP`, `PREVIEW_TOPK`
* In `app.py`: tweak `temperature` and `top_k` in the `model.generate(...)` call

---

## ODIN (voice assistant)

**What it is:** a local voice assistant pipeline — wake word → STT → LLM → TTS → desktop actions.

**Typical stack**

* **STT**: local engine (e.g., Vosk)
* **LLM**: local runtime (e.g., Ollama) or pluggable backend
* **TTS**: local TTS (e.g., pyttsx3)
* Optional integrations: calendars, system control, etc.

**Run it (example)**

```bash
cd ODIN_AI
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

> 🔐 Unavailible for Public Use (It's got Private API keys I haven't yet to hide).

---

## NOVA (desktop companion)

**What it is:** a desktop overlay companion with voice, memory, and optional vision hooks.

**Typical stack**

* **Electron** overlay UI (Node.js)
* Local STT/TTS, plus optional local LLMs
* Persistent memory (JSON/DB)

**Run it (example)**

```bash
cd NOVA_AI/electron
npm install
npm start
```

> 🔐 Unavailible for Public Use (It's got Private API keys I haven't yet to hide).
> It's really good at GeoGuesser!

---

## Comparison

| Aspect       | **ARES** (Homegrown LLM)         | **ODIN** (Voice Assistant)                   | **NOVA** (Desktop Companion)                |
| ------------ | -------------------------------- | -------------------------------------------- | ------------------------------------------- |
| Core model   | Pure Python **char-RNN**         | External/local LLM via a runtime             | External/local models                       |
| Dependencies | **Stdlib only**                  | Python libs for STT/TTS; local LLM runtime   | Node/Electron + optional Python backends    |
| Interface    | Web chat (tiny HTTP server)      | Voice: wake word → STT → LLM → TTS → actions | Desktop overlay + voice/memory/vision hooks |
| Offline      | Fully offline                    | if models local                              | if models local                             |
| Checkpoints  | Atomic JSON + auto-resume        | n/a (LLM external)                           | n/a (LLM external)                          |
| Best for     | Portfolio core: “I built an LLM” | Hands-free assistant and integrations        | Companion UX with presence and memory       |

---

## Setup & Run

### Prereqs

* **Python 3.10+** for ARES/ODIN
* **Node.js 18+** for NOVA Electron app

## Screenshots

* **ARES**: training preview sample (loss dropping), browser chat reply
* **ODIN**: a transcript of voice query → answer
* **NOVA**: overlay screenshot with a short interaction

```
/screenshots/
  ares_chat.png
  ares_training.png
  odin_voice.png
  nova_overlay.png
```

---

## Roadmap

* **ARES**: optional transformer path; mixed datasets; longer context blocks; simple `generate.py` CLI
* **ODIN**: pluggable tools (calendar/files); wake-word tuning; hotword model download helper
* **NOVA**: lighter packaging; model download scripts; memory inspector UI

---

## License

MIT — Idk. Just view only

---

**[← Back to Personal Projects](../README.md)** | **[← Back to Main Portfolio](../../README.md)**

`````
 
