# 🌐 My Projects Portfolio  

A collection of interactive web and programming projects — showcasing algorithms, simulations, AI systems, and academic work.  
Click the ***blue project titles*** to open working demos hosted on **GitHub Pages**.

---

## 🧩 Personal Projects  

![Personal Project](https://img.shields.io/badge/PERSONAL%20PROJECT-6A5ACD?style=for-the-badge&logo=github&logoColor=white)  
![Highlighted Item](https://img.shields.io/badge/HIGHLIGHTED%20ITEM-FFD700?style=for-the-badge&logo=starship&logoColor=black)

---

### [Video to ASCII](https://chaotic-mess.github.io/My-Code-/Personal%20Projects/VideoToASCII/index.html)

Convert videos into **animated ASCII art** — complete with synchronized **audio** playback.

**Features**
- Upload your own video or use built-in samples  
- Real-time ASCII rendering with adjustable resolution  
- Frame scrubber and pause/resume controls  
- Custom character sets (basic, extended, retro)  
- 100% client-side — runs entirely in the browser

**Tech:** HTML5, CSS3, JavaScript, Canvas API, Web Audio

---

### [Sorting Algorithm Visualizer](https://chaotic-mess.github.io/My-Code-/Personal%20Projects/SortingVisualizer/index.html)

Explore and compare sorting algorithms through colorful, animated bar charts.

**Features**
- Adjustable array size and speed  
- Real-time performance stats  
- 5 classic sorting algorithms (Bubble, Selection, Insertion, Quick, Merge)

**Tech:** HTML5, CSS3, JavaScript

---

### [3D Solar System](https://chaotic-mess.github.io/My-Code-/Personal%20Projects/3D_SolarSystem/index.html)

An immersive, orbit-accurate 3D simulation of our solar system.

**Features**
- Fully 3D interactive scene (orbit, zoom, click planets)  
- Realistic scale and speed ratios  
- Saturn’s rings, orbit trails, and star background  
- Planet info cards

**Tech:** HTML5, CSS3, JavaScript, Three.js (WebGL)

---

### [Pathfinding System](https://chaotic-mess.github.io/My-Code-/Personal%20Projects/PathfindingSystem/index.html)

![Personal Project](https://img.shields.io/badge/PERSONAL%20PROJECT-6A5ACD?style=for-the-badge&logo=github&logoColor=white)
![Highlighted Item](https://img.shields.io/badge/HIGHLIGHTED%20ITEM-FFD700?style=for-the-badge&logo=starship&logoColor=black)

A dynamic grid-based pathfinding visualizer featuring A*, Dijkstra, BFS, and DFS.

**Features**
- Real-time algorithm visualization  
- Interactive grid editing (set start/end, draw walls)  
- Random maze generator  
- Adjustable speed and metrics display

**Tech:** HTML5, CSS3, JavaScript

---

### [Farm Drone Game](https://chaotic-mess.github.io/My-Code-/Personal%20Projects/FarmDroneGame/index.html) *(Work in Progress)*  

A **C++-style automation simulation** inspired by *The Farmer Was Replaced*, reimagined for the web.  
Write code to automate farming drones that plant, water, and harvest crops efficiently.

**Features**
- Drone scripting with C++-like logic  
- Grid-based field simulation  
- Step-by-step execution visualization  
- Local save/load  
- GitHub Pages compatible (pure JS)

**Tech:** HTML5, CSS3, JavaScript (planned expansion: WebAssembly)

---

## 🤖 AI Projects — *Ares • Odin • Nova*  

Three different takes on personal AI — from a **homegrown LLM built with only the Python standard library** to a **voice assistant** and a **desktop companion**.

> Quick jumps: [ARES](#ares-homegrown-llm) · [ODIN](#odin-voice-assistant) · [NOVA](#nova-desktop-companion) · [Comparison](#comparison) · [Setup](#setup--run) · [Screenshots](#screenshots) · [Roadmap](#roadmap)

---

### tl;dr

- **ARES** — a from-scratch, stdlib-only character LLM: trains, checkpoints, and serves a web chat.  
- **ODIN** — a voice assistant pipeline (wake word → STT → LLM → TTS → actions).  
- **NOVA** — an Electron desktop companion with voice/overlay, memory, and hooks for vision.

---

### Repository Layout

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

### ARES (homegrown LLM)

A character-level RNN language model implemented end-to-end with only Python’s standard library. Trains on Shakespeare (or any text) and serves a tiny chat web UI.

**Highlights**
- Pure stdlib: **no NumPy, no PyTorch, no Flask**
- Checkpoints (atomic JSON) + auto-resume
- ETA & throughput, live preview samples, progress panel
- Temperature & top-k decoding

**Run it**
```bash
cd ARES_AI
python app.py           # → http://localhost:8000
python train.py         # trains; writes weights/ + progress_*.txt
```

---

### ODIN (voice assistant)

A local voice assistant pipeline — wake word → STT → LLM → TTS → desktop actions.

**Stack**
- **STT**: local engine (e.g., Vosk)
- **LLM**: local runtime (e.g., Ollama)
- **TTS**: local speech synthesis (e.g., pyttsx3)
- Optional integrations: calendars, system control, etc.

**Run it**
```bash
cd ODIN_AI
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python main.py
```
> 🔐 Not publicly available — contains private API keys.

---

### NOVA (desktop companion)

A desktop overlay companion with voice, memory, and optional vision hooks.

**Stack**
- Electron overlay UI (Node.js)
- Local STT/TTS + optional local LLMs
- Persistent memory (JSON/DB)

**Run it**
```bash
cd NOVA_AI/electron
npm install
npm start
```
> 🔐 Not publicly available — contains private API keys.  
> 💬 Known for being really good at GeoGuessr!

---

### Comparison

| Aspect       | **ARES** (Homegrown LLM)         | **ODIN** (Voice Assistant)                   | **NOVA** (Desktop Companion)                |
| ------------ | -------------------------------- | -------------------------------------------- | ------------------------------------------- |
| Core model   | Pure Python **char-RNN**         | External/local LLM via a runtime             | External/local models                       |
| Dependencies | **Stdlib only**                  | Python libs for STT/TTS; local LLM runtime   | Node/Electron + optional Python backends    |
| Interface    | Web chat (tiny HTTP server)      | Voice: wake word → STT → LLM → TTS → actions | Desktop overlay + voice/memory/vision hooks |
| Offline      | Fully offline                    | If models local                              | If models local                             |
| Checkpoints  | Atomic JSON + auto-resume        | n/a (LLM external)                           | n/a (LLM external)                          |
| Best for     | Portfolio core: “I built an LLM” | Hands-free assistant and integrations        | Companion UX with presence and memory       |

---

### Setup & Run

**Prereqs**
- Python 3.10+ for ARES/ODIN  
- Node.js 18+ for NOVA  

**Screenshots**
```
/screenshots/
  ares_chat.png
  ares_training.png
  odin_voice.png
  nova_overlay.png
```

---

### Roadmap

- **ARES**: optional transformer path, mixed datasets, longer context  
- **ODIN**: plugin tools (calendar/files), wake-word tuning  
- **NOVA**: lighter packaging, memory inspector UI  

---

### License

MIT — for viewing only.

---

## 🎓 University of Victoria  

![School Work](https://img.shields.io/badge/SCHOOL%20WORK-008080?style=for-the-badge&logo=google-classroom&logoColor=white)

### CSC111 – Introduction to Programming  
Python projects and exercises exploring fundamental programming concepts.

### CSC116 – Data Structures & Algorithms  
Focused on algorithm analysis, recursion, and OOP — with visual JS versions.

---

## 🧰 Tech Stack  

![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Three.js](https://img.shields.io/badge/Three.js-000000?style=for-the-badge&logo=three.js&logoColor=white)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-222222?style=for-the-badge&logo=githubpages&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🔗 Connect With Me  

Explore the projects, fork anything you like, and reach out with questions or ideas!  
⭐ *Don’t forget to star your favorites on GitHub!*

---

*Last Updated: October 2025*
