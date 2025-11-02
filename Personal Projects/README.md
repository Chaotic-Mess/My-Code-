# Personal Projects  

![Personal Project](https://img.shields.io/badge/PERSONAL%20PROJECT-6A5ACD?style=for-the-badge&logo=github&logoColor=white)

Interactive web demos, visualizers, and AI experiments.  
Click the ***blue project titles*** to open working demos hosted on **GitHub Pages**.

---

## Web Projects & Visualizers

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
- Saturn's rings, orbit trails, and star background  
- Planet info cards

**Tech:** HTML5, CSS3, JavaScript, Three.js (WebGL)

---

### [Pathfinding System](https://chaotic-mess.github.io/My-Code-/Personal%20Projects/PathfindingSystem/index.html)
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

## AI Projects — *Ares • Odin • Nova*  
![Highlighted Item](https://img.shields.io/badge/HIGHLIGHTED%20ITEM-FFD700?style=for-the-badge&logo=starship&logoColor=black)

Three different takes on personal AI — from a **homegrown LLM built with only the Python standard library** to a **voice assistant** and a **desktop companion**.

### tl;dr

- **ARES** — a from-scratch, stdlib-only character LLM: trains, checkpoints, and serves a web chat.  
- **ODIN** — a voice assistant pipeline (wake word → STT → LLM → TTS → actions).  
- **NOVA** — an Electron desktop companion with voice/overlay, memory, and hooks for vision.

---

### [ARES (homegrown LLM)](https://chaotic-mess.github.io/My-Code-/Personal%20Projects/AI/ARES_AI/V3/static/GITHUBONLY_Index.html)
A character-level RNN language model implemented end-to-end with only Python's standard library. Trains on Shakespeare (or any text) and serves a tiny chat web UI.

**Highlights**
- Pure stdlib: **no NumPy, no PyTorch, no Flask**
- Checkpoints (atomic JSON) + auto-resume
- ETA & throughput, live preview samples, progress panel
- Temperature & top-k decoding

---

### ODIN (voice assistant)

A local voice assistant pipeline — wake word → STT → LLM → TTS → desktop actions.

**Stack**
- **STT**: local engine (e.g., Vosk)
- **LLM**: local runtime (e.g., Ollama)
- **TTS**: local speech synthesis (e.g., pyttsx3)
- Optional integrations: calendars, system control, etc.

> Not publicly available — contains private API keys.

---

### NOVA (desktop companion)

A desktop overlay companion with voice, memory, and optional vision hooks.

**Stack**
- Electron overlay UI (Node.js)
- Local STT/TTS + optional local LLMs
- Persistent memory (JSON/DB)

> Not publicly available — contains private API keys.  
> Really good at GeoGuessr!

---

### Comparison

| Aspect       | **ARES** (Homegrown LLM)         | **ODIN** (Voice Assistant)                   | **NOVA** (Desktop Companion)                |
| ------------ | -------------------------------- | -------------------------------------------- | ------------------------------------------- |
| Core model   | Pure Python **char-RNN**         | External/local LLM via a runtime             | External/local models                       |
| Dependencies | **Stdlib only**                  | Python libs for STT/TTS; local LLM runtime   | Node/Electron + optional Python backends    |
| Interface    | Web chat (tiny HTTP server)      | Voice: wake word → STT → LLM → TTS → actions | Desktop overlay + voice/memory/vision hooks |
| Offline      | Fully offline                    | If models local                              | If models local                             |
| Checkpoints  | Atomic JSON + auto-resume        | n/a (LLM external)                           | n/a (LLM external)                          |
| Best for     | Portfolio core: "I built an LLM" | Hands-free assistant and integrations        | Companion UX with presence and memory       |

---

**[← Back to Main Portfolio](../README.md)**
