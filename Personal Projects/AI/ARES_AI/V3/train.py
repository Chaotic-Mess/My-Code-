# train.py — homegrown RNN trainer with time/ETA, history, checkpoints (stdlib-only)
import os, random, time, json
from model.tokenizer import CharTokenizer
from model.model import TinyCharRNN  # model.save() is already atomic in your updated model.py

DATA    = os.path.join("data", "tiny_shakespeare.txt")
WEIGHTS = os.path.join("weights", "model.json")
CKPT    = os.path.join("weights", "ckpt.json")

# -------------------------
# Config (tweak freely)
# -------------------------
BLOCK_LEN      = 128          # BPTT length (context window)
TOTAL_STEPS    = 20000        # total update steps
SAMPLE_EVERY   = 1000         # preview cadence (higher = less overhead)
SAVE_EVERY     = 1000         # checkpoint cadence
PREVIEW_TEMP   = 0.8          # sampling temperature for previews (0.6..0.9)
PREVIEW_TOPK   = 50           # top-k cutoff for previews (None to disable)
BASE_LR        = 0.03         # base learning rate (decays during run)

# -------------------------
# Time helpers
# -------------------------
TRAIN_START_TS = time.time()  # Start wallclock
EMA_STEP_SEC   = None         # (kept for API completeness; we use local ema_step below)

def CurrentTime():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _fmt_secs(s):
    s = int(max(0, s))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h: return f"{h}h {m}m {ss}s"
    if m: return f"{m}m {ss}s"
    return f"{ss}s"

def TotalCompletionActual():
    return _fmt_secs(time.time() - TRAIN_START_TS)

# -------------------------
# Atomic writers (protect against mid-write Ctrl+C)
# -------------------------
def _safe_save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)

def _safe_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

# -------------------------
# History logger
# -------------------------
def write_history(step, loss, sample_text):
    rec = (
        f"[{CurrentTime()}] [step {step}] loss={loss:.3f}\n"
        f"--- sample ---\n{sample_text}\n-------------\n"
    )
    # Overwrite latest snapshot (atomic)
    _safe_write_text("progress_latest.txt", rec)
    # Append full history (single append call = low corruption risk)
    with open("progress_history.txt", "a", encoding="utf-8") as f:
        f.write(rec)

# -------------------------
# Checkpoint helpers
# -------------------------
def _ckpt_pointer_write(path, step):
    _safe_save_json(CKPT, {"path": path, "step": step})

def _find_latest_stepfile():
    """If ckpt.json is missing/broken, find the highest model_step_*.json."""
    try:
        files = os.listdir("weights")
    except FileNotFoundError:
        return None, 0
    best_path, best_step = None, 0
    for name in files:
        if name.startswith("model_step_") and name.endswith(".json"):
            mid = name[len("model_step_"):-len(".json")]
            if mid.isdigit():
                s = int(mid)
                if s > best_step:
                    best_step = s
                    best_path = os.path.join("weights", name)
    return best_path, best_step

def save_ckpt(model, step):
    """
    Saves:
      - weights/model_step_{step}.json  (atomic via model.save)
      - weights/model.json              (latest copy, atomic)
      - weights/ckpt.json               (atomic pointer)
    """
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    step_path = os.path.join("weights", f"model_step_{step}.json")
    model.save(step_path)      # atomic inside model.save
    _ckpt_pointer_write(step_path, step)  # atomic pointer
    model.save(WEIGHTS)        # atomic latest copy

def load_ckpt_if_any(model):
    """
    Tries ckpt.json → falls back to latest model_step_*.json → else step=1.
    """
    # Prefer ckpt.json
    if os.path.exists(CKPT):
        try:
            meta = json.load(open(CKPT, "r", encoding="utf-8"))
            path = meta.get("path")
            step = int(meta.get("step", 0))
            if path and os.path.exists(path):
                model = TinyCharRNN.load(path)
                print(f"[resume] {path} @ step {step}")
                return model, step + 1
        except Exception as e:
            print("[resume] ckpt.json failed:", e)

    # Fallback to scanning dir
    latest_path, latest_step = _find_latest_stepfile()
    if latest_path and os.path.exists(latest_path):
        try:
            model = TinyCharRNN.load(latest_path)
            print(f"[resume] {latest_path} @ step {latest_step}")
            return model, latest_step + 1
        except Exception as e:
            print("[resume] fallback load failed:", e)

    # Fresh start
    return model, 1

# -------------------------
# Pre-run warmup ETA (does NOT change final weights)
# -------------------------
def estimate_steps_per_sec(model, ids, block_len=128, warmup_steps=300):
    t0 = time.time()
    for _ in range(warmup_steps):
        s = random.randint(0, len(ids) - block_len - 2)
        x = ids[s : s + block_len]
        y = ids[s + 1 : s + block_len + 1]
        _ = model.train_step(x, y)  # do real work to warm caches
    dt = time.time() - t0
    return warmup_steps / max(1e-9, dt)

# -------------------------
# Data + model init
# -------------------------
with open(DATA, encoding="utf-8") as f:
    text = f.read()

tok = CharTokenizer(text)
ids = tok.encode(text)

model = TinyCharRNN(vocab_size=len(tok.stoi), hidden=128, lr=BASE_LR)
model, start_step = load_ckpt_if_any(model)

# snapshot weights -> warmup -> restore (so warmup doesn't affect real training)
os.makedirs("weights", exist_ok=True)
_tmp = os.path.join("weights", "_tmp_warmup.json")
model.save(_tmp)  # atomic
sps = estimate_steps_per_sec(model, ids, block_len=BLOCK_LEN, warmup_steps=300)
model = TinyCharRNN.load(_tmp)
try: os.remove(_tmp)
except OSError: pass

remaining_steps = TOTAL_STEPS - start_step + 1
print(f"[throughput] ~{sps:.2f} steps/sec  |  rough ETA ≈ {int(remaining_steps / max(1e-9, sps) // 60)} min")
run_started_at = CurrentTime()
print(f"[start] {run_started_at} | steps={TOTAL_STEPS} | block={BLOCK_LEN} | base_lr={BASE_LR}")

# -------------------------
# Training loop  
# -------------------------
last_step = start_step - 1
ema_step = None

try:
    for step in range(start_step, TOTAL_STEPS + 1):
        t_step = time.time()
        last_step = step

        # gentle LR decay every 10k steps
        model.lr = BASE_LR * (0.5 ** (step // 10000))

        # sample a slice
        start = random.randint(0, len(ids) - BLOCK_LEN - 2)
        x = ids[start : start + BLOCK_LEN]
        y = ids[start + 1 : start + BLOCK_LEN + 1]

        loss = model.train_step(x, y)

        # timing / ETA
        dt = time.time() - t_step
        ema_step = dt if ema_step is None else (0.98 * ema_step + 0.02 * dt)
        elapsed = time.time() - TRAIN_START_TS
        eta = (TOTAL_STEPS - step) * (ema_step if ema_step is not None else 0.0)

        # preview (less frequent to reduce overhead)
        if step % SAMPLE_EVERY == 0 or step == start_step:
            preview = model.generate(
                tok, seed="ROMEO:\n", max_new=200,
                temperature=PREVIEW_TEMP, top_k=PREVIEW_TOPK
            )
            print(f"[step {step}] loss={loss:.3f} | ETA≈{_fmt_secs(eta)} | elapsed={_fmt_secs(elapsed)}")
            write_history(step, loss, preview)

        # checkpoint
        if step % SAVE_EVERY == 0 or step == TOTAL_STEPS:
            save_ckpt(model, step)

except KeyboardInterrupt:
    try:
        save_ckpt(model, last_step)
        print(f"\n[interrupt] saved checkpoint at step {last_step}")
    except KeyboardInterrupt:
        # If a second Ctrl+C hits during save, old files remain intact thanks to atomic writes
        print("\n[interrupt] second interrupt during save — previous checkpoint remains safe.")

# final persist
save_ckpt(model, TOTAL_STEPS)
print(f"[done] {CurrentTime()} | total elapsed={TotalCompletionActual()}")
