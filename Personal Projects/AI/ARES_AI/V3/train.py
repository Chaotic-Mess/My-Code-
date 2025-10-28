# train.py — ARES_V3.5 RNN trainer with time/ETA, history, checkpoints (stdlib-only)
import os, random, time, json
from model.tokenizer import CharTokenizer
from model.model import TinyCharRNN

# -------------------------
# Config
# -------------------------
BLOCK_LEN      = 128
TOTAL_STEPS    = 10000
SAMPLE_EVERY   = 1000
SAVE_EVERY     = 1000
PREVIEW_TEMP   = 0.8
PREVIEW_TOPK   = 50
BASE_LR        = 0.03

WEIGHTS = os.path.join("weights", "model.json")
CKPT    = os.path.join("weights", "ckpt.json")

# -------------------------
# Helpers
# -------------------------
def CurrentTime(): return time.strftime("%Y-%m-%d %H:%M:%S")

def _fmt_secs(s):
    s = int(max(0, s)); h=s//3600; m=(s%3600)//60; ss=s%60
    return f"{h}h {m}m {ss}s" if h else f"{m}m {ss}s" if m else f"{ss}s"

def _safe_json(path, data):
    tmp = path + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(data,f)
    os.replace(tmp,path)

def _safe_text(path, text):
    tmp = path + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f: f.write(text)
    os.replace(tmp,path)

def write_history(step, loss, sample):
    rec = f"[{CurrentTime()}] [step {step}] loss={loss:.3f}\n--- sample ---\n{sample}\n-------------\n"
    _safe_text("progress_latest.txt", rec)
    with open("progress_history.txt","a",encoding="utf-8") as f: f.write(rec)

# -------------------------
# Checkpoint helpers
# -------------------------
def save_ckpt(model, step):
    os.makedirs("weights", exist_ok=True)
    step_path = os.path.join("weights", f"model_step_{step}.json")
    model.save(step_path)
    _safe_json(CKPT, {"path": step_path, "step": step})
    model.save(WEIGHTS)

def load_ckpt(model):
    if os.path.exists(CKPT):
        try:
            meta = json.load(open(CKPT,"r",encoding="utf-8"))
            path, step = meta.get("path"), int(meta.get("step",0))
            if path and os.path.exists(path):
                model = TinyCharRNN.load(path)
                print(f"[resume] {path} @ step {step}")
                return model, step+1
        except Exception as e:
            print("[resume] failed:", e)
    # fallback
    files = [f for f in os.listdir("weights") if f.startswith("model_step_")]
    best = max((int(f.split("_")[2].split(".")[0]), f) for f in files) if files else (0,None)
    if best[1]:
        model = TinyCharRNN.load(os.path.join("weights",best[1]))
        return model, best[0]+1
    return model,1

# -------------------------
# Dataset merge
# -------------------------
texts=[]
for n in os.listdir("data"):
    if n.endswith(".txt"):
        p=os.path.join("data",n)
        with open(p,encoding="utf-8") as f: t=f.read().strip()
        texts.append(t)
        print(f"Loaded {n}: {len(t)} chars")
text="\n\n".join(texts)
print(f"Total dataset size: {len(text)} chars")

tok=CharTokenizer(text)
ids=tok.encode(text)
json.dump({"stoi":tok.stoi,"itos":tok.itos},
          open("weights/tokenizer.json","w",encoding="utf-8"))

model=TinyCharRNN(len(tok.stoi),hidden=256,lr=BASE_LR)
model,start_step=load_ckpt(model)

# -------------------------
# Warmup speed estimate
# -------------------------
def warmup(model, ids, n=300):
    t0=time.time()
    for _ in range(n):
        s=random.randint(0,len(ids)-BLOCK_LEN-2)
        x=ids[s:s+BLOCK_LEN]; y=ids[s+1:s+BLOCK_LEN+1]
        model.train_step(x,y)
    dt=time.time()-t0
    return n/max(1e-9,dt)

os.makedirs("weights",exist_ok=True)
tmp="weights/_tmp.json"
model.save(tmp)
sps=warmup(model,ids)
model=TinyCharRNN.load(tmp); os.remove(tmp)
print(f"[throughput] ~{sps:.2f} steps/sec  | ETA≈{int((TOTAL_STEPS/sps)//60)} min")

# -------------------------
# Train loop
# -------------------------
start=time.time(); last=start_step-1
try:
    for step in range(start_step,TOTAL_STEPS+1):
        t0=time.time(); last=step
        model.lr=BASE_LR*(0.5**(step//10000))
        s=random.randint(0,len(ids)-BLOCK_LEN-2)
        x=ids[s:s+BLOCK_LEN]; y=ids[s+1:s+BLOCK_LEN+1]
        loss=model.train_step(x,y)

        if step%SAMPLE_EVERY==0 or step==start_step:
            preview=model.generate(tok,"ROMEO:\n",200,PREVIEW_TEMP,PREVIEW_TOPK)
            elapsed=time.time()-start
            eta=(TOTAL_STEPS-step)*(time.time()-t0)
            print(f"[step {step}] loss={loss:.3f} | elapsed={_fmt_secs(elapsed)}")
            write_history(step,loss,preview)

        if step%SAVE_EVERY==0 or step==TOTAL_STEPS:
            save_ckpt(model,step)

except KeyboardInterrupt:
    save_ckpt(model,last)
    print(f"[interrupt] saved step {last}")

save_ckpt(model,TOTAL_STEPS)
print(f"[done] {CurrentTime()} | total elapsed={_fmt_secs(time.time()-start)}")
