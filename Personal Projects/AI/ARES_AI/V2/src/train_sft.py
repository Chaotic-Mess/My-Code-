# V2/src/train_sft.py
import os, math, json, time, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.model import TinyGPT
from src.tokenizer_util import AresTokenizer
from src.config import cfg

INSTRUCTIONS = "data/processed/instructions.jsonl"

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from torch.amp import autocast, GradScaler

# ---------------- Dataset ----------------
class SFTDataset(Dataset):
    def __init__(self, path, tok, block_size):
        self.tok, self.block = tok, block_size
        self.data = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run ingest_dialogue / make_instructions first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                prompt_ids = tok.encode_ids(j["prompt"], add_special=False)
                resp_ids   = tok.encode_ids(j["response"], add_special=False) + [tok.eos]
                x = prompt_ids + resp_ids
                y = [-100]*len(prompt_ids) + resp_ids[:]
                if len(x) > block_size:
                    x = x[-block_size:]; y = y[-block_size:]
                self.data.append((x, y))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

# ---------------- Collate (top-level; Windows-safe) ----------------
PAD_ID = 0
BLOCK_SIZE = 256
def collate(batch):
    cur_max = min(max(len(x) for x,_ in batch), BLOCK_SIZE)
    X, Y = [], []
    for x, y in batch:
        x = x[-cur_max:]; y = y[-cur_max:]
        X.append(x + [PAD_ID]*(cur_max - len(x)))
        Y.append(y + [-100]   *(cur_max - len(y)))
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

@torch.no_grad()
def evaluate(model, dl, device, use_amp):
    model.eval()
    losses = []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-100)
        losses.append(loss.item())
    return sum(losses)/len(losses) if losses else math.inf

def main():
    device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    use_amp = (device == "cuda")
    tok = AresTokenizer()

    global PAD_ID, BLOCK_SIZE
    PAD_ID = tok.pad
    BLOCK_SIZE = cfg.block_size

    ds = SFTDataset(INSTRUCTIONS, tok, cfg.block_size)
    if len(ds) < 2:
        raise SystemExit("Need at least 2 instruction pairs in instructions.jsonl.")

    # 5% val split (min 1)
    val_size = max(1, len(ds)//20)
    train_size = len(ds) - val_size
    gen = torch.Generator().manual_seed(getattr(cfg, "seed", 1337))
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=gen)

    is_windows = (os.name == "nt")
    num_workers = 0  # safest on Windows
    pin = (device == "cuda")

    # Auto batch size so we always have batches
    cfg_bs = int(getattr(cfg, "batch_size", 8))
    bs  = max(1, min(cfg_bs, len(train_ds)))
    vbs = max(1, min(bs,     len(val_ds)))

    print(f"[info] device={device} amp={use_amp} seed={getattr(cfg, 'seed', 1337)}")
    print(f"[info] pairs: train={len(train_ds)} | val={len(val_ds)} | batch_size={bs}")

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False,
                          collate_fn=collate, num_workers=num_workers, pin_memory=pin, persistent_workers=False)
    val_dl   = DataLoader(val_ds,   batch_size=vbs, shuffle=False, drop_last=False,
                          collate_fn=collate, num_workers=num_workers, pin_memory=pin, persistent_workers=False)

    # Safety: ensure we actually have batches
    if len(train_dl) == 0:
        raise SystemExit("No training batches. Reduce cfg.batch_size or add more instruction pairs.")

    model = TinyGPT(
        vocab_size=tok.vocab_size, block_size=cfg.block_size,
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
        dropout=cfg.dropout
    ).to(device)

    if getattr(cfg, "compile", False):
        try: model = torch.compile(model)
        except Exception as e: print(f"[warn] torch.compile disabled: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(device="cuda") if use_amp else None
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ------------ One-batch smoke test ------------
    xb, yb = next(iter(train_dl))
    xb = xb.to(device); yb = yb.to(device)
    t0 = time.time()
    if use_amp:
        with autocast(device_type="cuda", enabled=True):
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-100)
        (scaler or GradScaler(device="cuda")).scale(loss).backward()
    else:
        logits, _ = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-100)
        loss.backward()
    print(f"[sanity] 1st batch OK in {time.time()-t0:.2f}s | loss={float(loss):.3f}")
    opt.zero_grad(set_to_none=True)

    # ------------ Train ------------
    best = math.inf
    no_improve = 0
    patience = int(getattr(cfg, "patience", 5))
    grad_accum = max(1, int(getattr(cfg, "grad_accum", 1)))

    steps = 0
    pbar = tqdm(total=cfg.max_steps, desc="sft", dynamic_ncols=True)

    while steps < cfg.max_steps:
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)

            if use_amp:
                with autocast(device_type="cuda", enabled=True):
                    logits, _ = model(xb)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-100) / grad_accum
                scaler.scale(loss).backward()
            else:
                logits, _ = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-100) / grad_accum
                loss.backward()

            if (steps + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

                steps += 1
                pbar.update(1)
                if steps % 50 == 0:
                    pbar.set_postfix_str(f"loss={(loss.item()*grad_accum):.3f}")

                if steps % cfg.eval_every == 0:
                    val = evaluate(model, val_dl, device, use_amp)
                    print(f"\n[eval] steps={steps} val_loss={val:.3f}")
                    if val < best:
                        best = val; no_improve = 0
                        torch.save({
                            "model": model.state_dict(),
                            "tok_path": "tokenizer/tokenizer.json",
                            "cfg": dict(
                                vocab_size=tok.vocab_size, block_size=cfg.block_size,
                                n_layer=cfg.n_layer, n_head=cfg.n_head,
                                n_embd=cfg.n_embd, dropout=cfg.dropout
                            )
                        }, os.path.join(cfg.ckpt_dir, "best_sft.pt"))
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            print(f"Early stopping (no improvement for {patience} evals).")
                            pbar.close()
                            return
                if steps >= cfg.max_steps:
                    pbar.close()
                    return

if __name__ == "__main__":
    main()
