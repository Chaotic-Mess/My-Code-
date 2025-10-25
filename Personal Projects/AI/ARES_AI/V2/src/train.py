# LM warm-up on corpus.txt (optional)
import os, math, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.config import cfg
from src.model import TinyGPT
from src.tokenizer_util import AresTokenizer

DATA_TXT = "data/processed/corpus.txt"

class LMDataset(Dataset):
    def __init__(self, token_ids, block):
        self.ids, self.block = token_ids, block
    def __len__(self): return max(0, len(self.ids) - self.block - 1)
    def __getitem__(self, i):
        x = self.ids[i:i+self.block]
        y = self.ids[i+1:i+self.block+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def load_tokens(tok, path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    ids = tok.encode_ids(txt, add_special=False)
    return np.array(ids, dtype=np.int64)

def main():
    device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if not os.path.exists(DATA_TXT):
        raise SystemExit("Run prepare_data first to create corpus.txt")
    tok = AresTokenizer()
    token_ids = load_tokens(tok, DATA_TXT)
    if len(token_ids) < cfg.block_size + 10:
        raise SystemExit("corpus is too small; add more raw text to data/raw")

    n = int(0.95 * len(token_ids))
    train_ids, val_ids = token_ids[:n], token_ids[n:]

    train_dl = DataLoader(LMDataset(train_ids, cfg.block_size), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(LMDataset(val_ids,   cfg.block_size), batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    model = TinyGPT(vocab_size=tok.vocab_size, block_size=cfg.block_size,
                    n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                    dropout=cfg.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best = math.inf; os.makedirs(cfg.ckpt_dir, exist_ok=True)

    step = 0
    while step < cfg.max_steps:
        model.train()
        for x,y in train_dl:
            step += 1
            x,y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0: print(f"step {step} | train loss {loss.item():.3f}")

            if step % cfg.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    losses = []
                    for vx,vy in val_dl:
                        vx,vy = vx.to(device), vy.to(device)
                        _, vloss = model(vx, vy)
                        losses.append(vloss.item())
                    v = sum(losses)/len(losses)
                print(f"eval @ {step} | val loss {v:.3f}")
                if v < best:
                    best = v
                    torch.save({
                        "model": model.state_dict(),
                        "tok_path": "tokenizer/tokenizer.json",
                        "cfg": dict(vocab_size=tok.vocab_size, block_size=cfg.block_size,
                                    n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout)
                    }, os.path.join(cfg.ckpt_dir, "best.pt"))
                if step >= cfg.max_steps: break
        if step >= cfg.max_steps: break

if __name__ == "__main__":
    main()
