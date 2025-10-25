# Minimal eval: PPL over corpus + a tiny arithmetic EM
import os, math, json, random, torch, numpy as np
from src.model import TinyGPT
from src.tokenizer_util import AresTokenizer

def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device)
    m = TinyGPT(**ckpt["cfg"]).to(device); m.load_state_dict(ckpt["model"]); m.eval()
    tok = AresTokenizer(ckpt["tok_path"]); return m,tok

@torch.no_grad()
def perplexity(model, tok, text, block=256):
    ids = tok.encode_ids(text, add_special=False)
    if len(ids) <= block+1: return float("nan")
    xs = torch.tensor([ids[:-1][:block]], dtype=torch.long)
    ys = torch.tensor([ids[1:][:block]], dtype=torch.long)
    logits, loss = model(xs, ys)
    return float(torch.exp(loss))

def arithmetic_em(model, tok, n=20, device="cpu"):
    correct = 0
    for _ in range(n):
        a,b = random.randint(10,99), random.randint(10,99)
        prompt = f"<|system|> You are ARES.\n<|user|> What is {a} + {b}?\n<|assistant|> "
        ids = tok.encode_ids(prompt); x = torch.tensor([ids], dtype=torch.long, device=device)
        from src.infer import generate
        y = generate(model, x, 16, 0.7, 20)[0].tolist()
        out = tok.decode(y[len(ids):]).strip().split()[0]
        try:
            if int(out) == a+b: correct += 1
        except: pass
    return correct, n

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "checkpoints/best_sft.pt"
    if not os.path.exists(ckpt): ckpt = "checkpoints/best.pt"
    if not os.path.exists(ckpt): raise SystemExit("No checkpoint found.")
    model, tok = load_ckpt(ckpt, device)
    txt = ""
    cpath = "data/processed/corpus.txt"
    if os.path.exists(cpath):
        with open(cpath,"r",encoding="utf-8") as f: txt = f.read()
    ppl = perplexity(model, tok, txt) if txt else float("nan")
    correct, n = arithmetic_em(model, tok, 20, device)
    print(f"PPL (corpus): {ppl:.3f}")
    print(f"Arithmetic EM: {correct}/{n}")

if __name__ == "__main__":
    main()
