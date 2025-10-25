# src/infer.py
import torch
from src.model import TinyGPT
from src.tokenizer_util import AresTokenizer

def _apply_bans(logits, ban_ids):
    if ban_ids:
        for bid in ban_ids:
            if bid is not None:
                logits[:, bid] = -float("inf")

@torch.no_grad()
def generate(
    model, idx, max_new_tokens=128, temperature=0.8,
    top_k=40, top_p=0.95, stop_id=None, ban_ids=None,
    min_tokens=8, ws_ban_ids=None
):
    for t in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-5, temperature)

        # Always ban specials
        _apply_bans(logits, ban_ids)

        # Temporarily ban whitespace-only tokens at the start
        if ws_ban_ids and t < min_tokens:
            _apply_bans(logits, ws_ban_ids)

        # Soft-ban EOS until we've emitted a few tokens
        if stop_id is not None and t < min_tokens:
            logits[:, stop_id] = -float("inf")

        if top_k and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            mask = cum > top_p
            mask[:, 0] = False
            sorted_logits[mask] = -float("inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if stop_id is not None and t >= min_tokens and int(next_id[0, 0]) == stop_id:
            break

        idx = torch.cat([idx, next_id], dim=1)
    return idx

def load_checkpoint(path="checkpoints/best_sft.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["cfg"]
    m = TinyGPT(**cfg).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    tok = AresTokenizer(ckpt["tok_path"])
    return m, tok
