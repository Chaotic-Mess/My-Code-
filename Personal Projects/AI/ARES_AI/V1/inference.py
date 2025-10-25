# inference.py

import os
import torch

from tokenizers import ByteLevelBPETokenizer
from model import GPTConfig, GPT
from config import DEVICE, TEMPERATURE, TOP_K, MAX_GEN_LENGTH

@torch.no_grad()
def generate(
    model, 
    tokenizer, 
    device, 
    prompt: str, 
    max_length: int = 100, 
    temperature: float = 1.0, 
    top_k: int = 50
):
    model.eval()
    # encode prompt
    enc = tokenizer.encode(prompt)
    idx = torch.tensor(enc.ids, dtype=torch.long, device=device)[None, ...]  # (1, T)
    generated = idx

    for _ in range(max_length):
        # only keep the last block_size tokens as input
        input_ids = generated if generated.size(1) <= model.block_size \
                    else generated[:, -model.block_size:]
        logits = model(input_ids)           # (1, L, vocab_size)
        logits = logits[:, -1, :] / temperature  # Pick only the last token’s logits, now shape (1, vocab_size)
        # top-k filtering
        v, _ = torch.topk(logits, top_k)
        min_logits = v[:, -1].unsqueeze(1)
        logits[logits < min_logits] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)  # (1, vocab)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        generated = torch.cat((generated, next_id), dim=1)  # append

    # decode and return
    out = tokenizer.decode(generated[0].tolist())
    return out

def main():
    # ————— Config —————
    tokenizer_dir = "tokenizer"
    ckpt_path     = os.path.join("checkpoints", "gpt_model.pt")
    block_size    = 128
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ——————————————————

    # load tokenizer & model
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt")
    )
    vocab_size = tokenizer.get_vocab_size()
    config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # your prompt here
    prompt = "Once upon a time, in the Helios network,"
    output = generate(
        model, tokenizer, DEVICE,
        prompt="Once upon a time…",
        max_length=MAX_GEN_LENGTH,
        temperature=TEMPERATURE,
        top_k=TOP_K
    )
    print("\n=== GENERATED TEXT ===\n")
    print(output)

if __name__ == "__main__":
    main()
