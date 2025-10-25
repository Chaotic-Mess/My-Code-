# evaluate.py

import os
import math
import torch

from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from config import (
    RAW_TEXT_FILE,
    TOKENIZER_DIR,
    CHECKPOINT_PATH,
    BLOCK_SIZE,
    BATCH_SIZE,
    DEVICE
)
from model import GPTConfig, GPT
from train import TextDataset  # re-use the same Dataset class

def evaluate():
    # ——— Load tokenizer & dataset ———
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(TOKENIZER_DIR, "vocab.json"),
        os.path.join(TOKENIZER_DIR, "merges.txt")
    )
    vocab_size = tokenizer.get_vocab_size()
    dataset = TextDataset(RAW_TEXT_FILE, tokenizer, BLOCK_SIZE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # ——— Load model from checkpoint ———
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE  # plus other defaults from config.py
    )
    model = GPT(config).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # ——— Compute loss & perplexity ———
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, targets=y)
            total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(dataset)
    ppl = math.exp(avg_loss)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity:      {ppl:.2f}")

if __name__ == "__main__":
    evaluate()
