# train.py

import os
import torch

from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from model import GPTConfig, GPT
from config import RAW_TEXT_FILE, TOKENIZER_DIR, CHECKPOINT_PATH
from config import BLOCK_SIZE, BATCH_SIZE, LEARNING_RATE, DEVICE

class TextDataset(Dataset):
    """
    Reads the concatenated text file, tokenizes it, and builds
    (input, target) pairs of length block_size for causal language modeling.
    """
    def __init__(self, file_path: str, tokenizer: ByteLevelBPETokenizer, block_size: int):
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        # encode entire corpus to token IDs
        all_ids = tokenizer.encode(data).ids
        self.block_size = block_size
        self.examples = []
        # chop into block_size+1 chunks, create inputs & targets
        for i in range(0, len(all_ids) - block_size, block_size):
            chunk = all_ids[i : i + block_size + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:],   dtype=torch.long)
            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def train():
    # Load tokenizer
    tokenizer = ByteLevelBPETokenizer(
        f"{TOKENIZER_DIR}/vocab.json",
        f"{TOKENIZER_DIR}/merges.txt"
    )
    vocab_size = tokenizer.get_vocab_size()

    # Prepare dataset and dataloader
    dataset = TextDataset(RAW_TEXT_FILE, tokenizer, BLOCK_SIZE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # Instantiate model
    config = GPTConfig(vocab_size=vocab_size)
    model  = GPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop …
    # At the end, save:
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"✅ Model saved to {CHECKPOINT_PATH}")
    # ————— Configuration —————
    data_file        = "all_texts.txt"        # output from data_preprocessing.py
    tokenizer_dir    = "tokenizer"            # where vocab.json & merges.txt live
    block_size       = 128                    # sequence length
    batch_size       = 32
    epochs           = 3
    lr               = 3e-4
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ————————————————————————

    # Load tokenizer
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt")
    )
    vocab_size = tokenizer.get_vocab_size()

    # Prepare dataset + dataloader
    dataset = TextDataset(data_file, tokenizer, block_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Instantiate model
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=8,    # feel free to adjust
        n_head=8,
        n_embd=512
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0 or batch_idx == len(loader):
                avg = total_loss / batch_idx
                print(f"Epoch {epoch}/{epochs} — Batch {batch_idx}/{len(loader)} — Loss: {loss.item():.4f} — Avg: {avg:.4f}")

    # Save checkpoint
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "gpt_model.pt")
    torch.save(model.state_dict(), path)
    print(f"✅ Training complete. Model weights saved to '{path}'")

if __name__ == "__main__":
    train()
