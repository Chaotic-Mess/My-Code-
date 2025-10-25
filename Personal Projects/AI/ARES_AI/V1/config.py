# config.py

import os
import torch

# ——— Paths ———
DATA_DIR           = ""
RAW_TEXT_FILE      = os.path.join(DATA_DIR, "all_texts.txt")
TOKENIZER_DIR      = "tokenizer"
CHECKPOINT_DIR     = "checkpoints"
CHECKPOINT_PATH    = os.path.join(CHECKPOINT_DIR, "gpt_model.pt")

# ——— Model hyperparameters ———
VOCAB_SIZE         = None   # will be filled in after loading tokenizer
BLOCK_SIZE         = 128
N_LAYER            = 8
N_HEAD             = 8
N_EMBD             = 512

# ——— Training hyperparameters ———
BATCH_SIZE         = 32
EPOCHS             = 10
LEARNING_RATE      = 1e-4

# ——— Inference settings ———
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURE        = 0.5
TOP_K              = 20
MAX_GEN_LENGTH     = 200

# Utility to ensure directories exist
os.makedirs(TOKENIZER_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
