import os, random, numpy as np, torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
