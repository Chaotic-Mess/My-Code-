# V2/src/sanity_generate.py
import torch
from src.infer import load_checkpoint, generate
from src.tokenizer_util import AresTokenizer

m, tok = load_checkpoint(device="cpu")
prompt = "<|system|> You are ARES.\n<|user|> Say hi.\n<|assistant|> "
ids = [tok.bos] + tok.encode_ids(prompt, add_special=False)
x = torch.tensor([ids], dtype=torch.long)

ban_ids = [tok.pad, tok.bos, tok.system, tok.user, tok.assistant]
y = generate(m, x, 64, 0.8, 40, top_p=0.95, stop_id=tok.eos, ban_ids=ban_ids, min_tokens=8)[0].tolist()
print("---\n", tok.decode(y[len(ids):]))
