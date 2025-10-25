# Train a byte-level BPE tokenizer on data/processed/corpus.txt
# Run: python -m src.tokenizer_train
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

DATA = "data/processed/corpus.txt"
OUTDIR = "tokenizer"

def main():
    if not os.path.exists(DATA):
        raise SystemExit(f"Missing {DATA}. Run prepare_data first.")
    os.makedirs(OUTDIR, exist_ok=True)
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=2048,           # smaller is better for small datasets
        min_frequency=1,
        special_tokens=["[PAD]","[UNK]","<|bos|>","<|eos|>","<|system|>","<|user|>","<|assistant|>"]
    )
    tok.train([DATA], trainer)
    tok.save(os.path.join(OUTDIR,"tokenizer.json"))
    print("wrote", os.path.join(OUTDIR,"tokenizer.json"))

if __name__ == "__main__":
    main()
