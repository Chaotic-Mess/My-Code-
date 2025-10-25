# data_preprocessing.py

import os
import glob
import re

from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

def load_texts(data_dir):
    #Load all .txt files from data_dir into a list of raw strings.
    texts = []
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts

def clean_text(text):
    #Simple cleaning: lowercase, collapse whitespace, strip.
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def save_texts(texts, output_path):
    #Write each cleaned text as a new line in output_path.
    with open(output_path, "w", encoding="utf-8") as f:
        for txt in texts:
            f.write(txt + "\n")

if __name__ == "__main__":
    # ————— Configurable parameters —————
    data_dir               = "data/"            # folder containing your .txt files
    concatenated_file      = "all_texts.txt"    # intermediate file
    vocab_size             = 50_000
    min_freq               = 2
    special_tokens         = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    output_tokenizer_dir   = "tokenizer/"       # where to save tokenizer files
    # ————————————————————————————————————

    # 1️⃣ Load & clean
    raw_texts     = load_texts(data_dir)
    cleaned_texts = [clean_text(t) for t in tqdm(raw_texts, desc="Cleaning texts")]
    save_texts(cleaned_texts, concatenated_file)

    # 2️⃣ Train a Byte-Level BPE tokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[concatenated_file],
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens
    )

    # 3️⃣ Save the trained tokenizer
    os.makedirs(output_tokenizer_dir, exist_ok=True)
    tokenizer.save_model(output_tokenizer_dir)

    print(f"✅ Tokenizer trained and saved to '{output_tokenizer_dir}'")
