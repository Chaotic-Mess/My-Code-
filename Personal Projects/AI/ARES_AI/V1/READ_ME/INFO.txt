This model is kinda dumb rn. Not enough data for it to learn, so it has robotic spasms n shit. I’m slapping this note here on **2025-07-22** so my future self doesn’t have to scratch their head.

---

## 🛠 Prerequisites

* **Python** 3.10+ (tested on 3.12)
* **pip** package manager
* **Hardware**: GPU (CUDA) recommended, CPU-only works for tiny models

## 📦 Dependencies

Install everything in one go:

```bash
pip install torch tokenizers tqdm transformers datasets tf-keras
```

> * `torch`: model training & inference
> * `tokenizers`, `tqdm`: data preprocessing & progress bars
> * `transformers`, `datasets`: Hugging Face GPT-2 fine-tuning
> * `tf-keras`: required by `transformers` on Windows

## 📁 Project Structure

```
ARES_AI/
├── config.py           # central hyperparameters & paths
├── data/
│   ├── TestData.txt    # your original raw text
│   ├── all_texts.txt   # concatenated & cleaned text
│   └── dialogue_lines.txt  # dialogue pairs for chat fine-tune
├── tokenizer/          # BPE tokenizer files
├── checkpoints/        # from-scratch model weights
├── hf_checkpoints/     # GPT-2 fine-tuned weights
├── hf_logs/            # Hugging Face training logs
├── data_preprocessing.py  # builds `all_texts.txt` + trains BPE
├── model.py            # GPT-from-scratch model definition
├── train.py            # train-from-scratch script
├── evaluate.py         # evaluate loss & perplexity
├── inference.py        # generate text script
├── chat.py             # interactive chat loop
└── train_hf.py         # fine-tune GPT-2 on dialogue data
```

## ⚙️ How It Functions

1. **From-scratch GPT**:

   * `data_preprocessing.py` cleans raw `.txt` files, concatenates into `all_texts.txt`, and trains a byte-level BPE tokenizer.
   * `model.py` defines a small Transformer (8 layers, 8 heads, 512‑dim) using PyTorch.
   * `train.py` builds a dataset, trains the model, and saves weights to `checkpoints/gpt_model.pt`.
   * `evaluate.py` computes validation loss & perplexity on held-out data.
   * `inference.py` does autoregressive sampling (temperature + top‑k) to generate text.
   * `chat.py` wraps `inference.py` in a simple user–AI loop with sliding context.

2. **GPT-2 Fine-Tuning**:

   * `train_hf.py` uses Hugging Face’s `Trainer` API to load pre-trained `gpt2`, tokenize `dialogue_lines.txt`, and fine-tune for 5 epochs.
   * Outputs saved in `hf_checkpoints/`, ready for inference or chat via the same `inference.py` or modified `chat.py`.

## ▶️ Usage Commands

```bash
# 1) Preprocess text & train BPE
python data_preprocessing.py

# 2) Train from scratch
python train.py

# 3) Evaluate model
python evaluate.py

# 4) Generate sample
python inference.py

# 5) Chat interactively
python chat.py

# 6) Fine-tune GPT-2 on dialogue
python train_hf.py
```

## 🚀 Capabilities & Limitations

* **Capabilities**:

  * Generates story snippets, simple dialogue, code stubs, etc.
  * Fine-tuned chat responds in a basic conversational style.

* **Limitations**:

  * Very small dataset → low-quality output until more data is added.
  * Context window limited by `BLOCK_SIZE` (default 128 tokens).
  * No long-term memory or external knowledge beyond training data.

## 🔮 Future Improvements

* Increase dataset size (books, scripts, dialogues)
* Scale model up (more layers/embeddings)
* Add retrieval-augmented generation (RAG)
* Implement RLHF for style tuning

---

*Note written on 2025-07-22. Good luck, future me!*
