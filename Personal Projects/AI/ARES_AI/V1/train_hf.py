# train_hf.py

from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments

# 1. Load the pretrained tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 2. Load and preprocess your dialogue dataset
dataset = load_dataset(
    "text", 
    data_files={"train": "data/dialogue_lines.txt"}, 
    split="train"
)

def tokenize_function(examples):
    # Tokenize and group into blocks
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"]
)

# 3. Configure training arguments
training_args = TrainingArguments(
    output_dir="hf_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="hf_logs",
    logging_steps=100,
)

# 4. Initialize Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
trainer.save_model("hf_checkpoints")

print("✅ Fine-tuning complete. Model saved to 'hf_checkpoints' directory.")
