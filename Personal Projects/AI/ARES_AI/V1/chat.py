# chat.py

import torch
from tokenizers import ByteLevelBPETokenizer
from model import GPTConfig, GPT
from config import TOKENIZER_DIR, CHECKPOINT_PATH, BLOCK_SIZE, DEVICE, TEMPERATURE, TOP_K, MAX_GEN_LENGTH

def load_model():
    # load tokenizer
    tokenizer = ByteLevelBPETokenizer(
        f"{TOKENIZER_DIR}/vocab.json",
        f"{TOKENIZER_DIR}/merges.txt"
    )
    vocab_size = tokenizer.get_vocab_size()
    # load model
    config = GPTConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    return model, tokenizer

@torch.no_grad()
def generate_response(model, tokenizer, history, max_length=MAX_GEN_LENGTH):
    # history is list of strings like ["User: hi", "AI: hello"]
    prompt = "\n".join(history) + "\nAI:"
    enc = tokenizer.encode(prompt)
    idx = torch.tensor(enc.ids, dtype=torch.long, device=DEVICE)[None, :]
    generated = idx
    for _ in range(max_length):
        # only feed last BLOCK_SIZE tokens
        if generated.size(1) > BLOCK_SIZE:
            generated = generated[:, -BLOCK_SIZE:]
        # forward & sample
        logits = model(generated)[:, -1, :] / TEMPERATURE
        v, _ = torch.topk(logits, TOP_K)
        min_logits = v[:, -1].unsqueeze(1)
        logits[logits < min_logits] = -float("Inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_id), dim=1)
    out = tokenizer.decode(generated[0].tolist())
    # strip off the prompt
    return out[len(prompt):].strip()

def chat_loop():
    model, tokenizer = load_model()
    history = []
    print("=== Helios Chatbot ===")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit"):
            break
        history.append(f"User: {user}")
        ai_resp = generate_response(model, tokenizer, history)
        print(f"AI: {ai_resp}")
        history.append(f"AI: {ai_resp}")
        # keep only last ~6 turns to fit BLOCK_SIZE
        if len(history) > 6:
            history = history[-6:]

if __name__ == "__main__":
    chat_loop()
