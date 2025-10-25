# src/chat.py
import torch
from src.infer import load_checkpoint, generate

SYSTEM = "You are ARES, a concise, helpful, 100% local assistant."

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_checkpoint(device=device)
    history = []

    # Ban PAD/BOS and role tokens from sampling; EOS is used only to stop
    model, tok = load_checkpoint(device=device)

    # Ban specials (keep EOS for stopping)
    ban_ids = [tok.pad, tok.bos, tok.system, tok.user, tok.assistant]

    # Precompute token ids that decode to *only whitespace*
    id_to_ws = []
    vocab = tok.tok.get_vocab()              # token -> id
    inv = {i:t for t,i in vocab.items()}     # id -> token string (ByteLevel)
    for i in range(tok.vocab_size):
        s = tok.decode([i])                  # decode the single token
        if s.strip("") == "" and s != "":    # is whitespace-only?
            id_to_ws.append(i)

    for maybe in ("system", "user", "assistant"):
        if hasattr(tok, maybe):
            ban_ids.append(getattr(tok, maybe))

    print("ARES (local). Type 'exit' to quit.")
    while True:
        user = input("you: ").strip()
        if user.lower() in {"exit","quit"}:
            break

        prompt = f"<|system|> {SYSTEM}\n"
        for (u, a) in history[-4:]:
            prompt += f"<|user|> {u}\n<|assistant|> {a}\n"
        prompt += f"<|user|> {user}\n<|assistant|> "

        # IMPORTANT: add BOS only, NOT EOS (so the model can continue)
        ids = [tok.bos] + tok.encode_ids(prompt, add_special=False)
        x = torch.tensor([ids], dtype=torch.long, device=model.head.weight.device)

        y = generate(
            model, x,
            max_new_tokens=192,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            stop_id=tok.eos,
            ban_ids=ban_ids,
            ws_ban_ids=id_to_ws, 
            min_tokens=12
        )[0].tolist()


        out = tok.decode(y[len(ids):]).strip("\r\n")
        # If it generated role tags, cut at the first one
        for tag in ("<|user|>", "<|assistant|>", "<|system|>"):
            if tag in out:
                out = out.split(tag)[0].strip()
        if not out:
            out = "(no output — try lower temperature or a longer max_new_tokens)"
        print("ARES:", out)
        history.append((user, out))

if __name__ == "__main__":
    main()
