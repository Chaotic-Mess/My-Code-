from src.model import TinyGPT
def test_model_forward():
    import torch
    m = TinyGPT(vocab_size=256, block_size=16, n_layer=2, n_head=2, n_embd=32)
    x = torch.randint(0,256,(2,16))
    logits, loss = m(x, x)
    assert logits.shape == (2,16,256)
