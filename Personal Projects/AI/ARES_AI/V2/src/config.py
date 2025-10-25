from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "cuda"  # set to "cpu" if no GPU

    # data
    data_dir: str = "data"
    tokenizer_dir: str = "tokenizer"
    block_size: int = 192  # context length

    # model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    # train
    batch_size: int = 16
    lr: float = 3e-4
    max_steps: int = 1200
    eval_every: int = 150
    ckpt_dir: str = "checkpoints"

    grad_accum = 1
    patience   = 5

    
cfg = TrainConfig()
