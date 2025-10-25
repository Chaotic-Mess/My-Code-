# model.py

import math
import torch
import torch.nn as nn

from config import BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD

class GPTConfig:
    """Configuration for GPT model."""
    def __init__(self, vocab_size, block_size=BLOCK_SIZE,
                 n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer    = n_layer
        self.n_head     = n_head
        self.n_embd     = n_embd

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # projection layers
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # dropout
        self.attn_dropout  = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

        # causal mask (lower triangular) to ensure tokens only attend to earlier positions
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        # project to Q, K, V and reshape for multi-head
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.block_size = config.block_size

        # token and position embeddings
        self.tok_emb   = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb   = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop      = nn.Dropout(0.1)

        # transformer blocks
        self.blocks    = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f      = nn.LayerNorm(config.n_embd)

        # language modeling head
        self.head      = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, "Input sequence length exceeds block size!"

        # embed tokens and positions
        token_embeddings = self.tok_emb(idx)            # (B, T, C)
        position_embeddings = self.pos_emb[:, :T, :]    # (1, T, C)
        x = self.drop(token_embeddings + position_embeddings)

        # forward through transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # predict logits
        logits = self.head(x)  # (B, T, vocab_size)

        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return logits, loss
        return logits
