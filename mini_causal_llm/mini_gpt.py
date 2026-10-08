"""
mini-GPT: 从零训练的小型 Transformer。

架构：6 层，256 维，8 头，约 10M 参数。
词表：字符级（中文 + 特殊标记）。
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 词表
# ============================================================
class CharTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = {}
            self.inv_vocab = {}
        else:
            self.vocab = vocab
            self.inv_vocab = {v: k for k, v in vocab.items()}

    def build(self, texts, min_freq=2):
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(t)

        chars = [c for c, n in counter.most_common() if n >= min_freq]
        specials = ["<pad>", "<unk>", "<Q>", "</Q>", "<A>", "</A>", "<eos>"]

        self.vocab = {}
        for s in specials:
            self.vocab[s] = len(self.vocab)
        for c in chars:
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return self

    def encode(self, text):
        unk = self.vocab["<unk>"]
        ids = []
        i = 0
        while i < len(text):
            # 先尝试匹配特殊标记
            matched = False
            for s in ["</Q>", "</A>", "<Q>", "<A>", "<eos>"]:
                if text.startswith(s, i):
                    ids.append(self.vocab[s])
                    i += len(s)
                    matched = True
                    break
            if not matched:
                ids.append(self.vocab.get(text[i], unk))
                i += 1
        return ids

    def decode(self, ids):
        return "".join(self.inv_vocab.get(int(i), "") for i in ids)

    @property
    def size(self):
        return len(self.vocab)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)


# ============================================================
# Transformer
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

        # 因果掩码
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
                 .view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (head_size ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=128,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == self.vocab_eos:
                break
        self.train()
        return idx