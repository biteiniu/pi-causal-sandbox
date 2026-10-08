"""
训练 mini-GPT。

用法：
  python mini_causal_llm/train.py              # 自动选 GPU/CPU
  python mini_causal_llm/train.py --epochs 10  # 自定义
"""

import sys
import os
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from mini_causal_llm.mini_gpt import CharTokenizer, MiniGPT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CKPT_DIR = os.path.join(HERE, "checkpoints")


class QADataset(Dataset):
    def __init__(self, lines, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.samples = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            ids.append(tokenizer.vocab["<eos>"])
            if len(ids) < 4:
                continue
            self.samples.append(ids[:block_size])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate(batch):
    max_len = max(x.size(0) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = max_len - x.size(0)
        xs.append(torch.cat([x, torch.zeros(pad, dtype=torch.long)]))
        ys.append(torch.cat([y, torch.full((pad,), -1, dtype=torch.long)]))
    return torch.stack(xs), torch.stack(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"设备：{device}")

    # ---- 加载数据 ----
    with open(os.path.join(DATA_DIR, "train.txt"), encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(os.path.join(DATA_DIR, "val.txt"), encoding="utf-8") as f:
        val_lines = f.readlines()

    print(f"训练样本：{len(train_lines)}")
    print(f"验证样本：{len(val_lines)}")

    # ---- 构建词表 ----
    tok_path = os.path.join(CKPT_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = CharTokenizer.load(tok_path)
        print(f"加载已有词表，大小 {tokenizer.size}")
    else:
        print("构建词表...")
        tokenizer = CharTokenizer()
        tokenizer.build(train_lines, min_freq=2)
        tokenizer.save(tok_path)
        print(f"词表大小：{tokenizer.size}")

    # ---- 数据集 ----
    train_ds = QADataset(train_lines, tokenizer, args.block_size)
    val_ds = QADataset(val_lines, tokenizer, args.block_size)
    print(f"训练数据集：{len(train_ds)} 条")
    print(f"验证数据集：{len(val_ds)} 条")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # ---- 模型 ----
    model = MiniGPT(
        vocab_size=tokenizer.size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    ).to(device)

    print(f"模型参数量：{model.num_params() / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- 训练 ----
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0
        n_batches = 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                print(
                    f"  Epoch {epoch+1} | batch {n_batches}/{len(train_dl)} | "
                    f"loss {loss.item():.4f}"
                )

        train_loss = total_loss / max(n_batches, 1)

        # 验证
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        scheduler.step()
        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"{dt:.1f}s"
        )

        # 保存最优
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "vocab_size": tokenizer.size,
            }, os.path.join(CKPT_DIR, "best.pt"))
            print(f"  ✓ 保存 best.pt (val {val_loss:.4f})")

    print(f"\n训练完成。最优验证 loss: {best_val:.4f}")
    print(f"模型保存在：{os.path.join(CKPT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()