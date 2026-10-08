"""
mini-GPT 推理：加载训好的模型，问答。
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mini_causal_llm.mini_gpt import CharTokenizer, MiniGPT

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")


def load_model(ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = os.path.join(CKPT_DIR, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型 {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]

    tokenizer = CharTokenizer.load(
        os.path.join(CKPT_DIR, "tokenizer.json")
    )
    model = MiniGPT(
        vocab_size=ckpt["vocab_size"],
        block_size=args["block_size"],
        n_layer=args["n_layer"],
        n_head=args["n_head"],
        n_embd=args["n_embd"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.vocab_eos = tokenizer.vocab["<eos>"]
    return model, tokenizer


def ask(model, tokenizer, question, max_tokens=200):
    prompt = f"<Q>{question}</Q><A>"
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            idx, max_new_tokens=max_tokens,
            temperature=0.7, top_k=40,
        )

    full = tokenizer.decode(out[0].tolist())
    # 只取 <A> 之后的内容
    if "<A>" in full:
        ans = full.split("<A>", 1)[1]
        ans = ans.split("<eos>")[0]
        ans = ans.replace("</A>", "").strip()
        return ans
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=str, default=None)
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    print("加载模型...")
    model, tokenizer = load_model()
    print("✓ 加载完成\n")

    if args.question:
        ans = ask(model, tokenizer, args.question)
        print(f"Q: {args.question}")
        print(f"A: {ans}")
        return

    if args.interactive:
        print("=" * 60)
        print("输入问题（Ctrl+C 退出）")
        print("=" * 60)
        while True:
            try:
                q = input("\nQ: ").strip()
                if not q:
                    continue
                ans = ask(model, tokenizer, q)
                print(f"A: {ans}")
            except KeyboardInterrupt:
                print("\n退出。")
                break
        return

    # 默认：跑几个示例
    print("=" * 60)
    print("示例测试")
    print("=" * 60)
    tests = [
        "帮我分析医疗场景",
        "营销场景的因果结构是什么",
        "如果当时不吃药会怎样",
        "服务器场景的最优方案",
    ]
    for q in tests:
        ans = ask(model, tokenizer, q)
        print(f"\nQ: {q}")
        print(f"A: {ans}")


if __name__ == "__main__":
    main()