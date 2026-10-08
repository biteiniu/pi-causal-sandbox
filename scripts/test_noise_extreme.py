import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.block_world_fork import BlockWorldFork
from causal.interventional_discovery import (
    interventional_discovery,
    evaluate_against_true,
)

np.random.seed(42)

NOISE_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
EDGE_THRESHOLD = 0.15

print("=" * 78)
print("Extreme noise sweep (BlockWorldFork, additive noise)")
print("=" * 78)

env0 = BlockWorldFork(noise=0.05)
var_names = list(env0.scm.variables.keys())
true_edges = set()
for v in var_names:
    for p in env0.scm.variables[v].parents:
        true_edges.add((p, v))

print(f"\nTrue edges ({len(true_edges)}):")
for i, j in sorted(true_edges):
    print(f"  {i:12s} -> {j}")

print("\n" + "-" * 78)
print(f"{'noise':>6s} | {'F1':>6s} | {'P':>6s} | {'R':>6s} | "
      f"{'pred':>4s} | {'TP':>4s} | {'FN':>3s} | {'FP':>3s}")
print("-" * 78)

all_results = []
for noise in NOISE_LEVELS:
    env = BlockWorldFork(noise=noise)
    _, direct = interventional_discovery(
        env.scm, var_names, n_samples=500,
        edge_threshold=EDGE_THRESHOLD,
        residual_threshold=EDGE_THRESHOLD,
        verbose=False,
    )
    filtered = {k: v for k, v in direct.items() if v > EDGE_THRESHOLD}
    r = evaluate_against_true(filtered, true_edges, var_names)
    all_results.append((noise, r, direct))

    print(
        f"{noise:>6.2f} | {r['f1']:>6.3f} | {r['precision']:>6.3f} | "
        f"{r['recall']:>6.3f} | {r['n_pred']:>4d} | {r['tp']:>4d} | "
        f"{len(r['false_negatives']):>3d} | {len(r['false_positives']):>3d}"
    )

# 每个噪声下的边详情
print("\n" + "=" * 78)
print("Edge-by-edge strengths")
print("=" * 78)

for noise, r, direct in all_results:
    print(f"\n--- noise = {noise:.2f} | F1 = {r['f1']:.3f} ---")
    if r["false_negatives"]:
        print(f"  FN: {r['false_negatives']}")
    if r["false_positives"]:
        print(f"  FP: {r['false_positives']}")

    # 打印所有潜在边（含被过滤的）
    print(f"  Direct edges ({len(direct)}):")
    for (i, j), w in sorted(direct.items(), key=lambda x: -x[1]):
        tag = "T" if (i, j) in true_edges else "F"
        print(f"    [{tag}] {i:12s} -> {j:12s}  w={w:.3f}")

# 效应衰减曲线（每条真边在各级噪声下的 w 值）
print("\n" + "=" * 78)
print("True edge strength decay")
print("=" * 78)

print(f"\n{'edge':<32s} | " + " | ".join(f"σ={n:>3.1f}" for n, _, _ in all_results))
print("-" * 78)

for edge in sorted(true_edges):
    row = f"{edge[0]:12s} -> {edge[1]:12s}  "
    row += "| "
    for _, _, direct in all_results:
        w = direct.get(edge, 0.0)
        row += f"{w:>5.3f}  | "
    print(row)

print("\nDone.")