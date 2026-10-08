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

NOISE_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
THRESHOLDS = [0.15, 0.25, 0.35]

print("=" * 72)
print("Noise robustness sweep (BlockWorldFork, additive noise)")
print("=" * 72)

env0 = BlockWorldFork(noise=0.05)
var_names = list(env0.scm.variables.keys())
true_edges = set()
for v in var_names:
    for p in env0.scm.variables[v].parents:
        true_edges.add((p, v))

print(f"\nTrue edges ({len(true_edges)}):")
for i, j in sorted(true_edges):
    print(f"  {i:12s} -> {j}")

print("\n" + "-" * 72)
print(f"{'noise':>6s} | " + " | ".join(f"thr={t:>4.2f}" for t in THRESHOLDS))
print("-" * 72)

for noise in NOISE_LEVELS:
    env = BlockWorldFork(noise=noise)
    _, direct = interventional_discovery(
        env.scm, var_names, n_samples=400,
        edge_threshold=0.15, residual_threshold=0.15, verbose=False,
    )
    row = {}
    for thr in THRESHOLDS:
        filtered = {k: v for k, v in direct.items() if v > thr}
        row[thr] = evaluate_against_true(filtered, true_edges, var_names)
    line = f"{noise:>6.2f} | "
    line += " | ".join(f"F1={row[t]['f1']:.3f}" for t in THRESHOLDS)
    print(line)

print("\n" + "=" * 72)
print("Detailed edges at thr=0.15")
print("=" * 72)

for noise in NOISE_LEVELS:
    env = BlockWorldFork(noise=noise)
    _, direct = interventional_discovery(
        env.scm, var_names, n_samples=400,
        edge_threshold=0.15, residual_threshold=0.15, verbose=False,
    )
    filtered = {k: v for k, v in direct.items() if v > 0.15}
    r = evaluate_against_true(filtered, true_edges, var_names)
    print(f"\n--- noise = {noise:.2f} ---")
    print(f"  Pred: {r['n_pred']}, TP: {r['tp']}, "
          f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    if r["false_negatives"]:
        print(f"  FN: {r['false_negatives']}")
    if r["false_positives"]:
        print(f"  FP: {r['false_positives']}")

    print(f"  Edge strengths:")
    for (i, j), w in sorted(direct.items(), key=lambda x: -x[1]):
        tag = "T" if (i, j) in true_edges else "F"
        print(f"    [{tag}] {i:12s} -> {j:12s}  w={w:.3f}")

print("\nDone.")