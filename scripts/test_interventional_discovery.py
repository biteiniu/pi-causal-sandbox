import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.block_world import BlockWorld
from causal.interventional_discovery import (
    interventional_discovery,
    evaluate_against_true,
)

np.random.seed(42)
env = BlockWorld()
var_names = list(env.scm.variables.keys())

print("=" * 60)
print("干预驱动因果发现")
print("=" * 60)

# 真实边
true_edges = set()
for v in var_names:
    for p in env.scm.variables[v].parents:
        true_edges.add((p, v))

print(f"\n真实 DAG ({len(true_edges)} 条边):")
for i, j in sorted(true_edges):
    print(f"  {i:12s} -> {j}")

# 发现
print("\n=== 运行发现算法 ===")
potential, direct = interventional_discovery(
    env.scm, var_names, n_samples=400, edge_threshold=0.15, verbose=True
)

print("\n=== 潜在边（含间接）===")
for (i, j), w in sorted(potential.items(), key=lambda x: -x[1]):
    is_true = "✓ 真边" if (i, j) in true_edges else "  (间接/假)"
    print(f"  {i:12s} -> {j:12s}  w={w:.3f}  {is_true}")

print("\n=== 直接边（最终结果）===")
for (i, j), w in sorted(direct.items(), key=lambda x: -x[1]):
    is_true = "✓ 真边" if (i, j) in true_edges else "✗ 假边"
    print(f"  {i:12s} -> {j:12s}  w={w:.3f}  {is_true}")

# 评估
print("\n=== 评估 ===")
result = evaluate_against_true(direct, true_edges, var_names)
print(f"  Precision = {result['precision']:.3f}")
print(f"  Recall    = {result['recall']:.3f}")
print(f"  F1        = {result['f1']:.3f}")
print(f"  预测边数   = {result['n_pred']}")
print(f"  真实边数   = {result['n_true']}")
if result["false_positives"]:
    print(f"  假阳性     = {result['false_positives']}")
if result["false_negatives"]:
    print(f"  假阴性     = {result['false_negatives']}")

print("\n完成。")