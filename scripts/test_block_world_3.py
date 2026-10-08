import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.block_world_3 import BlockWorld3
from causal.interventional_discovery import (
    interventional_discovery,
    evaluate_against_true,
)

np.random.seed(42)
env = BlockWorld3()
var_names = list(env.scm.variables.keys())

print("=" * 60)
print("3-block chain world test")
print("=" * 60)

# True DAG
true_edges = set()
for v in var_names:
    for p in env.scm.variables[v].parents:
        true_edges.add((p, v))

print(f"\nVariables ({len(var_names)}): {var_names}")
print(f"\nTrue DAG ({len(true_edges)} edges):")
for i, j in sorted(true_edges):
    print(f"  {i:12s} -> {j}")

# Test 1: do(push, 1.0) effect
print("\n=== 1. Effect of do(push, 1.0) ===")
vals = []
for _ in range(50):
    env.reset()
    s = env.do({"type": "push", "force": 1.0})
    vals.append((s["Fallen_B"], s["Fallen_C"]))
print(f"  P(Fallen_B=1) = {np.mean([v[0] for v in vals]):.2f}")
print(f"  P(Fallen_C=1) = {np.mean([v[1] for v in vals]):.2f}")

# Test 2: do(wait) effect
print("\n=== 2. Effect of do(wait) ===")
vals = []
for _ in range(50):
    env.reset()
    s = env.do({"type": "wait"})
    vals.append((s["Fallen_B"], s["Fallen_C"]))
print(f"  P(Fallen_B=1) = {np.mean([v[0] for v in vals]):.2f}")
print(f"  P(Fallen_C=1) = {np.mean([v[1] for v in vals]):.2f}")

# Test 3: Discovery
print("\n=== 3. Interventional discovery ===")
potential, direct = interventional_discovery(
    env.scm, var_names, n_samples=300,
    edge_threshold=0.15, residual_threshold=0.15, verbose=True,
)

print("\n=== Direct edges ===")
for (i, j), w in sorted(direct.items(), key=lambda x: -x[1]):
    tag = "TRUE" if (i, j) in true_edges else "FALSE"
    print(f"  {i:12s} -> {j:12s}  w={w:.3f}  {tag}")

result = evaluate_against_true(direct, true_edges, var_names)
print(f"\n=== Evaluation ===")
print(f"  Precision = {result['precision']:.3f}")
print(f"  Recall    = {result['recall']:.3f}")
print(f"  F1        = {result['f1']:.3f}")
if result["false_positives"]:
    print(f"  FP = {result['false_positives']}")
if result["false_negatives"]:
    print(f"  FN = {result['false_negatives']}")

# Test 4: Counterfactual
print("\n=== 4. Counterfactual ===")
env.reset()
factual = env.do({"type": "push", "force": 1.0})
print(f"  Factual: push(1.0)  -> B={factual['Fallen_B']:.2f}, C={factual['Fallen_C']:.2f}")
cf = env.counterfactual({"type": "wait"})
print(f"  CF:      wait       -> B={cf['Fallen_B']:.2f}, C={cf['Fallen_C']:.2f}")

print("\nDone.")