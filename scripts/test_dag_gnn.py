import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.block_world import BlockWorld
from causal.notears import train_notears
from causal.dag_gnn import train_dag_gnn

np.random.seed(42)
env = BlockWorld()
var_names = list(env.scm.variables.keys())
n_vars = len(var_names)

print("=== 采集混合数据 ===")
data = []
for _ in range(800):
    if np.random.rand() < 0.5:
        s = env.scm.sample()
    else:
        f = float(np.random.choice([0.0, 0.5, 1.0]))
        s = env.scm.sample(interventions={"Force_A": f})
    data.append([s[v] for v in var_names])
data = np.array(data)
print(f"  shape={data.shape}")

true_dag = np.zeros((n_vars, n_vars))
for i, v in enumerate(var_names):
    for p in env.scm.variables[v].parents:
        j = var_names.index(p)
        true_dag[j, i] = 1.0
true_edges = set(zip(*np.where(true_dag > 0.5)))
print(f"\n=== 真实 DAG ({len(true_edges)} 条边) ===")
for i, j in sorted(true_edges):
    print(f"  {var_names[i]:12s} -> {var_names[j]}")


def eval_dag(learned, thr):
    pred = set(zip(*np.where(np.abs(learned) > thr)))
    if not pred:
        return 0.0, 0.0, 0.0, pred
    tp = len(true_edges & pred)
    p = tp / len(pred)
    r = tp / len(true_edges) if true_edges else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f, pred


print("\n=== DAG-GNN ===")
raw_A, norm_A, dag_A = train_dag_gnn(
    data, n_vars, epochs=5000, lr=1e-3, l1=1e-4, alpha=0.1, verbose=True
)

print("\n  原始 A:")
print("      " + " ".join(f"{v[:8]:>8s}" for v in var_names))
for i, v in enumerate(var_names):
    row = " ".join(f"{raw_A[i, j]:8.3f}" for j in range(n_vars))
    print(f"  {v[:10]:10s} {row}")

# ========== 排序诊断 ==========
print("\n=== 排序诊断：A 绝对值前 10 大边 ===")
flat = []
for i in range(n_vars):
    for j in range(n_vars):
        if i != j:
            flat.append((abs(raw_A[i, j]), i, j, raw_A[i, j]))
flat.sort(reverse=True)
for rank, (mag, i, j, val) in enumerate(flat[:10], 1):
    is_true = "✓ 真" if (i, j) in true_edges else "✗ 假"
    print(
        f"  {rank:2d}. {var_names[i]:12s} -> {var_names[j]:12s} "
        f"|A|={mag:.4f} 值={val:+.4f} {is_true}"
    )

print("\n=== F1（不同阈值）===")
best_f1 = 0.0
best_thr = None
for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    p, r, f, pred = eval_dag(norm_A, thr)
    print(f"  阈值 {thr}: P={p:.3f} R={r:.3f} F1={f:.3f} 边数={len(pred)}")
    if f > best_f1:
        best_f1 = f
        best_thr = thr

print(f"\n  最佳阈值: {best_thr}, F1={best_f1:.3f}")

print("\n完成。")