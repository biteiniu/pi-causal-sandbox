import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.block_world import BlockWorld
from causal.notears import train_notears
from causal.bayesian_scm import BayesianSCM

np.random.seed(42)
env = BlockWorld()
var_names = list(env.scm.variables.keys())

# ========== 1. 混合数据 ==========
print("=== 1. 采集混合数据 ===")
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

# ========== 2. NOTEARS ==========
print("\n=== 2. NOTEARS 学 DAG ===")
learned = train_notears(
    data, len(var_names),
    epochs=1000, lr=0.005, l1=0.05, alpha=1.0, verbose=True,
)
print("  学到的邻接矩阵:")
print("      " + " ".join(f"{v[:8]:>8s}" for v in var_names))
for i, v in enumerate(var_names):
    row = " ".join(f"{learned[i, j]:8.3f}" for j in range(len(var_names)))
    print(f"  {v[:10]:10s} {row}")

# ========== 3. DAG 对比 ==========
print("\n=== 3. 真实 DAG ===")
true_dag = np.zeros((len(var_names), len(var_names)))
for i, v in enumerate(var_names):
    for p in env.scm.variables[v].parents:
        j = var_names.index(p)
        true_dag[j, i] = 1.0
true_edges = set(zip(*np.where(true_dag > 0.5)))

for thr in [0.1, 0.2, 0.3, 0.4]:
    pred_edges = set(zip(*np.where(np.abs(learned) > thr)))
    if not pred_edges:
        print(f"  阈值 {thr}: 预测边数=0")
        continue
    tp = len(true_edges & pred_edges)
    prec = tp / len(pred_edges)
    rec = tp / len(true_edges) if true_edges else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(
        f"  阈值 {thr}: 预测={len(pred_edges):2d} | "
        f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}"
    )

# ========== 4. 贝叶斯 SCM ==========
print("\n=== 4. 贝叶斯 SCM ===")
structure = {v: env.scm.variables[v].parents for v in var_names}
bayes = BayesianSCM(structure, var_names)

int_data = []
for action in env.actions:
    for _ in range(200):
        env.reset()
        s = env.do(action)
        int_data.append(s)
bayes.fit_cpt(int_data)
print(f"  CPT 条目数: {len(bayes.cpt)}")
print(f"  边际分布:")
for v, m in bayes.marginals.items():
    print(f"    {v:12s} P(1)={m[1]:.3f}")

# ========== 5. 推理对比（observe = do(Force_A=0)）==========
print("\n=== 5. 精确推理 vs 真实 ===")
print(f"{'查询':<32s} {'贝叶斯':>8s} {'真实':>8s} {'误差':>8s}")
print("-" * 60)
queries = [
    ("P(Fallen_B|do(push,1.0))", {"Force_A": 1.0}, "Fallen_B"),
    ("P(Fallen_B|do(wait))",     {"Force_A": 0.0}, "Fallen_B"),
    ("P(Fallen_B|do(observe))",  {"Force_A": 0.0}, "Fallen_B"),
]
for label, intervention, target in queries:
    p_bayes = bayes.query(target, interventions=intervention)
    true_vals = []
    for _ in range(200):
        env.reset()
        s = env.scm.sample(interventions=intervention)
        true_vals.append(s[target])
    p_true = float(np.mean([1 if v > 0.5 else 0 for v in true_vals]))
    err = abs(p_bayes - p_true)
    print(f"{label:<32s} {p_bayes:>8.3f} {p_true:>8.3f} {err:>8.3f}")

print("\n完成。")