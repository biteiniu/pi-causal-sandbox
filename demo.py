"""
pi-causal-sandbox 30-second demo.

Run: python demo.py

This script walks through the full pipeline on a small causal system:
    1. Define a causal world (or use a built-in one)
    2. Discover the causal graph by intervening
    3. Check whether do(X) is identifiable
    4. Answer a counterfactual question
    5. Plan the optimal intervention with A*
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from env.block_world_fork import BlockWorldFork
from causal.interventional_discovery import (
    interventional_discovery, evaluate_against_true,
)
from causal.bayesian_scm import BayesianSCM
from causal.identifiability import IdentifiabilityChecker
from causal.interventional_discovery import edges_to_adjacency
from search.astar import astar_intervention

np.random.seed(42)

SEP = "=" * 60


def title(text):
    print(f"\n{SEP}\n{text}\n{SEP}")


# ============================================================
# 0. 场景
# ============================================================
title("Scenario: A supports B and C (fork + diamond)")
print("""
真实因果结构（我们知道，但算法不知道）：

    Force_A
        |
    Displaced_A
      /     \\
  Support_B  Support_C
      |          |
  Fallen_B --> Fallen_C

问题：
  1. 算法能自己找出这个结构吗？
  2. 我能预测 do(Force_A=1) 对 Fallen_C 的效果吗？
  3. 如果当时不推 A，C 会倒吗？
  4. 要让 C 倒下，最省力的干预是什么？
""")

env = BlockWorldFork(noise=0.05)
var_names = list(env.scm.variables.keys())
true_edges = set()
for v in var_names:
    for p in env.scm.variables[v].parents:
        true_edges.add((p, v))

# ============================================================
# 1. 结构发现
# ============================================================
title("Step 1: Discover the causal graph (no prior knowledge)")

print("正在主动干预...")
_, direct = interventional_discovery(
    env.scm, var_names, n_samples=300, verbose=False,
)

print("\n算法发现了这些因果边：")
for (cause, effect), weight in sorted(direct.items(), key=lambda x: -x[1]):
    tag = "✓" if (cause, effect) in true_edges else "✗ 假边"
    print(f"  {cause:14s} → {effect:14s}  (强度 {weight:.2f})  {tag}")

result = evaluate_against_true(direct, true_edges, var_names)
print(f"\n准确率: Precision={result['precision']:.2f}, "
      f"Recall={result['recall']:.2f}, F1={result['f1']:.2f}")

# ============================================================
# 2. 贝叶斯推理
# ============================================================
title("Step 2: Predict do(Force_A=1) effect on Fallen_C")

structure = {v: [] for v in var_names}
for (cause, effect) in direct:
    structure[effect].append(cause)

bayes = BayesianSCM(structure, var_names)
data = []
for action in env.actions:
    for _ in range(100):
        env.reset()
        data.append(env.do(action))
bayes.fit_cpt(data)

p_do_1 = bayes.query("Fallen_C", interventions={"Force_A": 1.0})
p_do_0 = bayes.query("Fallen_C", interventions={"Force_A": 0.0})

print(f"\nP(Fallen_C=1 | do(Force_A=1)) = {p_do_1:.2f}")
print(f"P(Fallen_C=1 | do(Force_A=0)) = {p_do_0:.2f}")
print(f"因果效应 = {p_do_1 - p_do_0:.2f}")

# ============================================================
# 3. 可识别性
# ============================================================
title("Step 3: Is do(Support_B) identifiable from observational data?")

adj = edges_to_adjacency(direct, var_names)
checker = IdentifiabilityChecker(adj, var_names)
info = checker.is_identifiable("Support_B", "Fallen_C")

print(f"\n问题：能不能从观测数据估计 P(Fallen_C | do(Support_B))？")
print(f"答案：{'能' if info['identifiable'] else '不能'}")
print(f"原因：{info['reason']}")
if info["adjustment_set"]:
    print(f"需要控制这些变量：{info['adjustment_set']}")

# ============================================================
# 4. 反事实
# ============================================================
title("Step 4: Counterfactual — what if we had NOT pushed?")

env.reset()
factual = env.do({"type": "push", "force": 1.0})
print(f"\n事实：我们推了 A (force=1.0)")
print(f"  → Fallen_B = {factual['Fallen_B']:.2f}")
print(f"  → Fallen_C = {factual['Fallen_C']:.2f}")

cf = env.counterfactual({"type": "wait"})
print(f"\n反事实：如果当时没推")
print(f"  → Fallen_B = {cf['Fallen_B']:.2f}")
print(f"  → Fallen_C = {cf['Fallen_C']:.2f}")

if factual["Fallen_C"] > 0.5 and cf["Fallen_C"] < 0.5:
    print("\n结论：推 A 是 C 倒下的必要原因。")

# ============================================================
# 5. A* 规划
# ============================================================
title("Step 5: Plan the cheapest intervention to topple C")

init_state = env.reset()
plan = astar_intervention(
    env, init_state, bayes_scm=bayes,
    target="Fallen_C", max_depth=3, max_nodes=100,
)

print(f"\n搜索节点数: {plan['n_expanded']}")
print(f"总代价: {plan['cost']:.3f}")
if plan["path"]:
    print("\n最优干预序列：")
    for i, step in enumerate(plan["path"], 1):
        action = dict(step)
        if action.get("type") == "push":
            print(f"  {i}. 推 A，力度 {action.get('force', 1.0)}")
        else:
            print(f"  {i}. {action.get('type')}")
else:
    print("  没有找到路径")

title("Demo finished.")
print("""
你现在知道这个项目能做什么了：

  1. 主动干预 → 发现因果结构（不需要先验知识）
  2. 贝叶斯推理 → 预测 do(X) 的效果
  3. 后门准则 → 判断能不能从观测数据估计因果
  4. 反事实推理 → 回答"如果当时..."
  5. A* 规划 → 找最省力的干预方案

下一步可以：
  - 改 BlockWorldFork 为自己的场景
  - 用 continuous_chain 处理连续变量
  - 看 docs/TECHNICAL_REPORT.html 理解原理
""")