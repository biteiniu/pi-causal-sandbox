"""
my_scenario.py — 把你自己的因果场景套进 π-因果沙盒。

运行：python my_scenario.py
改场景：见文件末尾"如何改成你自己的场景"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from env.scm import SCM
from causal.interventional_discovery import (
    interventional_discovery, evaluate_against_true, edges_to_adjacency,
)
from causal.bayesian_scm import BayesianSCM
from causal.identifiability import IdentifiabilityChecker
from search.astar import astar_intervention

np.random.seed(42)
SEP = "=" * 62


def title(t):
    print(f"\n{SEP}\n{t}\n{SEP}")


# ============================================================
# 通用环境：让任意 SCM 兼容 BlockWorld 接口
# ============================================================
class GenericEnv:
    def __init__(self, scm, actions, action_to_intervention):
        self.scm = scm
        self.actions = actions
        self._a2i = action_to_intervention
        self.state = None
        self.history = []

    def reset(self):
        self.state = self.scm.sample()
        self.history = [self.state.copy()]
        return self.state

    def do(self, action):
        self.state = self.scm.sample(interventions=self._a2i(action))
        self.history.append(self.state.copy())
        return self.state

    def counterfactual(self, action):
        return self.scm.counterfactual(
            self.history, interventions=self._a2i(action)
        )

    def _action_to_intervention(self, action):
        return self._a2i(action)


# ============================================================
# 场景 1：营销漏斗
#   Ad -> Exposure -> Click -> Convert
# ============================================================
def build_marketing():
    scm = SCM()
    scm.add_variable("Ad", [], lambda p, n: 0.0, noise_scale=0.0)
    scm.add_variable("Exposure", ["Ad"],
                     lambda p, n: np.clip(0.9 * p["Ad"] + n, 0, 1),
                     noise_scale=0.1)
    scm.add_variable("Click", ["Exposure"],
                     lambda p, n: np.clip(0.85 * p["Exposure"] + n, 0, 1),
                     noise_scale=0.1)
    scm.add_variable("Convert", ["Click"],
                     lambda p, n: np.clip(0.75 * p["Click"] + n, 0, 1),
                     noise_scale=0.1)

    actions = [{"type": "advertise"}, {"type": "hold"}]

    def a2i(a):
        return {"Ad": 1.0} if a["type"] == "advertise" else {"Ad": 0.0}

    return scm, actions, a2i, "Convert"


# ============================================================
# 场景 2：医疗
#   Drug -> Level -> Relief
# ============================================================
def build_medical():
    scm = SCM()
    scm.add_variable("Drug", [], lambda p, n: 0.0, noise_scale=0.0)
    scm.add_variable("Level", ["Drug"],
                     lambda p, n: np.clip(0.85 * p["Drug"] + n, 0, 1),
                     noise_scale=0.1)
    scm.add_variable("Relief", ["Level"],
                     lambda p, n: np.clip(0.7 * p["Level"] + n, 0, 1),
                     noise_scale=0.15)

    actions = [{"type": "give_drug"}, {"type": "placebo"}]

    def a2i(a):
        return {"Drug": 1.0} if a["type"] == "give_drug" else {"Drug": 0.0}

    return scm, actions, a2i, "Relief"


# ============================================================
# 场景 3：服务器调参
#   Cache -> HitRate -> Good
# ============================================================
def build_server():
    scm = SCM()
    scm.add_variable("Cache", [], lambda p, n: 0.0, noise_scale=0.0)
    scm.add_variable("HitRate", ["Cache"],
                     lambda p, n: np.clip(0.9 * p["Cache"] + n, 0, 1),
                     noise_scale=0.08)
    scm.add_variable("Good", ["HitRate"],
                     lambda p, n: np.clip(0.8 * p["HitRate"] + n, 0, 1),
                     noise_scale=0.08)

    actions = [{"type": "scale_up"}, {"type": "scale_down"}]

    def a2i(a):
        return {"Cache": 1.0} if a["type"] == "scale_up" else {"Cache": 0.0}

    return scm, actions, a2i, "Good"


SCENARIOS = {
    "marketing": build_marketing,
    "medical":   build_medical,
    "server":    build_server,
}


# ============================================================
# 通用 5 步流水线
# ============================================================
def run_pipeline(scenario_name="marketing"):
    title(f"Scenario: {scenario_name}")
    scm, actions, a2i, target = SCENARIOS[scenario_name]()
    env = GenericEnv(scm, actions, a2i)
    var_names = list(scm.variables.keys())

    true_edges = set()
    for v in var_names:
        for p in scm.variables[v].parents:
            true_edges.add((p, v))

    print(f"变量：{var_names}")
    print(f"目标：{target}")
    print(f"可干预动作：{[a['type'] for a in actions]}")

    # ---- Step 1 ----
    title("Step 1：发现因果结构（算法不知道真结构）")
    _, direct = interventional_discovery(
        scm, var_names, n_samples=400, verbose=False,
    )
    print(f"真实边数：{len(true_edges)}")
    print(f"发现的边：")
    for (u, v), w in sorted(direct.items(), key=lambda x: -x[1]):
        tag = "✓" if (u, v) in true_edges else "✗ 假边"
        print(f"  {u:12s} → {v:12s}  (强度 {w:.2f})  {tag}")
    result = evaluate_against_true(direct, true_edges, var_names)
    print(f"\nF1 = {result['f1']:.2f}  "
          f"(P={result['precision']:.2f}, R={result['recall']:.2f})")

    # ---- Step 2 ----
    title(f"Step 2：预测 do(干预) 对 {target} 的效果")
    structure = {v: [] for v in var_names}
    for (u, v) in direct:
        structure[v].append(u)

    bayes = BayesianSCM(structure, var_names)
    data = []
    for action in actions:
        for _ in range(150):
            env.reset()
            data.append(env.do(action))
    bayes.fit_cpt(data)

    int_pos = a2i(actions[0])
    int_neg = a2i(actions[1])
    p1 = bayes.query(target, interventions=int_pos)
    p0 = bayes.query(target, interventions=int_neg)
    print(f"  P({target}=1 | do({int_pos})) = {p1:.2f}")
    print(f"  P({target}=1 | do({int_neg})) = {p0:.2f}")
    print(f"  因果效应 = {p1 - p0:+.2f}")

    # ---- Step 3 ----
    title("Step 3：从观测数据能不能估出因果？")
    adj = edges_to_adjacency(direct, var_names)
    checker = IdentifiabilityChecker(adj, var_names)
    info = checker.is_identifiable(var_names[0], target)
    print(f"  查询：P({target} | do({var_names[0]}))")
    print(f"  可识别：{info['identifiable']}")
    print(f"  原因：{info['reason']}")
    if info["adjustment_set"]:
        print(f"  调整集：{info['adjustment_set']}")

    # ---- Step 4 ----
    title("Step 4：反事实——如果当时不做会怎样？")
    env.reset()
    factual = env.do(actions[0])
    print(f"  事实：{actions[0]['type']}  →  {target} = {factual[target]:.2f}")
    cf = env.counterfactual(actions[1])
    print(f"  反事实：{actions[1]['type']}  →  {target} = {cf[target]:.2f}")
    if factual[target] > 0.5 and cf[target] < 0.5:
        print(f"  结论：{actions[0]['type']} 是 {target}=1 的必要原因")

    # ---- Step 5 ----
    title(f"Step 5：最省力的干预方案")
    init_state = env.reset()
    plan = astar_intervention(
        env, init_state, bayes_scm=bayes,
        target=target, max_depth=3, max_nodes=100,
    )
    print(f"  搜索节点数：{plan['n_expanded']}")
    print(f"  代价：{plan['cost']:.3f}")
    if plan["path"]:
        print(f"  最优干预：")
        for i, step in enumerate(plan["path"], 1):
            print(f"    {i}. {dict(step)}")
    else:
        print(f"  没有找到路径")


if __name__ == "__main__":
    # ========================================================
    # 改这一行切换场景：marketing / medical / server
    # ========================================================
    run_pipeline("medical")


# ============================================================
# 如何改成你自己的场景
# ============================================================
#
# 步骤 1：在 SCENARIOS 里加一个 build_xxx() 函数
#
#   def build_my_scene():
#       scm = SCM()
#
#       # 根变量（你能直接干预的）：机制写 lambda p, n: 0.0
#       scm.add_variable("我的干预变量", [], lambda p, n: 0.0, noise_scale=0.0)
#
#       # 中间变量：写出它依赖谁 + 怎么依赖
#       scm.add_variable("中间变量", ["我的干预变量"],
#                        lambda p, n: np.clip(0.9 * p["我的干预变量"] + n, 0, 1),
#                        noise_scale=0.1)
#
#       # 目标变量
#       scm.add_variable("我的目标", ["中间变量"],
#                        lambda p, n: np.clip(0.7 * p["中间变量"] + n, 0, 1),
#                        noise_scale=0.1)
#
#       actions = [{"type": "动手"}, {"type": "不动"}]
#       def a2i(a):
#           return {"我的干预变量": 1.0} if a["type"] == "动手" else {"我的干预变量": 0.0}
#       return scm, actions, a2i, "我的目标"
#
# 步骤 2：注册
#   SCENARIOS["my_scene"] = build_my_scene
#
# 步骤 3：改 run_pipeline("my_scene")
#
# 注意事项：
#   - 机制函数的输出最好在 [0, 1]，因为发现算法按 0.5 二值化
#   - noise_scale 越小越确定，越大越难发现
#   - 每个变量只能有一个"根"（parents=[]），否则要重新设计因果结构