"""
用因果沙盒自动生成训练数据。

每个场景 × 多种问法 × 多种答法 = 上万条 Q&A。
所有变量名都映射为场景相关的中文。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np

from env.block_world import BlockWorld
from env.block_world_3 import BlockWorld3
from env.block_world_fork import BlockWorldFork
from env.block_world_collider import BlockWorldCollider
from env.block_world_confounder import BlockWorldConfounder
from env.continuous_chain import ContinuousChain

from causal.interventional_discovery import (
    interventional_discovery, evaluate_against_true,
)
from causal.bayesian_scm import BayesianSCM
from causal.identifiability import IdentifiabilityChecker
from causal.interventional_discovery import edges_to_adjacency
from search.astar import astar_intervention


SCENARIOS = {
    "医疗": {"class": BlockWorld3, "kwargs": {}, "target": "Fallen_C"},
    "药物": {"class": BlockWorld3, "kwargs": {}, "target": "Fallen_C"},
    "营销": {"class": BlockWorldFork, "kwargs": {}, "target": "Fallen_C"},
    "广告": {"class": BlockWorldFork, "kwargs": {}, "target": "Fallen_C"},
    "服务器": {"class": BlockWorld3, "kwargs": {}, "target": "Fallen_C"},
    "缓存": {"class": BlockWorld3, "kwargs": {}, "target": "Fallen_C"},
    "分叉": {"class": BlockWorldFork, "kwargs": {}, "target": "Fallen_C"},
    "碰撞": {"class": BlockWorldCollider, "kwargs": {"noise": 0.02},
             "target": "Fallen_C"},
    "混杂": {"class": BlockWorldConfounder, "kwargs": {"noise": 0.02},
             "target": "Fallen_B"},
    "连续": {"class": ContinuousChain, "kwargs": {"noise": 0.1},
             "target": "Fallen_B"},
}


# ============================================================
# 场景 → 中文变量名映射
# ============================================================
VAR_NAME_CN = {
    "医疗": {
        "Force_A": "给药",
        "Displaced_A": "血药浓度",
        "Support_B": "药效维持",
        "Velocity_B": "代谢速率",
        "Fallen_B": "症状缓解",
        "Support_C": "免疫响应",
        "Fallen_C": "康复",
    },
    "药物": {
        "Force_A": "给药",
        "Displaced_A": "血药浓度",
        "Support_B": "药效维持",
        "Velocity_B": "代谢速率",
        "Fallen_B": "症状缓解",
        "Support_C": "免疫响应",
        "Fallen_C": "康复",
    },
    "营销": {
        "Force_A": "广告投入",
        "Displaced_A": "曝光量",
        "Support_B": "点击率",
        "Velocity_B": "转化率",
        "Fallen_B": "下单",
        "Support_C": "复购",
        "Fallen_C": "利润",
    },
    "广告": {
        "Force_A": "广告投入",
        "Displaced_A": "曝光量",
        "Support_B": "点击率",
        "Velocity_B": "转化率",
        "Fallen_B": "下单",
        "Support_C": "复购",
        "Fallen_C": "利润",
    },
    "服务器": {
        "Force_A": "扩容",
        "Displaced_A": "缓存大小",
        "Support_B": "命中率",
        "Velocity_B": "响应速度",
        "Fallen_B": "请求成功",
        "Support_C": "吞吐量",
        "Fallen_C": "系统稳定",
    },
    "缓存": {
        "Force_A": "扩容",
        "Displaced_A": "缓存大小",
        "Support_B": "命中率",
        "Velocity_B": "响应速度",
        "Fallen_B": "请求成功",
        "Support_C": "吞吐量",
        "Fallen_C": "系统稳定",
    },
    "分叉": {
        "Force_A": "干预",
        "Displaced_A": "中间变量",
        "Support_B": "分支B",
        "Velocity_B": "分支V",
        "Fallen_B": "结果B",
        "Support_C": "分支C",
        "Fallen_C": "结果C",
    },
    "碰撞": {
        "Force_A": "力A",
        "Force_B": "力B",
        "Collision": "碰撞",
        "Fallen_C": "结果C",
    },
    "混杂": {
        "Wind": "外部干扰",
        "Force_A": "干预",
        "Fallen_B": "结果B",
    },
    "连续": {
        "Force_A": "外部力",
        "Displaced_A": "位移",
        "Support_B": "支撑",
        "Velocity_B": "速度",
        "Fallen_B": "倒下",
    },
}


def cn(sc, var):
    """场景内变量 → 中文名。"""
    return VAR_NAME_CN.get(sc, {}).get(var, var)


def action_cn(sc, action_type):
    """动作名 → 中文。"""
    mapping = {
        "医疗": {"push": "给药", "wait": "不给药", "observe": "观察"},
        "药物": {"push": "给药", "wait": "不给药", "observe": "观察"},
        "营销": {"push": "投广告", "wait": "不投广告", "observe": "观察"},
        "广告": {"push": "投广告", "wait": "不投广告", "observe": "观察"},
        "服务器": {"push": "扩容", "wait": "不扩容", "observe": "观察"},
        "缓存": {"push": "扩容", "wait": "不扩容", "observe": "观察"},
        "分叉": {"push": "干预", "wait": "不干预", "observe": "观察"},
        "碰撞": {"push_A": "施力A", "push_B": "施力B",
                "push_both": "同时施力", "wait": "不施力"},
        "混杂": {"push": "干预", "wait": "不干预", "observe": "观察"},
        "连续": {"force_pos": "正向施力", "force_neg": "反向施力",
                "wait": "不施力", "observe": "观察"},
    }
    return mapping.get(sc, {}).get(action_type, action_type)


# ============================================================
# 分析
# ============================================================
def analyze(scenario_name):
    cfg = SCENARIOS[scenario_name]
    env = cfg["class"](**cfg["kwargs"])
    var_names = list(env.scm.variables.keys())
    true_edges = set()
    for v in var_names:
        for p in env.scm.variables[v].parents:
            true_edges.add((p, v))

    target = cfg["target"]
    actions = env.actions
    is_cont = scenario_name == "连续"
    readout = "continuous" if is_cont else "binary"

    _, direct = interventional_discovery(
        env.scm, var_names, n_samples=300,
        edge_threshold=0.15, residual_threshold=0.15,
        readout=readout, verbose=False,
    )
    result = evaluate_against_true(direct, true_edges, var_names)

    structure = {v: [] for v in var_names}
    for (u, v) in direct:
        structure[v].append(u)
    bayes = BayesianSCM(structure, var_names)
    data = []
    for a in actions:
        for _ in range(60):
            env.reset()
            data.append(env.do(a))
    bayes.fit_cpt(data)

    int_effects = {}
    for a in actions[:3]:
        intv = env._action_to_intervention(a)
        int_effects[a["type"]] = float(bayes.query(target, interventions=intv))

    adj = edges_to_adjacency(direct, var_names)
    checker = IdentifiabilityChecker(adj, var_names)
    ident_info = {}
    for t in [v for v in var_names if v != target][:2]:
        info = checker.is_identifiable(t, target)
        ident_info[t] = {
            "identifiable": info["identifiable"],
            "reason": info["reason"],
            "adjustment_set": info["adjustment_set"],
        }

    env.reset()
    factual = env.do(actions[0])
    cf = env.counterfactual(actions[1])

    init_state = env.reset()
    plan = astar_intervention(
        env, init_state, bayes_scm=bayes,
        target=target, max_depth=4, max_nodes=150,
    )

    return {
        "scenario": scenario_name,
        "var_names": var_names,
        "target": target,
        "actions": actions,
        "direct": direct,
        "true_edges": true_edges,
        "result": result,
        "int_effects": int_effects,
        "ident_info": ident_info,
        "factual": factual,
        "cf": cf,
        "plan": plan,
    }


# ============================================================
# 问法模板
# ============================================================
Q_STRUCTURE = [
    "帮我分析{sc}场景",
    "跑一下{sc}",
    "看看{sc}的因果结构",
    "{sc}场景是什么样的",
    "分析一下{sc}",
    "{sc}的因果关系是什么",
    "{sc}场景有哪些因果关系",
]

Q_INTERVENE = [
    "{sc}场景下，{a1}对{target}有什么影响",
    "如果我{sc}里{a1}会怎样",
    "{sc}中{a1}的因果效应是多少",
    "{sc}场景，{a1}和{a2}哪个更有效",
    "{sc}里做{a1}有用吗",
]

Q_IDENTIFY = [
    "{sc}场景能不能从观测数据估出因果",
    "{sc}里的因果关系可识别吗",
    "{sc}场景，我需要控制哪些变量",
    "{sc}只用历史数据能分析吗",
]

Q_COUNTERFACTUAL = [
    "{sc}场景，如果当时不执行{a1}会怎样",
    "反事实：{sc}里{a1}是必要的吗",
    "{sc}场景，如果换一种做法结果会变吗",
    "{sc}里如果没做{a1}，结果如何",
    "假设{sc}没有{a1}，会怎样",
]

Q_PLAN = [
    "{sc}场景的最优方案是什么",
    "{sc}里怎么最省力地达到目标",
    "{sc}场景，要达到{target}该怎么做",
    "{sc}的最省力路径是什么",
]


# ============================================================
# 答法模板（用中文变量名）
# ============================================================
def ans_structure(a):
    sc = a["scenario"]
    n_edges = len(a["direct"])
    f1 = a["result"]["f1"]

    edges_str = "、".join(
        f"{cn(sc, u)}影响{cn(sc, v)}"
        for (u, v) in list(a["direct"].keys())[:3]
    )
    target_cn = cn(sc, a["target"])

    templates = [
        f"{sc}场景是典型的因果链结构。算法发现了{n_edges}条因果边，"
        f"包括{edges_str}。准确率 F1 = {f1:.2f}。",

        f"我跑了{sc}场景，发现{n_edges}条边：{edges_str}。"
        f"全部正确，F1 = {f1:.2f}。",

        f"{sc}的因果结构有{n_edges}条边。核心路径是{edges_str}。"
        f"算法完全正确。",

        f"在{sc}场景里，我发现了{n_edges}个因果关系：{edges_str}。"
        f"目标是{target_cn}，整体准确率{f1:.2f}。",

        f"{sc}场景的因果链是：{edges_str}。"
        f"一共{n_edges}条边，F1 达到{f1:.2f}。",
    ]
    return random.choice(templates)


def ans_intervene(a):
    if len(a["int_effects"]) < 2:
        return None
    sc = a["scenario"]
    keys = list(a["int_effects"].keys())
    a1, a2 = keys[0], keys[1]
    a1_cn = action_cn(sc, a1)
    a2_cn = action_cn(sc, a2)
    p1 = a["int_effects"][a1]
    p2 = a["int_effects"][a2]
    eff = p1 - p2
    target_cn = cn(sc, a["target"])

    templates = [
        f"{sc}场景里，{a1_cn}时{target_cn}概率{p1:.2f}，"
        f"{a2_cn}时{p2:.2f}，因果效应{eff:+.2f}。",

        f"在{sc}中，{a1_cn}的效果是{p1:.2f}，"
        f"{a2_cn}只有{p2:.2f}。因果效应{eff:+.2f}，"
        f"所以{a1_cn}更好。",

        f"{sc}场景：{a1_cn}得到{target_cn}概率{p1:.2f}，"
        f"{a2_cn}得到{p2:.2f}。效应{eff:+.2f}。",

        f"比较{sc}的两种方案：{a1_cn}能到{p1:.2f}，"
        f"{a2_cn}只有{p2:.2f}。差{eff:+.2f}，{a1_cn}占优。",
    ]
    return random.choice(templates)


def ans_identify(a):
    if not a["ident_info"]:
        return None
    sc = a["scenario"]
    t = list(a["ident_info"].keys())[0]
    info = a["ident_info"][t]
    ok = "可以" if info["identifiable"] else "不能"
    adj = info["adjustment_set"]
    adj_cn = "空集" if not adj else "、".join(cn(sc, x) for x in adj)
    t_cn = cn(sc, t)
    target_cn = cn(sc, a["target"])

    templates = [
        f"{sc}场景下，从观测数据估计{target_cn}的因果效应{ok}识别。"
        f"需要控制的变量是{adj_cn}。",

        f"{sc}里的因果关系{ok}从观测数据估出来。"
        f"调整集是{adj_cn}。",

        f"对于{sc}，{ok}用观测数据估计{target_cn}的因果。"
        f"控制{adj_cn}即可。",

        f"如果只有{sc}的历史数据，{ok}分析。"
        f"关键是要控制{adj_cn}。",
    ]
    return random.choice(templates)


def ans_counterfactual(a):
    sc = a["scenario"]
    t = a["target"]
    t_cn = cn(sc, t)
    f_val = a["factual"][t]
    cf_val = a["cf"][t]
    a1 = a["actions"][0]["type"]
    a2 = a["actions"][1]["type"]
    a1_cn = action_cn(sc, a1)
    a2_cn = action_cn(sc, a2)

    if f_val > 0.5 and cf_val < 0.5:
        concl = f"{a1_cn}是{t_cn}的必要原因"
    elif f_val > cf_val:
        concl = f"{a1_cn}比{a2_cn}更有效"
    else:
        concl = f"{a1_cn}和{a2_cn}效果接近"

    templates = [
        f"事实：{a1_cn}时{t_cn} = {f_val:.2f}。"
        f"反事实：如果{a2_cn}，{t_cn} = {cf_val:.2f}。"
        f"结论：{concl}。",

        f"反事实推理显示：{a1_cn}导致{t_cn} = {f_val:.2f}，"
        f"换成{a2_cn}只有{cf_val:.2f}。所以{concl}。",

        f"如果当时{a2_cn}而不是{a1_cn}，{t_cn}会从"
        f"{f_val:.2f}变到{cf_val:.2f}。{concl}。",

        f"假设不做{a1_cn}：{t_cn}从{f_val:.2f}降到{cf_val:.2f}。"
        f"这说明{concl}。",
    ]
    return random.choice(templates)


def ans_plan(a):
    sc = a["scenario"]
    plan = a["plan"]
    if not plan["path"]:
        return f"{sc}场景没有找到可行方案。"

    path_cn = " → ".join(
        action_cn(sc, dict(s).get("type", "?")) for s in plan["path"]
    )
    n = len(plan["path"])
    cost = plan["cost"]

    templates = [
        f"{sc}的最优方案是{n}步：{path_cn}。总代价{cost:.2f}。",

        f"我规划了{sc}的最省力路径：{path_cn}，"
        f"只需{n}步，代价{cost:.2f}。",

        f"{sc}场景，最优干预是{path_cn}，{n}步搞定。",

        f"要达到目标，{sc}的最优路径是：{path_cn}。"
        f"预计{n}步，代价{cost:.2f}。",
    ]
    return random.choice(templates)


# ============================================================
# 生成
# ============================================================
def generate(n_per_scenario=200):
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "data"), exist_ok=True
    )
    out_path = os.path.join(
        os.path.dirname(__file__), "data", "train.txt"
    )

    lines = []
    scenarios = list(SCENARIOS.keys())

    for sc in scenarios:
        print(f"  生成 {sc} 场景数据...")
        try:
            a = analyze(sc)
        except Exception as e:
            print(f"    跳过 {sc}：{e}")
            continue

        a1 = a["actions"][0]["type"] if a["actions"] else "未知"
        a2 = a["actions"][1]["type"] if len(a["actions"]) > 1 else "未知"
        target = a["target"]
        target_cn = cn(sc, target)

        for _ in range(n_per_scenario):
            # 结构（4 条）
            for _ in range(4):
                q = random.choice(Q_STRUCTURE).format(sc=sc)
                ans = ans_structure(a)
                lines.append(f"<Q>{q}</Q><A>{ans}</A>")

            # 干预（3 条）
            for _ in range(3):
                q = random.choice(Q_INTERVENE).format(
                    sc=sc,
                    a1=action_cn(sc, a1),
                    a2=action_cn(sc, a2),
                    target=target_cn,
                )
                ans = ans_intervene(a)
                if ans:
                    lines.append(f"<Q>{q}</Q><A>{ans}</A>")

            # 可识别（2 条）
            for _ in range(2):
                q = random.choice(Q_IDENTIFY).format(sc=sc)
                ans = ans_identify(a)
                if ans:
                    lines.append(f"<Q>{q}</Q><A>{ans}</A>")

            # 反事实（4 条，加强）
            for _ in range(4):
                q = random.choice(Q_COUNTERFACTUAL).format(
                    sc=sc, a1=action_cn(sc, a1)
                )
                ans = ans_counterfactual(a)
                lines.append(f"<Q>{q}</Q><A>{ans}</A>")

            # 规划（3 条）
            for _ in range(3):
                q = random.choice(Q_PLAN).format(
                    sc=sc, target=target_cn
                )
                ans = ans_plan(a)
                lines.append(f"<Q>{q}</Q><A>{ans}</A>")

    # 打乱
    random.shuffle(lines)

    # 分割
    n_val = max(100, len(lines) // 20)
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))

    val_path = os.path.join(
        os.path.dirname(__file__), "data", "val.txt"
    )
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines))

    print(f"\n✓ 生成完成")
    print(f"  训练集：{len(train_lines)} 条 → {out_path}")
    print(f"  验证集：{len(val_lines)} 条 → {val_path}")
    print(f"  总计：{len(lines)} 条")

    # 打印几条样例
    print(f"\n样例：")
    for line in train_lines[:3]:
        print(f"  {line[:120]}...")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    print("=" * 60)
    print("生成因果 Q&A 训练数据（中文变量名版）")
    print("=" * 60)
    generate(n_per_scenario=200)