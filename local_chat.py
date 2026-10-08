"""
π-因果沙盒 · 本地因果助手（完全离线，不依赖任何 LLM API）

原理：
  1. 正则匹配识别用户意图（结构/干预/可识别/反事实/规划）
  2. 识别场景（医疗/营销/服务器...）
  3. 直接调用因果沙盒计算
  4. 用中文模板生成回答

运行：streamlit run local_chat.py
"""

import sys
import os
import re
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np

from env.block_world import BlockWorld
from env.block_world_3 import BlockWorld3
from env.block_world_fork import BlockWorldFork
from env.block_world_collider import BlockWorldCollider
from env.block_world_confounder import BlockWorldConfounder
from env.continuous_chain import ContinuousChain

from causal.interventional_discovery import (
    interventional_discovery, evaluate_against_true, edges_to_adjacency,
)
from causal.bayesian_scm import BayesianSCM
from causal.identifiability import IdentifiabilityChecker
from search.astar import astar_intervention


# ============================================================
# 场景库：加入中文别名和变量翻译
# ============================================================
SCENARIOS = {
    "链": {
        "class": BlockWorld3, "kwargs": {}, "target": "Fallen_C",
        "desc": "链式因果 A→B→C",
        "aliases": ["医疗", "药物", "给药", "药", "服务器", "缓存",
                    "链", "A到B", "因果链"],
        "var_cn": {
            "Force_A": "给药", "Displaced_A": "血药浓度",
            "Support_B": "药效开始", "Fallen_B": "症状缓解",
            "Support_C": "药效持续", "Fallen_C": "完全康复",
        },
        "target_cn": "完全康复",
    },
    "分叉": {
        "class": BlockWorldFork, "kwargs": {}, "target": "Fallen_C",
        "desc": "分叉+菱形结构",
        "aliases": ["营销", "广告", "漏斗", "分叉", "菱形", "fork"],
        "var_cn": {
            "Force_A": "广告投入", "Displaced_A": "曝光量",
            "Support_B": "点击率", "Support_C": "复购率",
            "Fallen_B": "加购", "Fallen_C": "转化",
        },
        "target_cn": "转化",
    },
    "碰撞": {
        "class": BlockWorldCollider, "kwargs": {"noise": 0.02},
        "target": "Fallen_C", "desc": "碰撞结构 A→C←B",
        "aliases": ["碰撞", "collider", "汇聚"],
        "var_cn": {
            "Force_A": "A的力量", "Force_B": "B的力量",
            "Collision": "碰撞发生", "Fallen_C": "C倒下",
        },
        "target_cn": "C倒下",
    },
    "混杂": {
        "class": BlockWorldConfounder, "kwargs": {"noise": 0.02},
        "target": "Fallen_B", "desc": "混杂结构 U→A,U→B",
        "aliases": ["混杂", "confounder", "共因"],
        "var_cn": {
            "Wind": "风力", "Force_A": "外力A", "Fallen_B": "B倒下",
        },
        "target_cn": "B倒下",
    },
    "连续": {
        "class": ContinuousChain, "kwargs": {"noise": 0.1},
        "target": "Fallen_B", "desc": "线性高斯连续变量",
        "aliases": ["连续", "线性", "continuous"],
        "var_cn": {
            "Force_A": "外力", "Displaced_A": "位移",
            "Support_B": "支撑", "Velocity_B": "速度",
            "Fallen_B": "倒下",
        },
        "target_cn": "倒下",
    },
    "简单": {
        "class": BlockWorld, "kwargs": {}, "target": "Fallen_B",
        "desc": "简单链 A→B",
        "aliases": ["简单", "基本", "最简"],
        "var_cn": {
            "Force_A": "外力", "Displaced_A": "位移",
            "Support_B": "支撑", "Velocity_B": "速度",
            "Fallen_B": "倒下",
        },
        "target_cn": "倒下",
    },
}


def cn(var, sc):
    """变量英文名 → 中文名"""
    return SCENARIOS[sc]["var_cn"].get(var, var)


# ============================================================
# 因果分析核心（带缓存）
# ============================================================
@st.cache_resource(show_spinner=False)
def analyze(scenario_name: str):
    cfg = SCENARIOS[scenario_name]
    np.random.seed(42)
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
        for _ in range(100):
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
    for t in [v for v in var_names if v != target][:3]:
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
        target=target, max_depth=4, max_nodes=200,
    )

    return {
        "scenario": scenario_name,
        "env": env,
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
# 意图识别
# ============================================================
def detect_scenario(text):
    """从问题里识别场景"""
    for sc, cfg in SCENARIOS.items():
        for alias in cfg["aliases"]:
            if alias in text:
                return sc
    return None


def detect_intent(text):
    """识别问题类型"""
    # 闲聊
    if re.search(r"^(你好|哈喽|hi|hello|在吗|嗨)", text, re.I):
        return "greet"
    if re.search(r"(谢谢|感谢|多谢|thank)", text, re.I):
        return "thanks"
    if re.search(r"(你是谁|你叫什么|你是什么|介绍.*自己)", text):
        return "who"
    if re.search(r"(再见|拜拜|bye)", text, re.I):
        return "bye"

    # 因果意图（优先级：规划 > 反事实 > 可识别 > 干预 > 结构）
    if re.search(r"(最优|最省|最便宜|规划|怎么办|怎么做|方案)", text):
        return "plan"
    if re.search(r"(反事实|如果当时|当时不|如果没|假使不)", text):
        return "counterfactual"
    if re.search(r"(可识别|能不能.*观测|能不能.*估|观测.*估计|能否识别)", text):
        return "identify"
    if re.search(r"(如果.*会|干预|do\(|效果|效应|影响|干预.*)", text):
        return "intervene"
    if re.search(r"(分析|结构|关系|谁导致|因果图|发现|看看|跑一下|查看)", text):
        return "structure"

    return "unknown"


# ============================================================
# 回答生成（中文模板，多种句式）
# ============================================================
def answer_greet():
    return random.choice([
        "你好！我可以帮你分析因果场景。\n\n"
        "试试说：\n"
        "- `帮我分析医疗场景`\n"
        "- `广告投入对转化有什么影响`\n"
        "- `如果当时不吃药会怎样`\n"
        "- `服务器场景最优方案`",
    ])


def answer_thanks():
    return random.choice(["不客气！", "随时问我。", "别客气～"])


def answer_who():
    return (
        "我是 π-因果沙盒的本地助手。\n\n"
        "**我不联网、不花钱、不上传数据。**\n\n"
        "我能做这些：\n"
        "- **发现因果结构**：谁导致谁\n"
        "- **干预预测**：如果做 X，Y 会怎样\n"
        "- **可识别性**：能不能从观测数据估出因果\n"
        "- **反事实**：如果当时不那样会怎样\n"
        "- **最优规划**：最省力的方案\n\n"
        "支持场景：医疗、营销、服务器、碰撞、混杂、连续。"
    )


def answer_bye():
    return random.choice(["再见！", "回头见～", "拜拜！"])


def answer_structure(a):
    sc = a["scenario"]
    cfg = SCENARIOS[sc]
    n_edges = len(a["direct"])
    f1 = a["result"]["f1"]
    target_cn = cfg["target_cn"]

    # 列出边（中文）
    edges_cn = []
    for (u, v) in list(a["direct"].keys())[:4]:
        edges_cn.append(f"**{cn(u, sc)}** → **{cn(v, sc)}**")

    edges_str = "、".join(edges_cn)

    openers = [
        f"好的，我跑了「{sc}」场景（{cfg['desc']}）。",
        f"我分析了一下{sc}场景。",
        f"跑完了，这是{sc}场景的结果。",
    ]
    body = [
        f"算法发现了 **{n_edges}** 条因果边：{edges_str}。",
        f"准确率 F1 = {f1:.2f}，全部正确。" if f1 > 0.99 else
        f"准确率 F1 = {f1:.2f}。",
    ]
    return random.choice(openers) + "\n\n" + "\n".join(body)


def answer_intervene(a):
    sc = a["scenario"]
    cfg = SCENARIOS[sc]
    target_cn = cfg["target_cn"]

    if not a["int_effects"]:
        return "这个场景不支持干预分析。"

    effects = a["int_effects"]
    keys = list(effects.keys())
    p1 = effects[keys[0]]
    p0 = effects[keys[1]] if len(keys) > 1 else None

    lines = [f"**关于 {target_cn} 的干预效果：**\n"]
    for k in keys[:3]:
        p = effects[k]
        lines.append(f"- 执行 `{k}` → {target_cn}概率 **{p:.2f}**")

    if p0 is not None:
        eff = p1 - p0
        if eff > 0.3:
            concl = f"因果效应 **{eff:+.2f}**，非常强。"
        elif eff > 0:
            concl = f"因果效应 {eff:+.2f}，有正向影响。"
        else:
            concl = f"因果效应 {eff:+.2f}，负向。"
        lines.append(f"\n{concl}")

    return "\n".join(lines)


def answer_identify(a):
    sc = a["scenario"]
    cfg = SCENARIOS[sc]
    target_cn = cfg["target_cn"]

    if not a["ident_info"]:
        return "无法识别。"

    lines = [f"**能不能从观测数据估计「{target_cn}」的因果效应？**\n"]
    for t, info in list(a["ident_info"].items())[:2]:
        ok = "✅ 能" if info["identifiable"] else "❌ 不能"
        t_cn = cn(t, sc)
        adj = info["adjustment_set"]
        adj_cn = "空集" if not adj else "、".join(cn(x, sc) for x in adj)

        lines.append(f"**P({target_cn} | do({t_cn}))**")
        lines.append(f"- 可识别：{ok}")
        lines.append(f"- 需要控制：**{adj_cn}**")
        lines.append("")

    return "\n".join(lines)


def answer_counterfactual(a):
    sc = a["scenario"]
    cfg = SCENARIOS[sc]
    target_cn = cfg["target_cn"]
    a1 = a["actions"][0]["type"]
    a2 = a["actions"][1]["type"]
    f_val = a["factual"][a["target"]]
    cf_val = a["cf"][a["target"]]

    lines = ["**反事实推理：**\n"]
    lines.append(f"- **事实**：执行 `{a1}` → {target_cn} = **{f_val:.2f}**")
    lines.append(f"- **反事实**：如果执行 `{a2}` → {target_cn} = **{cf_val:.2f}**")
    lines.append("")

    if f_val > 0.5 and cf_val < 0.5:
        lines.append(f"**结论**：`{a1}` 是 {target_cn} 达成的**必要原因**。")
    elif f_val > cf_val:
        lines.append(f"**结论**：`{a1}` 比 `{a2}` 效果更好。")
    else:
        lines.append(f"**结论**：`{a1}` 和 `{a2}` 效果接近。")

    return "\n".join(lines)


def answer_plan(a):
    sc = a["scenario"]
    cfg = SCENARIOS[sc]
    target_cn = cfg["target_cn"]
    plan = a["plan"]

    if not plan["path"]:
        return f"没有找到让 {target_cn} 达成的方案。"

    n = len(plan["path"])
    actions = [dict(s).get("type", "?") for s in plan["path"]]
    path_str = " → ".join(f"`{x}`" for x in actions)

    lines = [f"**让 {target_cn} 达成的最省力方案：**\n"]
    lines.append(f"共 **{n}** 步：{path_str}")
    if plan["cost"] < float("inf"):
        lines.append(f"总代价：{plan['cost']:.3f}")
    return "\n".join(lines)


# ============================================================
# 主应答
# ============================================================
def respond(text):
    text = text.strip()
    if not text:
        return "请说点什么～"

    intent = detect_intent(text)

    # 闲聊
    if intent == "greet":
        return answer_greet()
    if intent == "thanks":
        return answer_thanks()
    if intent == "who":
        return answer_who()
    if intent == "bye":
        return answer_bye()

    # 因果意图
    scenario = detect_scenario(text)
    if scenario is None:
        # 没识别到场景，问用户
        return (
            "我还不确定你说的场景。试试这些：\n\n"
            "- `帮我分析医疗场景`\n"
            "- `营销场景的因果结构是什么`\n"
            "- `如果当时不吃药会怎样`\n"
            "- `服务器场景的最优方案`\n\n"
            "支持场景：**医疗、营销、服务器、碰撞、混杂、连续**"
        )

    # 跑分析
    try:
        a = analyze(scenario)
    except Exception as e:
        return f"分析出错：`{e}`"

    # 按意图回答
    if intent == "structure":
        return answer_structure(a)
    if intent == "intervene":
        return answer_intervene(a)
    if intent == "identify":
        return answer_identify(a)
    if intent == "counterfactual":
        return answer_counterfactual(a)
    if intent == "plan":
        return answer_plan(a)

    # 默认：全讲一遍
    parts = [
        answer_structure(a),
        "",
        answer_intervene(a),
    ]
    return "\n\n---\n\n".join(parts)


# ============================================================
# Streamlit 界面
# ============================================================
st.set_page_config(
    page_title="π-因果沙盒 · 本地助手",
    page_icon="π",
    layout="centered",
)
st.title("π-因果沙盒 · 本地助手")
st.caption("完全离线 · 不花钱 · 不联网 · 不依赖 API")

# 侧边栏
with st.sidebar:
    st.header("📖 支持的场景")
    for sc, cfg in SCENARIOS.items():
        st.markdown(f"**{sc}**：{cfg['desc']}")
        st.caption(f"别名：{'、'.join(cfg['aliases'][:3])}")

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "你好！我是 π-因果沙盒的**本地助手**。\n\n"
            "**完全离线**，不花钱，不联网，不上传数据。\n\n"
            "试试这些：\n"
            "- `帮我分析医疗场景`\n"
            "- `广告投入对转化有什么影响`\n"
            "- `如果当时不吃药会怎样`\n"
            "- `服务器场景最优方案`\n\n"
            "或者随便说点什么～"
        ),
    }]

# 展示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入
if user_input := st.chat_input("说点什么..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = respond(user_input)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)