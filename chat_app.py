"""
π-因果沙盒 · 对话界面
运行：streamlit run chat_app.py

直接打字说话，它跑因果分析并用中文回你。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np

# 环境
from env.block_world import BlockWorld
from env.block_world_3 import BlockWorld3
from env.block_world_fork import BlockWorldFork
from env.block_world_collider import BlockWorldCollider
from env.block_world_confounder import BlockWorldConfounder
from env.continuous_chain import ContinuousChain

# 因果
from causal.interventional_discovery import (
    interventional_discovery, evaluate_against_true, edges_to_adjacency,
)
from causal.bayesian_scm import BayesianSCM
from causal.identifiability import IdentifiabilityChecker
from search.astar import astar_intervention

# ============================================================
# 页面
# ============================================================
st.set_page_config(
    page_title="π-因果沙盒 · 对话",
    page_icon="💬",
    layout="centered",
)
st.title("π-因果沙盒 · 对话版")
st.caption("你打字 → 它跑因果分析 → 用中文回答")

# ============================================================
# 场景库
# ============================================================
SCENARIOS = {
    "分叉": {
        "class": BlockWorldFork, "kwargs": {},
        "target": "Fallen_C",
        "desc": "A 影响 B 和 C，B 也影响 C",
    },
    "链": {
        "class": BlockWorld3, "kwargs": {},
        "target": "Fallen_C",
        "desc": "A → B → C",
    },
    "碰撞": {
        "class": BlockWorldCollider, "kwargs": {"noise": 0.02},
        "target": "Fallen_C",
        "desc": "A → C ← B",
    },
    "混杂": {
        "class": BlockWorldConfounder, "kwargs": {"noise": 0.02},
        "target": "Fallen_B",
        "desc": "U → A, U → B",
    },
    "连续": {
        "class": ContinuousChain, "kwargs": {"noise": 0.1},
        "target": "Fallen_B",
        "desc": "线性高斯连续变量",
    },
    "简单": {
        "class": BlockWorld, "kwargs": {},
        "target": "Fallen_B",
        "desc": "A → B 最简",
    },
}

# 关键词 → 场景
SCENARIO_KEYWORDS = {
    "分叉": "分叉", "菱形": "分叉", "fork": "分叉", "营销": "分叉",
    "广告": "分叉", "漏斗": "分叉",
    "碰撞": "碰撞", "collider": "碰撞",
    "混杂": "混杂", "confound": "混杂", "confounder": "混杂",
    "医疗": "链", "药": "链", "drug": "链",
    "服务器": "链", "cache": "链", "缓存": "链",
    "连续": "连续", "continuous": "连续", "线性": "连续",
    "简单": "简单", "基本": "简单", "最简单": "简单",
}

# ============================================================
# 会话状态
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "你好！我帮你做因果分析。\n\n"
                "**你可以这样说：**\n"
                "- `帮我分析医疗场景`\n"
                "- `分析营销漏斗`\n"
                "- `看下服务器场景`\n"
                "- `分析分叉结构`\n\n"
                "**想只做某一步，也可以说：**\n"
                "- `发现结构`\n"
                "- `预测干预效果`\n"
                "- `能不能从观测估因果`\n"
                "- `反事实`\n"
                "- `最优方案`\n\n"
                "说一句就行，我在几秒内给你结果。"
            ),
        }
    ]

if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ============================================================
# 展示历史消息
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# 核心分析函数
# ============================================================
@st.cache_resource(show_spinner=False)
def analyze(scenario_name: str):
    """跑完整流水线，返回所有结果。"""
    np.random.seed(42)
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

    # Step 1: 发现
    _, direct = interventional_discovery(
        env.scm, var_names,
        n_samples=300,
        edge_threshold=0.15, residual_threshold=0.15,
        readout=readout, verbose=False,
    )
    result = evaluate_against_true(direct, true_edges, var_names)

    # Step 2: 贝叶斯
    structure = {v: [] for v in var_names}
    for (u, v) in direct:
        structure[v].append(u)
    bayes = BayesianSCM(structure, var_names)
    data = []
    for action in actions:
        for _ in range(100):
            env.reset()
            data.append(env.do(action))
    bayes.fit_cpt(data)

    # 干预效果
    int_effects = {}
    if len(actions) >= 2:
        for a in actions[:3]:
            intv = env._action_to_intervention(a)
            p = bayes.query(target, interventions=intv)
            int_effects[a["type"]] = p

    # Step 3: 可识别性
    adj = edges_to_adjacency(direct, var_names)
    checker = IdentifiabilityChecker(adj, var_names)
    treatments = [v for v in var_names if v != target]
    ident_info = {}
    for t in treatments[:3]:
        ident_info[t] = checker.is_identifiable(t, target)

    # Step 4: 反事实
    env.reset()
    factual = env.do(actions[0])
    cf = env.counterfactual(actions[1])

    # Step 5: A*
    init_state = env.reset()
    plan = astar_intervention(
        env, init_state, bayes_scm=bayes,
        target=target, max_depth=4, max_nodes=200,
    )

    return {
        "scenario": scenario_name,
        "desc": cfg["desc"],
        "var_names": var_names,
        "target": target,
        "actions": actions,
        "true_edges": true_edges,
        "direct": direct,
        "result": result,
        "bayes": bayes,
        "int_effects": int_effects,
        "ident_info": ident_info,
        "factual": factual,
        "cf": cf,
        "plan": plan,
    }


# ============================================================
# 意图解析
# ============================================================
def parse_scenario(text: str):
    """从文本里找场景。返回场景名或 None。"""
    text_lower = text.lower()
    for kw, s in SCENARIO_KEYWORDS.items():
        if kw in text_lower or kw in text:
            return s
    return None


def detect_intent(text: str):
    """识别用户想干什么。"""
    t = text
    intents = []
    if any(k in t for k in ["结构", "发现", "谁导致", "因果图", "关系", "全部", "完整", "分析"]):
        intents.append("all")
    if any(k in t for k in ["干预", "do(", "如果我", "预测"]):
        intents.append("intervene")
    if any(k in t for k in ["可识别", "观测", "估计", "能不能"]):
        intents.append("identify")
    if any(k in t for k in ["反事实", "如果当时", "当时不"]):
        intents.append("counterfactual")
    if any(k in t for k in ["最优", "最省", "方案", "规划", "怎么办"]):
        intents.append("plan")
    if not intents:
        intents.append("all")
    # all 优先级最高
    if "all" in intents:
        return "all"
    return intents[0]


# ============================================================
# 生成中文回答
# ============================================================
def reply_structure(a):
    lines = [f"**场景**：{a['scenario']}（{a['desc']}）\n"]
    lines.append(f"**变量**：{', '.join(a['var_names'])}")
    lines.append(f"**目标**：{a['target']}\n")
    lines.append("### 算法发现的因果边\n")
    for (u, v), w in sorted(a["direct"].items(), key=lambda x: -x[1]):
        tag = "✅ 真边" if (u, v) in a["true_edges"] else "❌ 假边"
        lines.append(f"- `{u}` → `{v}`  强度 {w:.2f}  {tag}")
    r = a["result"]
    lines.append(
        f"\n**准确率**：F1 = {r['f1']:.3f}"
        f"（P = {r['precision']:.2f}, R = {r['recall']:.2f}）"
    )
    if r["false_positives"]:
        lines.append(f"\n⚠️ 假阳性：{r['false_positives']}")
    return "\n".join(lines)


def reply_intervene(a):
    lines = [f"### 预测：对 `{a['target']}` 的干预效果\n"]
    for action_type, p in a["int_effects"].items():
        lines.append(f"- 执行 `{action_type}` → P({a['target']}=1) = **{p:.2f}**")
    if len(a["int_effects"]) >= 2:
        vals = list(a["int_effects"].values())
        effect = vals[0] - vals[1]
        lines.append(f"\n**因果效应**：{effect:+.2f}")
    return "\n".join(lines)


def reply_identify(a):
    lines = ["### 能不能从观测数据估出因果？\n"]
    for t, info in a["ident_info"].items():
        ok = "✅ 能" if info["identifiable"] else "❌ 不能"
        lines.append(f"**P({a['target']} | do({t}))**")
        lines.append(f"- 可识别：{ok}")
        lines.append(f"- 原因：{info['reason']}")
        if info["adjustment_set"]:
            lines.append(f"- 需控制：{info['adjustment_set']}")
        lines.append("")
    return "\n".join(lines)


def reply_counterfactual(a):
    t = a["target"]
    lines = [f"### 反事实推理\n"]
    lines.append(f"**事实**：执行 `{a['actions'][0]['type']}` "
                 f"→ `{t}` = {a['factual'][t]:.2f}")
    lines.append(f"**反事实**：如果当时执行 `{a['actions'][1]['type']}` "
                 f"→ `{t}` = {a['cf'][t]:.2f}")
    if a["factual"][t] > 0.5 and a["cf"][t] < 0.5:
        lines.append(
            f"\n**结论**：`{a['actions'][0]['type']}` 是 `{t}=1` 的必要原因。"
        )
    return "\n".join(lines)


def reply_plan(a):
    lines = [f"### 让 `{a['target']} = 1` 的最省力方案\n"]
    lines.append(f"- 搜索节点数：{a['plan']['n_expanded']}")
    cost = a["plan"]["cost"]
    lines.append(f"- 代价：{cost:.3f}" if cost < float("inf") else "- 代价：∞")
    if a["plan"]["path"]:
        lines.append(f"\n**最优干预路径**（{len(a['plan']['path'])} 步）：")
        for i, step in enumerate(a["plan"]["path"], 1):
            lines.append(f"{i}. `{dict(step)}`")
    else:
        lines.append("\n❌ 未找到可行路径。")
    return "\n".join(lines)


# ============================================================
# 主应答
# ============================================================
def respond(user_text: str) -> str:
    # 1. 识别场景
    scenario = parse_scenario(user_text)

    # 如果是"切换场景"，用上次的
    if scenario is None:
        scenario = st.session_state.current_scenario

    if scenario is None:
        return (
            "我还不确定你要分析哪个场景。试试这样说：\n\n"
            "- `分析医疗场景`\n"
            "- `看下服务器`\n"
            "- `帮我分析营销漏斗`"
        )

    st.session_state.current_scenario = scenario

    # 2. 跑分析（缓存）
    with st.spinner(f"正在分析「{scenario}」场景..."):
        try:
            a = analyze(scenario)
        except Exception as e:
            return f"⚠️ 分析出错：`{e}`"

    st.session_state.analysis = a

    # 3. 识别意图，生成回答
    intent = detect_intent(user_text)

    if intent == "all":
        parts = [
            f"## 📊 「{scenario}」场景分析\n",
            reply_structure(a),
            "\n---\n",
            reply_intervene(a),
            "\n---\n",
            reply_identify(a),
            "\n---\n",
            reply_counterfactual(a),
            "\n---\n",
            reply_plan(a),
        ]
        return "\n".join(parts)

    if intent == "intervene":
        return reply_intervene(a)
    if intent == "identify":
        return reply_identify(a)
    if intent == "counterfactual":
        return reply_counterfactual(a)
    if intent == "plan":
        return reply_plan(a)

    return reply_structure(a)


# ============================================================
# 输入框
# ============================================================
user_input = st.chat_input("说点什么... 例如：帮我分析医疗场景")

if user_input:
    # 显示用户消息
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # 生成回复
    answer = respond(user_input)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)