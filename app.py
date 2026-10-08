"""
π-因果沙盒 · 可视化界面
运行：streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

# 项目模块
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
# 页面配置
# ============================================================
st.set_page_config(
    page_title="π-因果沙盒",
    page_icon="π",
    layout="wide",
)

st.title("π-因果沙盒")
st.caption("主动干预 → 发现因果 → 预测干预 → 反事实 → 规划最优方案")

# ============================================================
# 场景定义
# ============================================================
SCENARIOS = {
    "分叉+菱形 (BlockWorldFork)": {
        "class": BlockWorldFork,
        "kwargs": {},
        "target": "Fallen_C",
        "desc": "A 同时影响 B 和 C，B 也影响 C。经典菱形结构。",
    },
    "2块链 (BlockWorld)": {
        "class": BlockWorld,
        "kwargs": {},
        "target": "Fallen_B",
        "desc": "最简单：A → B。",
    },
    "3块链 (BlockWorld3)": {
        "class": BlockWorld3,
        "kwargs": {},
        "target": "Fallen_C",
        "desc": "A → B → C，链式传递。",
    },
    "碰撞 (Collider)": {
        "class": BlockWorldCollider,
        "kwargs": {"noise": 0.02},
        "target": "Fallen_C",
        "desc": "A → C ← B。控制 C 会引入伪相关。",
    },
    "混杂 (Confounder)": {
        "class": BlockWorldConfounder,
        "kwargs": {"noise": 0.02},
        "target": "Fallen_B",
        "desc": "U → A, U → B。不控制 U 会把相关当因果。",
    },
    "连续链 (ContinuousChain)": {
        "class": ContinuousChain,
        "kwargs": {"noise": 0.1},
        "target": "Fallen_B",
        "desc": "线性高斯连续变量。",
    },
}

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("① 选择场景")

    scenario_name = st.selectbox(
        "场景",
        list(SCENARIOS.keys()),
        index=0,
    )
    scenario = SCENARIOS[scenario_name]
    st.info(scenario["desc"])

    st.header("② 参数")

    n_samples = st.slider(
        "每次干预的采样数",
        min_value=100, max_value=1000, value=300, step=50,
        help="越大越准，但越慢。建议 300~500。",
    )

    edge_thr = st.slider(
        "边判定阈值",
        min_value=0.05, max_value=0.5, value=0.15, step=0.05,
        help="效应超过这个值才认为是真边。",
    )

    use_continuous = scenario_name.startswith("连续")
    readout = "continuous" if use_continuous else "binary"

    st.header("③ 执行")

    run_button = st.button(
        "🚀 运行完整流程",
        use_container_width=True,
        type="primary",
    )

    st.divider()
    st.caption("提示：先选场景 → 调参数 → 点运行。")

# ============================================================
# 主流程
# ============================================================
if run_button:
    np.random.seed(42)

    # 构造环境
    env = scenario["class"](**scenario["kwargs"])
    var_names = list(env.scm.variables.keys())
    true_edges = set()
    for v in var_names:
        for p in env.scm.variables[v].parents:
            true_edges.add((p, v))

    target = scenario["target"]
    actions = env.actions

    # 进度条
    progress = st.progress(0, text="准备中...")

    # ---------------- Step 1: 发现 ----------------
    progress.progress(10, text="Step 1/5 · 正在主动干预发现因果结构...")
    _, direct = interventional_discovery(
        env.scm, var_names,
        n_samples=n_samples,
        edge_threshold=edge_thr,
        residual_threshold=edge_thr,
        readout=readout,
        verbose=False,
    )
    result = evaluate_against_true(direct, true_edges, var_names)

    # ---------------- Step 2: 贝叶斯 ----------------
    progress.progress(35, text="Step 2/5 · 拟合贝叶斯网络...")
    structure = {v: [] for v in var_names}
    for (u, v) in direct:
        structure[v].append(u)

    bayes = BayesianSCM(structure, var_names)
    data = []
    for action in actions:
        for _ in range(max(30, n_samples // 3)):
            env.reset()
            data.append(env.do(action))
    bayes.fit_cpt(data)

    # 干预效果
    if len(actions) >= 2:
        i0 = env._action_to_intervention(actions[0])
        i1 = env._action_to_intervention(actions[1])
        p1 = bayes.query(target, interventions=i0)
        p0 = bayes.query(target, interventions=i1)
    else:
        p1, p0 = None, None

    # ---------------- Step 3: 可识别性 ----------------
    progress.progress(60, text="Step 3/5 · 后门准则可识别性检查...")
    adj = edges_to_adjacency(direct, var_names)
    checker = IdentifiabilityChecker(adj, var_names)

    # 找出第一个非目标变量作为 treatment
    treatments = [v for v in var_names if v != target]
    ident_results = []
    for t in treatments[:3]:
        info = checker.is_identifiable(t, target)
        ident_results.append((t, info))

    # ---------------- Step 4: 反事实 ----------------
    progress.progress(80, text="Step 4/5 · 反事实推理...")
    env.reset()
    factual = env.do(actions[0])
    cf = env.counterfactual(actions[1])

    # ---------------- Step 5: A* ----------------
    progress.progress(95, text="Step 5/5 · A* 规划最优干预...")
    init_state = env.reset()
    plan = astar_intervention(
        env, init_state, bayes_scm=bayes,
        target=target, max_depth=4, max_nodes=200,
    )

    progress.progress(100, text="完成")
    progress.empty()

    # ============================================================
    # 结果展示
    # ============================================================
    st.success(f"完成！运行环境：{scenario_name}")

    # ---- 顶部指标卡 ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1 分数", f"{result['f1']:.3f}")
    col2.metric("真实边数", result["n_true"])
    col3.metric("发现边数", result["n_pred"])
    col4.metric("假阳性", len(result["false_positives"]))

    # ---- Tab 切换 ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["1️⃣ 因果结构", "2️⃣ 干预预测", "3️⃣ 可识别性",
         "4️⃣ 反事实", "5️⃣ A* 规划"]
    )

    # ---------- Tab 1 ----------
    with tab1:
        st.subheader("发现的因果结构")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**✓ 发现的边**")
            rows = []
            for (u, v), w in sorted(direct.items(), key=lambda x: -x[1]):
                tag = "真边" if (u, v) in true_edges else "❌ 假边"
                rows.append({
                    "原因": u,
                    "结果": v,
                    "效应强度": f"{w:.3f}",
                    "判定": tag,
                })
            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("算法没有发现任何边。")

        with col_b:
            st.markdown("**真实结构（已知但算法不知道）**")
            true_rows = [{"原因": u, "结果": v} for u, v in sorted(true_edges)]
            st.dataframe(
                pd.DataFrame(true_rows),
                use_container_width=True,
                hide_index=True,
            )

        # 画图
        if direct:
            st.markdown("**边效应强度对比**")
            fig, ax = plt.subplots(figsize=(8, 4))
            edges_sorted = sorted(direct.items(), key=lambda x: -x[1])
            labels = [f"{u} → {v}" for (u, v), _ in edges_sorted]
            values = [w for _, w in edges_sorted]
            colors = [
                "steelblue" if (u, v) in true_edges else "lightcoral"
                for (u, v), _ in edges_sorted
            ]
            bars = ax.barh(range(len(labels)), values, color=colors)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.axvline(edge_thr, color="red", ls="--", lw=1,
                       label=f"阈值 {edge_thr}")
            ax.set_xlabel("效应强度")
            ax.invert_yaxis()
            ax.legend()
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ---------- Tab 2 ----------
    with tab2:
        st.subheader(f"干预对 {target} 的预测效果")

        if p1 is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric(f"P({target} | do({actions[0]['type']}))", f"{p1:.2f}")
            col2.metric(f"P({target} | do({actions[1]['type']}))", f"{p0:.2f}")
            col3.metric("因果效应", f"{p1 - p0:+.2f}")

            st.markdown("**图：干预效果对比**")
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(
                [actions[0]["type"], actions[1]["type"]],
                [p1, p0],
                color=["steelblue", "lightcoral"],
            )
            for bar, v in zip(bars, [p1, p0]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=12,
                )
            ax.set_ylabel(f"P({target} = 1)")
            ax.set_ylim(0, 1.15)
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.caption(f"贝叶斯网络 CPT 条目数：{len(bayes.cpt)}")

    # ---------- Tab 3 ----------
    with tab3:
        st.subheader("后门准则可识别性检查")
        st.caption("问：能不能从观测数据估计 P(目标 | do(干预))？")

        rows = []
        for t, info in ident_results:
            rows.append({
                "干预变量": t,
                "目标": target,
                "可识别": "✅" if info["identifiable"] else "❌",
                "原因": info["reason"],
                "调整集": ", ".join(info["adjustment_set"])
                          if info["adjustment_set"] else "（空）",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "**解读**：如果可识别，说明你能只靠观测数据算因果效应（控制调整集即可）；"
            "如果不可识别，说明存在无法阻断的后门路径。"
        )

    # ---------- Tab 4 ----------
    with tab4:
        st.subheader("反事实推理")
        st.markdown(
            f"**事实**：执行了 `{actions[0]['type']}` → `{target} = {factual[target]:.2f}`"
        )
        st.markdown(
            f"**反事实**：如果当时执行 `{actions[1]['type']}` "
            f"→ `{target} = {cf[target]:.2f}`"
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            ["事实", "反事实"],
            [factual[target], cf[target]],
            color=["steelblue", "orange"],
        )
        for bar, v in zip(bars, [factual[target], cf[target]]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=12,
            )
        ax.set_ylabel(f"{target}")
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if factual[target] > 0.5 and cf[target] < 0.5:
            st.success(
                f"结论：`{actions[0]['type']}` 是 `{target}=1` 的必要原因。"
            )

    # ---------- Tab 5 ----------
    with tab5:
        st.subheader("A* 最优干预规划")
        st.markdown(f"**目标**：让 `{target} = 1`")

        col1, col2 = st.columns(2)
        col1.metric("搜索节点数", plan["n_expanded"])
        col2.metric(
            "代价",
            f"{plan['cost']:.3f}" if plan["cost"] < float("inf") else "∞",
        )

        if plan["path"]:
            st.success(f"找到路径（{len(plan['path'])} 步）")
            for i, step in enumerate(plan["path"], 1):
                st.markdown(f"**{i}.** `{dict(step)}`")
        else:
            st.error("未找到可行路径。可能目标太难，或深度/节点限制太低。")

# ============================================================
# 首次提示
# ============================================================
else:
    st.info(
        "👈 **从左侧开始**：\n"
        "1. 选一个场景\n"
        "2. 调参数（默认就行）\n"
        "3. 点 **🚀 运行完整流程**\n\n"
        "几秒后你会看到 5 个标签页：\n"
        "- 因果结构（算法发现了什么）\n"
        "- 干预预测（do(X) 的效果）\n"
        "- 可识别性（能不能只靠观测数据算）\n"
        "- 反事实（如果当时…会怎样）\n"
        "- A* 规划（最省力的方案）"
    )