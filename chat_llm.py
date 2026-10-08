"""
π-因果沙盒 · LLM 对话版

支持三类 LLM：
  - DeepSeek（推荐，便宜，中文好）
  - OpenAI
  - Ollama 本地（免费，无需 API key）

运行：streamlit run chat_llm.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import streamlit as st
import numpy as np

# ============================================================
# 因果沙盒工具
# ============================================================
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


SCENARIOS = {
    "链":   {"class": BlockWorld3, "kwargs": {}, "target": "Fallen_C",
             "desc": "A → B → C 链式因果"},
    "分叉": {"class": BlockWorldFork, "kwargs": {}, "target": "Fallen_C",
             "desc": "A 影响 B 和 C，B 也影响 C"},
    "碰撞": {"class": BlockWorldCollider, "kwargs": {"noise": 0.02},
             "target": "Fallen_C", "desc": "A → C ← B"},
    "混杂": {"class": BlockWorldConfounder, "kwargs": {"noise": 0.02},
             "target": "Fallen_B", "desc": "U → A, U → B"},
    "连续": {"class": ContinuousChain, "kwargs": {"noise": 0.1},
             "target": "Fallen_B", "desc": "线性高斯连续"},
    "简单": {"class": BlockWorld, "kwargs": {}, "target": "Fallen_B",
             "desc": "A → B"},
}


def run_causal_analysis(scenario_name: str, task: str = "all") -> dict:
    """LLM 调用的因果分析工具。"""
    if scenario_name not in SCENARIOS:
        return {"error": f"未知场景 {scenario_name}。可选：{list(SCENARIOS.keys())}"}

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
        int_effects[a["type"]] = float(
            bayes.query(target, interventions=intv)
        )

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
        "description": cfg["desc"],
        "variables": var_names,
        "target": target,
        "true_edges": [list(e) for e in sorted(true_edges)],
        "discovered_edges": [
            {"cause": u, "effect": v, "strength": round(w, 3)}
            for (u, v), w in sorted(direct.items(), key=lambda x: -x[1])
        ],
        "metrics": {
            "f1": round(result["f1"], 3),
            "precision": round(result["precision"], 3),
            "recall": round(result["recall"], 3),
            "n_true": result["n_true"],
            "n_pred": result["n_pred"],
            "false_positives": [list(e) for e in result["false_positives"]],
        },
        "intervention_effects": int_effects,
        "identifiability": ident_info,
        "counterfactual": {
            "factual_action": actions[0]["type"],
            "factual_result": round(float(factual[target]), 3),
            "counterfactual_action": actions[1]["type"],
            "counterfactual_result": round(float(cf[target]), 3),
        },
        "optimal_plan": {
            "n_expanded": plan["n_expanded"],
            "cost": (None if plan["cost"] == float("inf")
                     else round(plan["cost"], 3)),
            "path": [dict(s) for s in plan["path"]] if plan["path"] else None,
        },
    }


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="π-因果沙盒 · LLM 对话",
    page_icon="💬",
    layout="wide",
)
st.title("π-因果沙盒 · LLM 对话版")

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("⚙️ LLM 设置")

    provider = st.selectbox(
        "服务商",
        ["DeepSeek", "OpenAI", "Ollama 本地"],
    )

    if provider == "DeepSeek":
        default_base = "https://api.deepseek.com/v1"
        default_model = "deepseek-chat"
        api_key = st.text_input(
            "DeepSeek API Key", type="password",
            help="去 platform.deepseek.com 申请",
        )
        base_url = default_base
        model = default_model
        st.caption("推荐。价格低，中文好。")

    elif provider == "OpenAI":
        default_base = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"
        api_key = st.text_input("OpenAI API Key", type="password")
        base_url = default_base
        model = st.selectbox(
            "模型", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        )

    else:  # Ollama
        api_key = "ollama"
        base_url = st.text_input(
            "Ollama 地址", value="http://localhost:11434/v1"
        )
        model = st.text_input("模型名", value="qwen2.5:7b")
        st.caption("无需 API key。需先本地安装 Ollama。")

    st.divider()
    st.caption("**不会配置？** 选 Ollama 本地，但先装 ollama.com")

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# OpenAI 客户端
# ============================================================
try:
    from openai import OpenAI
except ImportError:
    st.error("缺少 `openai` 库。请运行：`pip install openai`")
    st.stop()


def make_client():
    if provider == "Ollama 本地":
        return OpenAI(api_key="ollama", base_url=base_url)
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


client = make_client()


# ============================================================
# System Prompt（聊天 + 因果，口语化输出）
# ============================================================
SYSTEM_PROMPT = """你是一个通用 AI 助手，能聊任何话题，也懂得因果分析。

**默认行为**：
- 正常聊天：问候、闲聊、情感、生活、知识问答
- 用户说什么就聊什么，别强行扯因果

**什么时候调用 `run_causal_analysis` 工具**：
- 用户说"分析X"、"跑一下X"、"看看X"
- 用户问因果问题："谁导致谁"、"如果X会怎样"、"X对Y有什么影响"
- 用户说"发现结构"、"预测干预"、"反事实"、"最优方案"

**场景映射**：
- 医疗 / 药物 / 给药 → "链"
- 营销 / 广告 / 漏斗 → "分叉"
- 服务器 / 缓存 → "链"
- 分叉 / 菱形 / fork → "分叉"
- 碰撞 / collider → "碰撞"
- 混杂 / confounder → "混杂"
- 连续 / 线性 → "连续"

**⚠️ 最重要的规则——拿到工具结果后怎么回复**：

不要罗列 JSON 字段！不要把数字堆在一起！
用**聊天口吻**把关键发现"讲"给用户听。

**示例（正确风格）**：

用户：`帮我分析医疗场景`

你（调用工具后，用聊天口吻回复）：
> 好的，我跑了医疗场景（药物 → 血药浓度 → 缓解 这条链）。
>
> 算法自己找出了两条因果边，全都对：药物确实影响血药浓度，血药浓度影响缓解。准确率满分。
>
> 关键发现是——**给药时缓解概率 0.70，不给药是 0.00**。也就是说因果效应非常强，+0.70。
>
> 如果反事实地问"当时不给药会怎样"，缓解概率会直接掉到 0。所以在这个场景里，**给药是缓解的必要原因**。
>
> 最省力的方案也很简单：直接给药，一步就够。

**示例（错误风格，不要这样）**：
> 场景：链
> 变量：['Drug', 'Level', 'Relief']
> 发现的边：
>   Drug → Level 强度 1.00 ✓
>   Level → Relief 强度 0.93 ✓
> F1 = 1.000
> ...

**风格要求**：
- 中文回答，自然、口语化、像朋友聊天
- 聊因果时，用讲故事的方式讲数字，不要让用户看表格
- 适当加粗关键结论
- 普通聊天就正常聊，别端着
- 被问奇怪的话（比如"大河向东流"）正常接梗

**示例（普通聊天）**：
- "你好" → "你好呀，有什么想聊的？"
- "大河向东流" → "天上的星星参北斗～"
- "什么是因果推断" → 直接解释
"""


# ============================================================
# 工具定义
# ============================================================
TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_causal_analysis",
        "description": "运行因果沙盒分析，返回因果结构、干预效果、可识别性、反事实、最优规划。仅在用户明确询问因果问题时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "enum": list(SCENARIOS.keys()),
                    "description": "分析哪个因果场景",
                },
                "task": {
                    "type": "string",
                    "enum": ["all", "structure", "intervene",
                             "identify", "counterfactual", "plan"],
                    "description": "只做某一步，默认 all",
                },
            },
            "required": ["scenario"],
        },
    },
}]


# ============================================================
# 聊天状态
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 首次提示
if not st.session_state.messages:
    with st.chat_message("assistant"):
        if provider == "Ollama 本地":
            st.markdown(
                "你好！我可以陪你聊天，也能做因果分析。\n\n"
                "**随便说点什么吧：**\n"
                "- `你好`\n"
                "- `大河向东流`\n"
                "- `什么是因果推断`\n"
                "- `帮我分析医疗场景`\n\n"
                "⚠️ 使用前确认本地 Ollama 已启动。"
            )
        elif not api_key:
            st.warning(
                f"⚠️ 请先在左侧填入 {provider} 的 API Key，然后开始聊天。"
            )
        else:
            st.markdown(
                "你好！我可以陪你聊天，也能做因果分析。\n\n"
                "**随便说点什么吧：**\n"
                "- `你好`\n"
                "- `大河向东流`\n"
                "- `什么是因果推断`\n"
                "- `帮我分析医疗场景`"
            )

# 展示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tool_calls" in msg:
            with st.expander("🔧 调用的工具"):
                for tc in msg["tool_calls"]:
                    st.code(
                        f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})",
                        language="python",
                    )


# ============================================================
# 处理用户输入
# ============================================================
if prompt := st.chat_input("说点什么..."):
    if client is None:
        st.error(f"请先在左侧填入 {provider} 的 API Key。")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        tool_calls_made = []

        with st.spinner("思考中..."):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=api_messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.7,
                )
                msg = response.choices[0].message

                if msg.tool_calls:
                    api_messages.append(msg)

                    for tool_call in msg.tool_calls:
                        fn_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        tool_calls_made.append({"name": fn_name, "args": args})

                        if fn_name == "run_causal_analysis":
                            result = run_causal_analysis(
                                args.get("scenario", "链"),
                                args.get("task", "all"),
                            )
                        else:
                            result = {"error": f"未知工具 {fn_name}"}

                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        })

                    response2 = client.chat.completions.create(
                        model=model,
                        messages=api_messages,
                        temperature=0.7,
                    )
                    final_text = response2.choices[0].message.content

                else:
                    final_text = msg.content

                placeholder.markdown(final_text)

                saved = {"role": "assistant", "content": final_text}
                if tool_calls_made:
                    saved["tool_calls"] = tool_calls_made
                st.session_state.messages.append(saved)

                if tool_calls_made:
                    with st.expander("🔧 调用的工具"):
                        for tc in tool_calls_made:
                            st.code(
                                f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})",
                                language="python",
                            )

            except Exception as e:
                st.error(f"出错：{e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ 出错：{e}",
                })