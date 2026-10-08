"""反事实展开树 + 最小割剪枝"""

import numpy as np
import networkx as nx
from .tree import AndOrNode


class CounterfactualTree:
    """
    反事实展开：给定事实轨迹，枚举替代干预，展开成树。
    用最小割找最优保留/删除边界。
    """

    def __init__(self, scm, max_depth: int = 4, max_branch: int = 3):
        self.scm = scm
        self.max_depth = max_depth
        self.max_branch = max_branch
        self.root = None

    def expand(self, factual_trace: list, candidate_actions: list):
        """展开反事实树"""
        self.root = AndOrNode("or", "counterfactual_root")

        for i, action in enumerate(candidate_actions[: self.max_branch]):
            intervention = self._action_to_intervention(action)
            cf_result = self.scm.counterfactual(
                factual_trace, interventions=intervention
            )

            leaf = AndOrNode(
                "leaf", f"cf_{i}", cf_result.get("Fallen_B", 0.0)
            )
            # 预测误差 = 反事实结果与事实结果的差异
            leaf.pred_error = abs(
                cf_result.get("Fallen_B", 0.0)
                - factual_trace[-1].get("Fallen_B", 0.0)
            )
            self.root.children.append(leaf)

    def min_cut_prune(self, source: str = "s", sink: str = "t"):
        """
        最小割剪枝：
        源->节点 容量=信息增益，节点->汇 容量=复杂度代价
        最小割 = 最优剪枝边界
        """
        if self.root is None:
            return None

        G = nx.DiGraph()
        G.add_node(source)
        G.add_node(sink)

        def add_node(node, parent=None):
            nid = id(node)
            G.add_node(nid, label=node.label)

            if parent is not None:
                # 父->子边容量：预测误差越大，越应保留
                G.add_edge(
                    parent, nid, capacity=max(node.pred_error, 0.01)
                )

            if node.is_leaf():
                # 叶节点->汇：复杂度代价
                G.add_edge(nid, sink, capacity=node.pred_error + 0.01)

            for child in node.children:
                add_node(child, nid)

        add_node(self.root)
        G.add_edge(source, id(self.root), capacity=1.0)

        try:
            cut_value, partition = nx.minimum_cut(G, source, sink)
            return {"cut_value": cut_value, "partition": partition}
        except nx.NetworkXError:
            return None

    def _action_to_intervention(self, action: dict) -> dict:
        if action["type"] == "push":
            return {"Force_A": action.get("force", 1.0)}
        elif action["type"] == "observe":
            return {}
        elif action["type"] == "wait":
            return {"Force_A": 0.0}
        return {}