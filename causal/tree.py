"""与或树 + 代价复杂度剪枝"""

import numpy as np


class AndOrNode:
    def __init__(self, node_type: str, label: str, value: float = 0.0):
        self.type = node_type  # "and" | "or" | "leaf"
        self.label = label
        self.value = value
        self.children = []
        self.cost = {}
        self.pred_error = 0.0

    def is_leaf(self) -> bool:
        return self.type == "leaf" or not self.children

    def count_leaves(self) -> int:
        if self.is_leaf():
            return 1
        return sum(c.count_leaves() for c in self.children)

    def prediction_error(self) -> float:
        if self.is_leaf():
            return self.pred_error
        return sum(c.prediction_error() for c in self.children)

    def collapse_to_leaf(self):
        self.children = []
        self.type = "leaf"

    def f(self) -> float:
        """与或树代价"""
        if self.is_leaf():
            return self.value
        if self.type == "or":
            return min(
                self.cost.get(c.label, 0) + c.f() for c in self.children
            )
        if self.type == "and":
            return sum(
                self.cost.get(c.label, 0) + c.f() for c in self.children
            )
        return 0.0


def build_and_or_tree(scm, target: str, visited=None) -> AndOrNode:
    """
    把SCM转成与或树：
    - 目标变量 -> 或节点（多个父路径任选）
    - 父变量集合 -> 与节点（必须同时满足）
    """
    visited = visited or set()
    if target in visited:
        return AndOrNode("leaf", target)
    visited.add(target)

    parents = scm.variables[target].parents
    if not parents:
        return AndOrNode("leaf", target)

    root = AndOrNode("or", target)
    and_node = AndOrNode("and", f"parents_of_{target}")
    for p in parents:
        and_node.children.append(build_and_or_tree(scm, p, visited))
    root.children.append(and_node)
    return root


class CostComplexityPruner:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def cost_complexity(self, node: AndOrNode) -> float:
        """R_alpha(T) = R(T) + alpha * |T|"""
        return node.prediction_error() + self.alpha * node.count_leaves()

    def prune(self, node: AndOrNode) -> AndOrNode:
        if node.is_leaf():
            return node
        for child in node.children:
            self.prune(child)

        subtree_cost = self.cost_complexity(node)
        leaf_cost = node.prediction_error() + self.alpha * 1
        if leaf_cost <= subtree_cost:
            node.collapse_to_leaf()
        return node