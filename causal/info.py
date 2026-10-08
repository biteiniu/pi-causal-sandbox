"""信息度量：熵、信息增益、基尼不纯度、互信息"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


def entropy(probs: List[float]) -> float:
    """香农熵 H(X) = -sum p_i log2 p_i"""
    p = np.array([x for x in probs if x > 1e-9])
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def binary_entropy(p: float) -> float:
    """二值熵，p 是正类概率"""
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def gini(vals: List[float], threshold: float = 0.5) -> float:
    """
    基尼不纯度 Gini(D) = 1 - sum p_k^2
    输入 0/1 标签，或概率值（>= threshold 视为 1）
    """
    if len(vals) == 0:
        return 0.0
    p = np.mean([1 if v >= threshold else 0 for v in vals])
    return float(1 - p**2 - (1 - p)**2)


def information_gain(
    data: List[Tuple[dict, dict]],
    action: dict,
    target: str = "Fallen_B",
    threshold: float = 0.5,
) -> float:
    """
    信息增益 IG(D, do(action)) = H(D) - sum_v (|D_v|/|D|) H(D_v)

    data: [(intervention_dict, sample_dict), ...]
    action: 待评估的动作
    target: 目标变量名
    """
    if len(data) < 2:
        return 0.0

    # 非干预动作不提供因果信息
    intervention_of_action = _action_to_intervention(action)
    if not intervention_of_action:
        return 0.0

    targets = [s.get(target, 0.0) for _, s in data]
    p1 = np.mean([1 if t > threshold else 0 for t in targets])
    H_D = binary_entropy(p1)

    partitions: Dict[tuple, List[float]] = {}
    action_keys = set(intervention_of_action.keys())

    for intervention, sample in data:
        key = tuple(
            sorted((k, v) for k, v in intervention.items() if k in action_keys)
        )
        partitions.setdefault(key, []).append(sample.get(target, 0.0))

    H_cond = 0.0
    n = len(data)
    for key, vals in partitions.items():
        w = len(vals) / n
        p = np.mean([1 if v > threshold else 0 for v in vals])
        H_cond += w * binary_entropy(p)

    return float(H_D - H_cond)


def mutual_information(x_vals: List, y_vals: List) -> float:
    """互信息 I(X;Y) = sum p(x,y) log2(p(x,y)/(p(x)p(y)))"""
    n = len(x_vals)
    if n == 0:
        return 0.0
    xy = Counter(zip(x_vals, y_vals))
    x = Counter(x_vals)
    y = Counter(y_vals)
    mi = 0.0
    for (xv, yv), c in xy.items():
        p_xy = c / n
        p_x = x[xv] / n
        p_y = y[yv] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return float(mi)


def _action_to_intervention(action: dict) -> dict:
    """动作 -> 干预字典（与 BlockWorld 保持一致）"""
    if action.get("type") == "push":
        return {"Force_A": action.get("force", 1.0)}
    elif action.get("type") == "observe":
        return {}
    elif action.get("type") == "wait":
        return {"Force_A": 0.0}
    return {}