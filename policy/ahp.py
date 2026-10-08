"""AHP：多准则权重 + 一致性检验 + 元学习修正"""

import numpy as np
from typing import List, Dict, Optional


# 随机一致性指标 RI（n=1..10）
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


class AHPWeights:
    """
    双轨AHP：
    - 先验矩阵 A0：专家设定
    - 学习矩阵 A_learned：从奖励反馈中元学习
    最终 A = (1-lam) * A0 + lam * A_learned
    """

    def __init__(self, lam: float = 0.3, lr: float = 0.01):
        self.criteria = [
            "info_gain", "goal_reward", "cost", "safety", "explainability"
        ]
        self.n = len(self.criteria)
        self.lam = lam
        self.lr = lr

        # 先验判断矩阵（准则间相对重要性）
        self.A0 = np.array([
            [1,   2,   3,   4,   5],
            [1/2, 1,   2,   3,   4],
            [1/3, 1/2, 1,   2,   3],
            [1/4, 1/3, 1/2, 1,   2],
            [1/5, 1/4, 1/3, 1/2, 1],
        ])
        self.A_learned = self.A0.copy()

        # 初始化占位
        self.lambda_max = 0.0
        self.CR = 0.0

        # 先算权重（内部会更新 lambda_max），再做一致性检验
        self.w = self._compute_weights(self._combined_matrix())
        self.consistency_check()

    # ---------- 内部 ----------

    def _combined_matrix(self) -> np.ndarray:
        """先验 + 学习矩阵融合，并强制互反 A_ij = 1/A_ji"""
        A = (1 - self.lam) * self.A0 + self.lam * self.A_learned
        # 互反化：A = (A + 1/A.T) / 2
        A = (A + 1.0 / np.maximum(A.T, 1e-6)) / 2
        np.fill_diagonal(A, 1.0)
        return A

    def _compute_weights(self, A: np.ndarray) -> np.ndarray:
        """主特征向量作为权重（Aw = lambda_max * w）"""
        eigvals, eigvecs = np.linalg.eig(A)
        idx = int(np.argmax(eigvals.real))
        w = np.abs(eigvecs[:, idx].real)
        self.lambda_max = float(eigvals[idx].real)
        return w / w.sum()

    # ---------- 对外 ----------

    def consistency_check(self) -> tuple:
        """一致性检验：CI = (lambda_max - n)/(n-1), CR = CI/RI"""
        CI = (self.lambda_max - self.n) / (self.n - 1)
        RI = RI_TABLE.get(self.n, 1.12)
        self.CR = CI / RI if RI > 0 else 0.0
        return self.CR, self.CR < 0.1

    def score(self, action: dict, causal_state: dict, pi) -> float:
        """加权评分 S_i = sum_j w_j * x_ij"""
        x = self._feature_vector(action, causal_state, pi)
        self.w = self._compute_weights(self._combined_matrix())
        return float(np.dot(self.w, x))

    def _feature_vector(self, action: dict, causal_state: dict, pi) -> np.ndarray:
        """构造准则特征向量（归一化到合理范围）"""
        atype = action.get("type", "observe")

        # 1) 信息增益
        ig = float(pi._info_gain(action, causal_state))
        ig = np.clip(ig, 0.0, 1.0)

        # 2) 任务奖励
        gr = float(pi._goal_reward(action, causal_state))
        gr = np.clip(gr, 0.0, 1.0)

        # 3) 代价（越低越好）
        cost = -0.1 if atype == "push" else 0.0
        cost_norm = 1.0 + cost  # push -> 0.9, 其他 -> 1.0

        # 4) 安全性
        safety = 1.0 if atype == "observe" else (0.7 if atype == "wait" else 0.4)

        # 5) 可解释性
        explain = 1.0 if atype in ("observe", "wait") else 0.3

        return np.array([ig, gr, cost_norm, safety, explain])

    def update_from_reward(self, action: dict, reward: float, causal_state: dict, pi):
        """元学习：按梯度微调 A_learned"""
        x = self._feature_vector(action, causal_state, pi)
        pred = float(np.dot(self.w, x))
        err = reward - pred

        grad = err * x
        self.A_learned += self.lr * np.outer(grad, np.ones_like(grad))
        self.A_learned = np.clip(self.A_learned, 0.1, 10.0)
        np.fill_diagonal(self.A_learned, 1.0)

    def report(self) -> dict:
        return {
            "criteria": self.criteria,
            "weights": dict(zip(self.criteria, self.w.round(4).tolist())),
            "lambda_max": round(self.lambda_max, 4),
            "CR": round(self.CR, 4),
            "consistent": self.CR < 0.1,
        }