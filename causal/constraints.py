"""DAG学习的约束：稀疏性 + 无环性"""

import numpy as np
import torch


def sparsity_loss(W: torch.Tensor, l1: float = 0.01) -> torch.Tensor:
    """L1稀疏：鼓励因果图稀疏"""
    return l1 * torch.sum(torch.abs(W))


def acyclicity_constraint(W: torch.Tensor) -> torch.Tensor:
    """
    NOTEARS无环约束：h(W) = tr(e^{W∘W}) - d
    当且仅当 W 对应 DAG 时，h(W) = 0
    """
    d = W.shape[0]
    W_sq = W * W
    E = torch.matrix_exp(W_sq)
    h = torch.trace(E) - d
    return h


def dag_loss(
    W: torch.Tensor,
    data: torch.Tensor,
    l1: float = 0.01,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    总损失：重构误差 + 稀疏 + 无环
    data: (n_samples, n_vars)
    W:    (n_vars, n_vars) 加权邻接矩阵
    """
    recon = torch.mean((data - data @ W) ** 2)
    sp = sparsity_loss(W, l1)
    h = acyclicity_constraint(W)
    return recon + sp + alpha * h


def _has_cycle(adj: np.ndarray) -> bool:
    """DFS 检测有向环"""
    n = adj.shape[0]
    visited = [0] * n  # 0=未访问, 1=访问中, 2=已完成

    def dfs(u):
        visited[u] = 1
        for v in range(n):
            if abs(adj[u, v]) > 1e-6:
                if visited[v] == 1:
                    return True
                if visited[v] == 0 and dfs(v):
                    return True
        visited[u] = 2
        return False

    return any(visited[i] == 0 and dfs(i) for i in range(n))


def project_to_dag(W: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    后处理：把近似 DAG 投影为严格 DAG。
    按权重绝对值降序加入边，遇到成环就跳过。
    """
    W = W.copy()
    np.fill_diagonal(W, 0)

    edges = [
        (i, j, abs(W[i, j]))
        for i in range(W.shape[0])
        for j in range(W.shape[1])
        if i != j and abs(W[i, j]) > threshold
    ]
    edges.sort(key=lambda x: -x[2])

    adj = np.zeros_like(W)
    for i, j, w in edges:
        adj[i, j] = W[i, j]
        if _has_cycle(adj):
            adj[i, j] = 0.0
    return adj