"""NOTEARS：可微 DAG 学习"""

import numpy as np
import torch
import torch.nn as nn

from causal.constraints import dag_loss, project_to_dag


class NOTEARS(nn.Module):
    def __init__(self, n_vars: int, l1: float = 0.01, alpha: float = 1.0):
        super().__init__()
        self.n_vars = n_vars
        self.W = nn.Parameter(torch.zeros(n_vars, n_vars))
        self.l1 = l1
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.W

    def loss(self, X: torch.Tensor) -> torch.Tensor:
        return dag_loss(self.W, X, self.l1, self.alpha)

    def get_dag(self) -> np.ndarray:
        with torch.no_grad():
            W = self.W.detach().cpu().numpy()
        return project_to_dag(W)


def train_notears(
    data: np.ndarray,
    n_vars: int,
    epochs: int = 500,
    lr: float = 1e-3,
    l1: float = 0.01,
    alpha: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """训练 NOTEARS，返回 DAG 邻接矩阵"""
    # 标准化，避免尺度问题
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-6
    X_np = (data - mean) / std

    X = torch.tensor(X_np, dtype=torch.float32)
    model = NOTEARS(n_vars, l1, alpha)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = model.loss(X)
        loss.backward()
        opt.step()

        # 防止 W 发散
        with torch.no_grad():
            model.W.clamp_(-5.0, 5.0)

        losses.append(loss.item())

        if verbose and epoch % 100 == 0:
            print(f"  NOTEARS epoch {epoch:4d} | loss={loss.item():.4f}")

    return model.get_dag()