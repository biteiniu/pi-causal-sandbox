"""DAG-GNN：A 初始化为非零，让 MLP 先看到输入"""

import numpy as np
import torch
import torch.nn as nn

from causal.constraints import acyclicity_constraint, sparsity_loss, project_to_dag


class DAGGNN(nn.Module):
    def __init__(self, n_vars: int, hidden: int = 64, init_A: float = 0.5):
        super().__init__()
        self.n_vars = n_vars
        # 【关键】从 0.5 而非 0 开始，让梯度一开始就流通
        self.A = nn.Parameter(torch.ones(n_vars, n_vars) * init_A)
        self.register_buffer("diag_mask", 1.0 - torch.eye(n_vars))

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_vars, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_vars)
        ])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        A_eff = self.A * self.diag_mask
        outs = []
        for j in range(self.n_vars):
            masked = X * A_eff[:, j].unsqueeze(0)
            outs.append(self.mlps[j](masked))
        return torch.cat(outs, dim=1)

    def loss(self, X, l1=1e-4, alpha=0.1):
        X_hat = self.forward(X)
        recon = torch.mean((X - X_hat) ** 2)
        A_eff = self.A * self.diag_mask
        sp = sparsity_loss(A_eff, l1)
        h = acyclicity_constraint(A_eff)
        return recon + sp + alpha * h

    def get_raw_A(self) -> np.ndarray:
        with torch.no_grad():
            A = self.A.detach().cpu().numpy().copy()
        np.fill_diagonal(A, 0.0)
        return A

    def get_normalized_A(self) -> np.ndarray:
        A = self.get_raw_A()
        a_max = np.abs(A).max()
        if a_max > 1e-9:
            A = A / a_max
        return A

    def get_dag(self, threshold: float = 0.3) -> np.ndarray:
        A = self.get_normalized_A()
        return project_to_dag(A, threshold=threshold)


def train_dag_gnn(
    data: np.ndarray,
    n_vars: int,
    epochs: int = 5000,
    lr: float = 1e-3,
    l1: float = 1e-4,
    alpha: float = 0.1,
    hidden: int = 64,
    init_A: float = 0.5,
    verbose: bool = True,
) -> tuple:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-6
    X_np = (data - mean) / std

    X = torch.tensor(X_np, dtype=torch.float32)
    model = DAGGNN(n_vars, hidden=hidden, init_A=init_A)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(epochs):
        opt.zero_grad()
        loss = model.loss(X, l1=l1, alpha=alpha)
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.A.clamp_(-10.0, 10.0)

        if verbose and epoch % 500 == 0:
            with torch.no_grad():
                a_abs = (model.A * model.diag_mask).abs()
                a_norm = a_abs.sum().item()
                a_max = a_abs.max().item()
                a_min_nonzero = a_abs[a_abs > 1e-6].min().item() if (a_abs > 1e-6).any() else 0.0
            print(
                f"  epoch {epoch:4d} | loss={loss.item():.4f} | "
                f"|A|_1={a_norm:.4f} max={a_max:.4f} min+={a_min_nonzero:.4f}"
            )

    return model.get_raw_A(), model.get_normalized_A(), model.get_dag()