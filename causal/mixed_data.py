"""混合数据：观测数据 + 干预数据联合训练"""

import numpy as np
import torch


def mixed_likelihood(
    W: torch.Tensor,
    X_obs: torch.Tensor,
    X_int: torch.Tensor,
    int_mask: torch.Tensor,
    n_obs: int,
    n_int: int,
) -> torch.Tensor:
    """
    观测数据：X = XW + noise
    干预数据：对被干预变量，切断父节点，用干预值替代

    int_mask: (n_int, n_vars)，1表示该样本该变量被干预
    """
    # 观测似然
    recon_obs = torch.mean((X_obs - X_obs @ W) ** 2)

    # 干预似然：对被干预变量，只保留非干预变量的重构
    X_int_recon = X_int @ W
    # 干预位置用真实值替代（do-calculus：切断父节点）
    X_int_recon = X_int_recon * (1 - int_mask) + X_int * int_mask
    recon_int = torch.mean((X_int - X_int_recon) ** 2)

    # 加权联合
    w_obs = n_obs / (n_obs + n_int)
    w_int = n_int / (n_obs + n_int)
    return w_obs * recon_obs + w_int * recon_int


class MixedDataLearner:
    """混合数据DAG学习器"""

    def __init__(self, n_vars: int):
        self.n_vars = n_vars
        self.obs_buffer = []
        self.int_buffer = []

    def add_observational(self, sample: dict, variables: list):
        self.obs_buffer.append([sample.get(v, 0.0) for v in variables])

    def add_interventional(self, intervention: dict, sample: dict, variables: list):
        row = [sample.get(v, 0.0) for v in variables]
        mask = [1.0 if v in intervention else 0.0 for v in variables]
        self.int_buffer.append((row, mask))

    def to_tensors(self):
        X_obs = torch.tensor(self.obs_buffer, dtype=torch.float32)
        if self.int_buffer:
            rows = [r for r, _ in self.int_buffer]
            masks = [m for _, m in self.int_buffer]
            X_int = torch.tensor(rows, dtype=torch.float32)
            M_int = torch.tensor(masks, dtype=torch.float32)
        else:
            X_int = torch.zeros(0, self.n_vars)
            M_int = torch.zeros(0, self.n_vars)
        return X_obs, X_int, M_int
        