"""Linear-Gaussian SCM: OLS fit + exact do() query via linear propagation."""

import numpy as np
from typing import Dict, List


class LinearGaussianSCM:
    """
    Continuous SCM where P(X_j | Pa(X_j)) = N(a_j · Pa_j + b_j, σ_j²).
    Coefficients are fit by OLS. do() query is exact via topological propagation.
    """

    def __init__(self, structure: Dict[str, List[str]], var_names: List[str]):
        self.structure = structure
        self.var_names = var_names
        self.coeffs: Dict[str, Dict[str, float]] = {}
        self.sigma: Dict[str, float] = {}

    def fit(self, data: List[dict]):
        X = np.array([[s[v] for v in self.var_names] for s in data])
        for j, var in enumerate(self.var_names):
            parents = self.structure.get(var, [])
            if not parents:
                self.coeffs[var] = {"_intercept": float(X[:, j].mean())}
                self.sigma[var] = float(X[:, j].std() + 1e-6)
                continue

            pidx = [self.var_names.index(p) for p in parents]
            Xp = X[:, pidx]
            y = X[:, j]
            A = np.hstack([Xp, np.ones((len(Xp), 1))])
            theta, *_ = np.linalg.lstsq(A, y, rcond=None)

            coef = {p: float(theta[k]) for k, p in enumerate(parents)}
            coef["_intercept"] = float(theta[-1])
            self.coeffs[var] = coef

            resid = y - A @ theta
            self.sigma[var] = float(resid.std() + 1e-6)

    def query_mean(self, target: str, interventions: Dict[str, float] = None) -> float:
        """E[target | do(interventions)] via topological linear propagation."""
        interventions = interventions or {}
        order = self._topo_order()
        values = {}
        for var in order:
            if var in interventions:
                values[var] = interventions[var]
                continue
            coef = self.coeffs.get(var, {})
            mean = coef.get("_intercept", 0.0)
            for p in self.structure.get(var, []):
                mean += coef.get(p, 0.0) * values.get(p, 0.0)
            values[var] = mean
        return values[target]

    def _topo_order(self):
        visited, order = set(), []

        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            for p in self.structure.get(v, []):
                dfs(p)
            order.append(v)

        for v in self.var_names:
            dfs(v)
        return order

    def predict(self, sample: dict, target: str) -> float:
        """Predict target from observed parents (no intervention)."""
        coef = self.coeffs.get(target, {})
        mean = coef.get("_intercept", 0.0)
        for p in self.structure.get(target, []):
            mean += coef.get(p, 0.0) * sample.get(p, 0.0)
        return mean