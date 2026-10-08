"""贝叶斯SCM：CPT估计 + 精确推理（枚举法，适合小规模）"""

import numpy as np
from collections import defaultdict
from itertools import product
from typing import Dict, List


def _bin(x) -> int:
    """与真实 SCM 一致的阈值：>= 0.5 记为 1"""
    return int(x >= 0.5)


class BayesianSCM:
    def __init__(self, structure: Dict[str, List[str]], var_names: List[str]):
        self.structure = structure
        self.var_names = var_names
        self.cpt: Dict[tuple, Dict[int, float]] = {}
        self.marginals: Dict[str, Dict[int, float]] = {}

    def fit_cpt(self, data: List[dict]):
        counts = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        var_counts = defaultdict(lambda: defaultdict(int))

        for sample in data:
            for var, parents in self.structure.items():
                parent_key = tuple(_bin(sample.get(p, 0)) for p in parents)
                value_key = _bin(sample.get(var, 0))
                counts[(var, parent_key)][value_key] += 1
                totals[(var, parent_key)] += 1
                var_counts[var][value_key] += 1

        n = len(data) if data else 1
        for var in self.structure:
            self.marginals[var] = {
                0: var_counts[var].get(0, 0) / n,
                1: var_counts[var].get(1, 0) / n,
            }

        for key, c in counts.items():
            total = totals[key]
            self.cpt[key] = {
                0: c.get(0, 0) / max(total, 1),
                1: c.get(1, 0) / max(total, 1),
            }

    def query(
        self,
        target: str,
        evidence: Dict[str, float] = None,
        interventions: Dict[str, float] = None,
    ) -> float:
        evidence = evidence or {}
        interventions = interventions or {}

        fixed = {}
        for k, v in evidence.items():
            fixed[k] = _bin(v)
        for k, v in interventions.items():
            fixed[k] = _bin(v)

        free_vars = [v for v in self.var_names if v not in fixed]

        total = 0.0
        target_sum = 0.0

        for combo in product([0, 1], repeat=len(free_vars)):
            assignment = dict(zip(free_vars, combo))
            assignment.update(fixed)

            p = self._joint_prob(assignment, self.structure)
            total += p
            if assignment.get(target, 0) > 0.5:
                target_sum += p

        return target_sum / total if total > 0 else 0.0

    def _joint_prob(self, assignment: dict, structure: dict) -> float:
        p = 1.0
        for var, parents in structure.items():
            parent_key = tuple(_bin(assignment.get(pp, 0)) for pp in parents)
            value = _bin(assignment.get(var, 0))
            cpt_key = (var, parent_key)

            if cpt_key in self.cpt:
                p *= self.cpt[cpt_key].get(value, 0.5)
            else:
                p *= self.marginals.get(var, {0: 0.5, 1: 0.5}).get(value, 0.5)
        return p

    def report(self) -> dict:
        return {"structure": self.structure, "n_cpt": len(self.cpt)}