"""
Identifiability check via back-door criterion.

Given a DAG and (treatment, outcome), determine:
- is P(outcome | do(treatment)) identifiable from observational data?
- if yes, which adjustment set works?
"""

import numpy as np
import networkx as nx
from itertools import combinations
from typing import List, Dict, Optional


class IdentifiabilityChecker:
    def __init__(self, adjacency: np.ndarray, var_names: List[str], threshold: float = 0.1):
        """
        adjacency: (n, n) weighted matrix, [i, j] = i -> j
        """
        self.var_names = var_names
        self.adj = (np.abs(adjacency) > threshold).astype(int)
        self.G = nx.DiGraph()
        for v in var_names:
            self.G.add_node(v)
        for i, vi in enumerate(var_names):
            for j, vj in enumerate(var_names):
                if self.adj[i, j]:
                    self.G.add_edge(vi, vj)

    def is_identifiable(self, treatment: str, outcome: str) -> dict:
        """
        Back-door criterion:
        A set Z is valid if:
        1) Z blocks every back-door path from treatment to outcome
        2) Z contains no descendants of treatment
        """
        if treatment not in self.var_names or outcome not in self.var_names:
            return {"identifiable": False, "reason": "variable not found"}

        # Back-door paths: paths from treatment to outcome starting with an incoming edge
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)

        if not backdoor_paths:
            return {
                "identifiable": True,
                "reason": "no back-door paths",
                "adjustment_set": [],
                "backdoor_paths": [],
            }

        # Try to find an adjustment set
        descendants = nx.descendants(self.G, treatment)
        candidates = [
            v for v in self.var_names
            if v not in descendants and v != treatment and v != outcome
        ]

        for size in range(len(candidates) + 1):
            for subset in combinations(candidates, size):
                if self._blocks_all_backdoor(subset, backdoor_paths):
                    return {
                        "identifiable": True,
                        "reason": "back-door criterion satisfied",
                        "adjustment_set": list(subset),
                        "backdoor_paths": backdoor_paths,
                    }

        return {
            "identifiable": False,
            "reason": "no valid adjustment set",
            "adjustment_set": None,
            "backdoor_paths": backdoor_paths,
        }

    def _find_backdoor_paths(self, treatment: str, outcome: str) -> list:
        """Paths from treatment to outcome starting with treatment <- ..."""
        paths = []
        for parent in self.G.predecessors(treatment):
            try:
                for path in nx.all_simple_paths(self.G, parent, outcome):
                    full = [treatment] + path
                    paths.append(full)
            except nx.NetworkXNoPath:
                continue
        return paths

    def _blocks_all_backdoor(self, Z, backdoor_paths) -> bool:
        """Check if Z blocks every path in backdoor_paths (all nodes in path are not in Z)"""
        Z_set = set(Z)
        for path in backdoor_paths:
            # Path is blocked if any intermediate node is in Z
            intermediates = path[1:-1]
            if not any(n in Z_set for n in intermediates):
                return False
        return True

    def get_dag(self) -> np.ndarray:
        return self.adj.copy()

    def report(self, treatment: str, outcome: str) -> dict:
        return self.is_identifiable(treatment, outcome)