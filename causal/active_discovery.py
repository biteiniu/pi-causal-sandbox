"""
Active interventional discovery.

Given a budget of interventions, choose the X_i that maximizes
expected information gain about the causal structure.

Stage 1: intervene on selected variables, measure effect on all others.
Stage 2: greedy mediator search to filter indirect edges.
"""

import numpy as np
from itertools import combinations, product


class ActiveDiscovery:
    def __init__(
        self,
        scm,
        var_names,
        budget=5,
        readout="binary",
        edge_threshold=0.15,
        n_per=300,
    ):
        self.scm = scm
        self.var_names = list(var_names)
        self.budget = budget
        self.readout = readout
        self.edge_threshold = edge_threshold
        self.n_per = n_per

        # Tie-break uses declaration order (roots usually come first)
        self._order = {v: i for i, v in enumerate(self.var_names)}

        self.effects = {}       # (vi, vj) -> measured effect
        self.intervened = set()
        self.history = []

    # ---------- measurement ----------

    def _measure_effect(self, vi, vj, fixed=None):
        if self.readout == "continuous":
            pos, neg = 1.0, -1.0
        else:
            pos, neg = 1.0, 0.0

        fixed = fixed or {}
        vals_pos, vals_neg = [], []
        for _ in range(self.n_per):
            s_pos = self.scm.sample(interventions={vi: pos, **fixed})
            vals_pos.append(s_pos[vj])
            s_neg = self.scm.sample(interventions={vi: neg, **fixed})
            vals_neg.append(s_neg[vj])

        if self.readout == "binary":
            mp = np.mean([1 if v > 0.5 else 0 for v in vals_pos])
            mn = np.mean([1 if v > 0.5 else 0 for v in vals_neg])
        else:
            mp = np.mean(vals_pos)
            mn = np.mean(vals_neg)
        return float(abs(mp - mn))

    def _intervene_row(self, vi):
        for vj in self.var_names:
            if vi == vj:
                continue
            eff = self._measure_effect(vi, vj)
            self.effects[(vi, vj)] = eff
        self.intervened.add(vi)
        self.history.append(vi)

    # ---------- selection ----------

    def _info_gain_estimate(self, vi):
        """
        Score of intervening vi:
        - Unknown outgoing edges (one per unmeasured X_j)
        - Mediator bonus for existing potential edges
        """
        if vi in self.intervened:
            return -1.0

        unknown_out = sum(
            1 for vj in self.var_names
            if vj != vi and (vi, vj) not in self.effects
        )

        mediator_bonus = 0
        for (a, b), eff in self.effects.items():
            if eff > self.edge_threshold and vi not in (a, b):
                mediator_bonus += 1

        return unknown_out + 0.3 * mediator_bonus

    def select_next(self):
        """
        Pick highest-scoring un-intervened variable.
        Tie-break: declaration order (index in var_names).
        """
        candidates = [
            vi for vi in self.var_names if vi not in self.intervened
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda vi: (-self._info_gain_estimate(vi), self._order[vi])
        )
        best = candidates[0]
        if self._info_gain_estimate(best) <= 0:
            return None
        return best

    # ---------- main loop ----------

    def run(self, verbose=False):
        while len(self.history) < self.budget:
            vi = self.select_next()
            if vi is None:
                break
            self._intervene_row(vi)
            if verbose:
                print(f"  Round {len(self.history)}: intervene on {vi}")
        return self._extract_edges()

    # ---------- stage 2 (filtering) ----------

    def _all_fixed_eliminate(self, vi, vj, subset):
        for fixed_vals in product([0.0, 1.0], repeat=len(subset)):
            fixed = dict(zip(subset, fixed_vals))
            eff = self._measure_effect(vi, vj, fixed=fixed)
            if eff >= self.edge_threshold:
                return False
        return True

    def _extract_edges(self):
        potential = {}
        for (vi, vj), eff in self.effects.items():
            if eff > self.edge_threshold:
                potential[(vi, vj)] = eff

        direct = {}
        for (vi, vj), eff in potential.items():
            mediators = [
                v for v in self.var_names
                if v not in (vi, vj)
                and (vi, v) in potential
                and (v, vj) in potential
            ]
            is_indirect = False
            if mediators:
                ranked = sorted(
                    mediators,
                    key=lambda k: -potential.get((vi, k), 0)
                    * potential.get((k, vj), 0),
                )
                for size in range(1, len(ranked) + 1):
                    if is_indirect:
                        break
                    for subset in combinations(ranked, size):
                        if self._all_fixed_eliminate(vi, vj, subset):
                            is_indirect = True
                            break
            if not is_indirect:
                direct[(vi, vj)] = eff

        return potential, direct