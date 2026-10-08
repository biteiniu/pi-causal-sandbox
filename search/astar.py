"""A* search over intervention sequences (deterministic via Bayesian expectations)."""

import heapq
import numpy as np
from causal.info import binary_entropy


class AStarPlanner:
    def __init__(self, env, bayes_scm=None, target="Fallen_B",
                 max_depth=5, max_nodes=500):
        self.env = env
        self.bayes_scm = bayes_scm
        self.target = target
        self.max_depth = max_depth
        self.max_nodes = max_nodes

        # Cache var names for deterministic transition
        self.var_names = None
        if bayes_scm is not None:
            self.var_names = list(bayes_scm.var_names)

    def _action_to_intervention(self, action: dict) -> dict:
        t = action.get("type")
        if t == "push":
            return {"Force_A": action.get("force", 1.0)}
        elif t == "observe":
            return {}
        elif t == "wait":
            return {"Force_A": 0.0}
        elif t == "give_drug":
            return {"Drug": 1.0}
        elif t == "placebo":
            return {"Drug": 0.0}
        elif t == "advertise":
            return {"Ad": 1.0}
        elif t == "hold":
            return {"Ad": 0.0}
        elif t == "scale_up":
            return {"Cache": 1.0}
        elif t == "scale_down":
            return {"Cache": 0.0}
        elif t == "force_pos":
            return {"Force_A": 1.0}
        elif t == "force_neg":
            return {"Force_A": -1.0}
        return {}

    def _heuristic(self, state: dict, action: dict) -> float:
        if self.bayes_scm is not None:
            intervention = self._action_to_intervention(action)
            p = self.bayes_scm.query(self.target, interventions=intervention)
            return 1.0 - p
        p = state.get(self.target, 0.5)
        return binary_entropy(p)

    def _state_key(self, state: dict) -> tuple:
        """Bucket values to 0.1 precision so visited set is meaningful."""
        return tuple(
            sorted((k, round(v * 10) / 10) for k, v in state.items())
        )

    def _simulate_step(self, state: dict, action: dict) -> dict:
        """
        Deterministic transition via Bayesian expectations:
        - intervened variables set exactly
        - others set to P(v=1 | do(intervention))
        """
        intervention = self._action_to_intervention(action)

        if self.bayes_scm is None:
            # Fallback: sample (non-deterministic)
            return self.env.scm.sample(interventions=intervention)

        next_state = state.copy()
        for v in self.var_names:
            if v in intervention:
                next_state[v] = intervention[v]
            else:
                next_state[v] = self.bayes_scm.query(
                    v, interventions=intervention
                )
        return next_state

    def plan(self, initial_state: dict) -> dict:
        start_key = self._state_key(initial_state)
        frontier = [(0.0, 0, start_key, (), initial_state)]
        visited = set()
        n_expanded = 0

        while frontier and n_expanded < self.max_nodes:
            f, g, state_key, path, state = heapq.heappop(frontier)
            if state_key in visited:
                continue
            visited.add(state_key)
            n_expanded += 1

            # Goal check
            if state.get(self.target, 0) > 0.5:
                return {"path": list(path), "n_expanded": n_expanded, "cost": f}

            if len(path) >= self.max_depth:
                continue

            for action in self.env.actions:
                force = action.get("force", 0.0)
                g_new = g + 1.0 + 0.01 * (1.0 - force)
                h_new = self._heuristic(state, action)

                next_state = self._simulate_step(state, action)
                next_key = self._state_key(next_state)
                new_path = path + (tuple(sorted(action.items())),)

                heapq.heappush(
                    frontier,
                    (g_new + h_new, g_new, next_key, new_path, next_state),
                )

        return {"path": None, "n_expanded": n_expanded, "cost": float("inf")}


def astar_intervention(env, initial_state, bayes_scm=None,
                       target="Fallen_B", max_depth=5, max_nodes=500) -> dict:
    planner = AStarPlanner(env, bayes_scm, target, max_depth, max_nodes)
    return planner.plan(initial_state)