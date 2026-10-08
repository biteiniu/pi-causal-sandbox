"""π-策略核心：target 可配置"""

import numpy as np
from typing import List, Dict


class PiCore:
    def __init__(
        self,
        actions: List[dict],
        learner=None,
        ahp=None,
        bayes_scm=None,
        force_observe_steps: int = 1,
        target: str = "Fallen_B",
    ):
        self.actions = actions
        self.learner = learner
        self.bayes_scm = bayes_scm
        self.force_observe_steps = force_observe_steps
        self.target = target
        self.step_count = 0

        from policy.ahp import AHPWeights
        self.ahp = ahp if ahp is not None else AHPWeights()

        self.mode = "explore"
        self.last_scores: Dict[str, float] = {}

    def __call__(self, causal_state: dict) -> dict:
        if self.step_count < self.force_observe_steps:
            self.step_count += 1
            for a in self.actions:
                if a.get("type") == "observe":
                    return a

        scores = []
        self.last_scores = {}
        for action in self.actions:
            s = self.ahp.score(action, causal_state, self)
            scores.append(s)
            self.last_scores[action["type"]] = s
        best_idx = int(np.argmax(scores))
        return self.actions[best_idx]

    def reset_exploration(self):
        self.step_count = 0

    def update(self, reward: float, causal_state: dict, action: dict):
        self.ahp.update_from_reward(action, reward, causal_state, self)
        w = self.ahp.w
        self.mode = "explore" if w[0] >= w[1] else "exploit"

    def _info_gain(self, action: dict, causal_state: dict) -> float:
        if action.get("type") == "observe":
            return 0.5

        intervention = self._action_to_intervention(action)
        if not intervention:
            return 0.0

        if self.bayes_scm is not None:
            try:
                p_do = self.bayes_scm.query(
                    self.target, interventions=intervention
                )
                from causal.info import binary_entropy
                return float(1.0 - binary_entropy(p_do))
            except Exception:
                pass

        data = self._get_data()
        if len(data) >= 2:
            from causal.info import information_gain
            ig = information_gain(data, action, target=self.target)
            if ig > 0:
                return float(ig)

        uncertainty = causal_state.get("uncertainty", {})
        if not uncertainty:
            return 0.3 if action["type"] == "push" else 0.1
        mean_unc = float(np.mean(list(uncertainty.values())))
        if action["type"] == "push":
            return mean_unc * 2.0
        elif action["type"] == "wait":
            return mean_unc * 0.2
        return 0.0

    def _goal_reward(self, action: dict, causal_state: dict) -> float:
        obs = causal_state.get("observations", {})
        if obs:
            if obs.get(self.target, 0.0) > 0.5:
                return 0.0
            if action["type"] == "push":
                return 1.0 * action.get("force", 1.0)
            return 0.1
        if action["type"] == "push":
            return 1.0 * action.get("force", 1.0)
        return 0.0

    def _get_data(self):
        if self.learner is None:
            return []
        if hasattr(self.learner, "int_buffer") and self.learner.int_buffer:
            return self.learner.int_buffer
        if hasattr(self.learner, "data") and self.learner.data:
            return self.learner.data
        return []

    def _action_to_intervention(self, action: dict) -> dict:
        t = action.get("type")
        if t == "push":
            return {"Force_A": action.get("force", 1.0)}
        elif t == "observe":
            return {}
        elif t == "wait":
            return {"Force_A": 0.0}
        return {}

    def report(self) -> dict:
        return {
            "mode": self.mode,
            "target": self.target,
            "ahp": self.ahp.report(),
            "last_scores": {k: round(v, 4) for k, v in self.last_scores.items()},
            "bayes_scm": self.bayes_scm is not None,
        }