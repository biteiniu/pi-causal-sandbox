
"""Unit tests for policy.ahp."""

import numpy as np
import pytest
from policy.ahp import AHPWeights


class FakePi:
    def _info_gain(self, action, state):
        return 0.9 if action["type"] == "push" else 0.2

    def _goal_reward(self, action, state):
        return 1.0 if action["type"] == "push" else 0.0


class TestAHPInit:
    def test_weights_sum_to_one(self):
        ahp = AHPWeights()
        assert ahp.w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_consistency(self):
        ahp = AHPWeights()
        cr, ok = ahp.consistency_check()
        assert ok
        assert cr < 0.1

    def test_lambda_max(self):
        ahp = AHPWeights()
        assert ahp.lambda_max == pytest.approx(5.068, abs=0.05)

    def test_weight_ordering(self):
        ahp = AHPWeights()
        assert ahp.w[0] > ahp.w[1] > ahp.w[4]


class TestAHPScoring:
    def test_push_preferred(self):
        ahp = AHPWeights()
        pi = FakePi()
        state = {"observations": {}, "depth": 0}
        s_push = ahp.score({"type": "push", "force": 1.0}, state, pi)
        s_obs = ahp.score({"type": "observe"}, state, pi)
        assert s_push > s_obs

    def test_observe_gets_safety(self):
        ahp = AHPWeights()
        pi = FakePi()
        state = {"observations": {}, "depth": 0}
        s_wait = ahp.score({"type": "wait"}, state, pi)
        s_obs = ahp.score({"type": "observe"}, state, pi)
        # observe has higher safety/explainability
        assert s_obs > s_wait


class TestAHPMetaUpdate:
    def test_weights_change_after_update(self):
        ahp = AHPWeights()
        pi = FakePi()
        state = {"observations": {}, "depth": 0}
        action = {"type": "push", "force": 1.0}

        w_before = ahp.w.copy()
        for _ in range(20):
            ahp.update_from_reward(action, 1.0, state, pi)
        ahp.score(action, state, pi)

        diff = np.abs(ahp.w - w_before).sum()
        assert diff > 0, "Weights should change after meta-updates"

    def test_report_structure(self):
        ahp = AHPWeights()
        rpt = ahp.report()
        assert "criteria" in rpt
        assert "weights" in rpt
        assert "CR" in rpt
        assert rpt["consistent"] is True