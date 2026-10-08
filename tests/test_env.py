"""Tests for SCM and BlockWorld environments."""

import numpy as np
import pytest


class TestSCM:
    def test_sample_returns_dict(self, block_world):
        state = block_world.reset()
        assert isinstance(state, dict)
        for v in block_world.scm.variables:
            assert v in state

    def test_intervention_is_exact(self, block_world):
        block_world.reset()
        state = block_world.do({"type": "push", "force": 1.0})
        assert state["Force_A"] == 1.0

    def test_counterfactual_reverts(self, block_world):
        block_world.reset()
        block_world.do({"type": "push", "force": 1.0})
        cf = block_world.counterfactual({"type": "wait"})
        assert cf["Force_A"] == 0.0


class TestBlockWorld2:
    def test_push_causes_fall(self, block_world):
        vals = []
        for _ in range(50):
            block_world.reset()
            s = block_world.do({"type": "push", "force": 1.0})
            vals.append(s["Fallen_B"])
        p = np.mean([1 if v > 0.5 else 0 for v in vals])
        assert p == pytest.approx(1.0, abs=0.05)

    def test_wait_no_fall(self, block_world):
        vals = []
        for _ in range(50):
            block_world.reset()
            s = block_world.do({"type": "wait"})
            vals.append(s["Fallen_B"])
        p = np.mean([1 if v > 0.5 else 0 for v in vals])
        assert p == pytest.approx(0.0, abs=0.05)


class TestBlockWorld3:
    def test_chain(self, block_world_3):
        block_world_3.reset()
        s = block_world_3.do({"type": "push", "force": 1.0})
        assert s["Fallen_B"] > 0.5
        assert s["Fallen_C"] > 0.5

    def test_wait_chain_holds(self, block_world_3):
        block_world_3.reset()
        s = block_world_3.do({"type": "wait"})
        assert s["Fallen_B"] < 0.5
        assert s["Fallen_C"] < 0.5


class TestBlockWorldFork:
    def test_push_both_branches(self, block_world_fork):
        block_world_fork.reset()
        s = block_world_fork.do({"type": "push", "force": 1.0})
        assert s["Fallen_B"] > 0.5
        assert s["Fallen_C"] > 0.5

    def test_wait_holds_both(self, block_world_fork):
        block_world_fork.reset()
        s = block_world_fork.do({"type": "wait"})
        assert s["Fallen_B"] < 0.5
        assert s["Fallen_C"] < 0.5