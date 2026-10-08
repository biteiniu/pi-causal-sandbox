"""End-to-end integration tests."""

import numpy as np
import pytest
from env.block_world_fork import BlockWorldFork
from policy.pi_core import PiCore
from causal.bayesian_scm import BayesianSCM
from causal.interventional_discovery import (
    interventional_discovery, edges_to_adjacency,
)
from causal.identifiability import IdentifiabilityChecker
from search.astar import astar_intervention


@pytest.fixture
def pipeline(block_world_fork, var_names_fork):
    env = block_world_fork
    _, direct = interventional_discovery(
        env.scm, var_names_fork, n_samples=200, verbose=False,
    )
    structure = {v: [] for v in var_names_fork}
    for (cause, effect) in direct:
        structure[effect].append(cause)

    bayes = BayesianSCM(structure, var_names_fork)
    data = []
    for action in env.actions:
        for _ in range(50):
            env.reset()
            data.append(env.do(action))
    bayes.fit_cpt(data)

    adj = edges_to_adjacency(direct, var_names_fork)
    checker = IdentifiabilityChecker(adj, var_names_fork)

    return {
        "env": env, "structure": structure, "bayes": bayes,
        "checker": checker, "adj": adj,
        "var_names": var_names_fork,
    }


class TestPipelineStructure:
    def test_fallen_c_has_two_parents(self, pipeline):
        parents = pipeline["structure"]["Fallen_C"]
        assert "Support_C" in parents
        assert "Fallen_B" in parents

    def test_force_a_is_root(self, pipeline):
        assert pipeline["structure"]["Force_A"] == []


class TestIdentifiability:
    def test_do_force_a(self, pipeline):
        info = pipeline["checker"].is_identifiable("Force_A", "Fallen_C")
        assert info["identifiable"]
        assert info["adjustment_set"] == []

    def test_do_fallen_b(self, pipeline):
        info = pipeline["checker"].is_identifiable("Fallen_B", "Fallen_C")
        assert info["identifiable"]
        # adjustment set should block backdoor; specific set may vary
        assert info["adjustment_set"] is not None


class TestAStar:
    def test_one_step_plan(self, pipeline):
        env = pipeline["env"]
        init = env.reset()
        plan = astar_intervention(
            env, init, bayes_scm=pipeline["bayes"],
            target="Fallen_C", max_depth=3, max_nodes=100,
        )
        assert plan["path"] is not None
        assert len(plan["path"]) == 1

    def test_plan_reaches_target(self, pipeline):
        env = pipeline["env"]
        init = env.reset()
        plan = astar_intervention(
            env, init, bayes_scm=pipeline["bayes"],
            target="Fallen_C", max_depth=3, max_nodes=100,
        )
        assert plan["cost"] < float("inf")


class TestPolicy:
    def test_pi_prefers_push(self, pipeline):
        env = pipeline["env"]
        pi = PiCore(
            env.actions, bayes_scm=pipeline["bayes"],
            force_observe_steps=0, target="Fallen_C",
        )
        state = env.reset()
        causal_state = {"observations": state, "uncertainty": {}, "depth": 0}
        action = pi(causal_state)
        assert action["type"] == "push"


class TestCounterfactual:
    def test_counterfactual_differs(self, pipeline):
        env = pipeline["env"]
        env.reset()
        factual = env.do({"type": "push", "force": 1.0})
        cf = env.counterfactual({"type": "wait"})
        assert factual["Fallen_C"] > 0.5
        assert cf["Fallen_C"] < 0.5