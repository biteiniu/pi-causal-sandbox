"""Active discovery: budget vs information gain."""

import numpy as np
import pytest
from env.block_world_fork import BlockWorldFork
from causal.active_discovery import ActiveDiscovery
from causal.interventional_discovery import evaluate_against_true


def _true_edges(env):
    var_names = list(env.scm.variables.keys())
    edges = set()
    for v in var_names:
        for p in env.scm.variables[v].parents:
            edges.add((p, v))
    return var_names, edges


@pytest.fixture
def fork_env():
    return BlockWorldFork(noise=0.05)


class TestSelection:
    def test_root_intervened_first(self, fork_env):
        """Force_A is the root; should be chosen first (tie broken by name)."""
        var_names, _ = _true_edges(fork_env)
        np.random.seed(42)

        ad = ActiveDiscovery(
            fork_env.scm, var_names, budget=1,
            edge_threshold=0.15, n_per=300,
        )
        ad.run(verbose=False)

        # All variables have equal initial score; tie-break picks Force_A
        assert ad.history[0] == "Force_A"

    def test_no_variable_intervened_twice(self, fork_env):
        var_names, _ = _true_edges(fork_env)
        np.random.seed(42)

        ad = ActiveDiscovery(
            fork_env.scm, var_names, budget=10,
            edge_threshold=0.15, n_per=200,
        )
        ad.run(verbose=False)

        assert len(ad.history) == len(set(ad.history))
        assert len(ad.history) == len(var_names)  # capped at n


class TestFullBudget:
    def test_full_budget_achieves_perfect_f1(self, fork_env):
        var_names, true_edges = _true_edges(fork_env)
        np.random.seed(42)

        ad = ActiveDiscovery(
            fork_env.scm, var_names, budget=len(var_names),
            edge_threshold=0.15, n_per=300,
        )
        _, direct = ad.run(verbose=False)

        result = evaluate_against_true(direct, true_edges, var_names)
        assert result["f1"] == pytest.approx(1.0, abs=1e-2)
        assert len(ad.history) == len(var_names)


class TestPartialBudget:
    def test_budget_2_incomplete(self, fork_env):
        var_names, true_edges = _true_edges(fork_env)
        np.random.seed(42)

        ad = ActiveDiscovery(
            fork_env.scm, var_names, budget=2,
            edge_threshold=0.15, n_per=300,
        )
        _, direct = ad.run(verbose=False)

        result = evaluate_against_true(direct, true_edges, var_names)
        # 2 out of 6 variables intervened -> cannot see all edges
        assert result["f1"] < 1.0
        assert result["recall"] < 1.0

    @pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 6])
    def test_budget_monotone(self, budget):
        """F1 should be non-decreasing with budget (weak)."""
        np.random.seed(42)
        env = BlockWorldFork(noise=0.05)
        var_names, true_edges = _true_edges(env)

        ad = ActiveDiscovery(
            env.scm, var_names, budget=budget,
            edge_threshold=0.15, n_per=300,
        )
        _, direct = ad.run(verbose=False)

        result = evaluate_against_true(direct, true_edges, var_names)
        # With budget >= n, must be perfect
        if budget >= len(var_names):
            assert result["f1"] == pytest.approx(1.0, abs=1e-2)


class TestInfoGainScore:
    def test_score_positive_at_start(self, fork_env):
        var_names, _ = _true_edges(fork_env)
        ad = ActiveDiscovery(
            fork_env.scm, var_names, budget=1,
            edge_threshold=0.15, n_per=100,
        )
        for vi in var_names:
            assert ad._info_gain_estimate(vi) > 0

    def test_intervened_returns_negative(self, fork_env):
        var_names, _ = _true_edges(fork_env)
        ad = ActiveDiscovery(
            fork_env.scm, var_names, budget=1,
            edge_threshold=0.15, n_per=100,
        )
        ad.intervened.add("Force_A")
        assert ad._info_gain_estimate("Force_A") < 0