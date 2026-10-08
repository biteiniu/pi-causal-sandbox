
"""Interventional causal discovery tests."""

import pytest
from causal.interventional_discovery import (
    interventional_discovery, evaluate_against_true,
)


class TestDiscovery2Block:
    def test_f1_perfect(self, block_world, var_names_2, true_edges_2):
        _, direct = interventional_discovery(
            block_world.scm, var_names_2,
            n_samples=300, verbose=False,
        )
        result = evaluate_against_true(direct, true_edges_2, var_names_2)
        assert result["f1"] == pytest.approx(1.0, abs=1e-3)
        assert result["precision"] == pytest.approx(1.0, abs=1e-3)
        assert result["recall"] == pytest.approx(1.0, abs=1e-3)


class TestDiscovery3Block:
    def test_f1_perfect(self, block_world_3, var_names_3, true_edges_3):
        _, direct = interventional_discovery(
            block_world_3.scm, var_names_3,
            n_samples=300, verbose=False,
        )
        result = evaluate_against_true(direct, true_edges_3, var_names_3)
        assert result["f1"] == pytest.approx(1.0, abs=1e-3)


class TestDiscoveryFork:
    def test_f1_perfect(self, block_world_fork, var_names_fork, true_edges_fork):
        _, direct = interventional_discovery(
            block_world_fork.scm, var_names_fork,
            n_samples=300, verbose=False,
        )
        result = evaluate_against_true(direct, true_edges_fork, var_names_fork)
        assert result["f1"] == pytest.approx(1.0, abs=1e-3)
        assert result["n_pred"] == 6


class TestNoiseRobustness:
    @pytest.mark.parametrize("noise", [0.05, 0.2, 0.5])
    def test_stable_low_noise(self, noise):
        from env.block_world_fork import BlockWorldFork
        env = BlockWorldFork(noise=noise)
        var_names = list(env.scm.variables.keys())
        true_edges = set()
        for v in var_names:
            for p in env.scm.variables[v].parents:
                true_edges.add((p, v))

        _, direct = interventional_discovery(
            env.scm, var_names, n_samples=300, verbose=False,
        )
        result = evaluate_against_true(direct, true_edges, var_names)
        assert result["f1"] == pytest.approx(1.0, abs=1e-3)

    def test_collapse_at_extreme_noise(self):
        """At σ=8.0, effect sizes fall well below threshold; F1 should drop."""
        from env.block_world_fork import BlockWorldFork
        env = BlockWorldFork(noise=8.0)
        var_names = list(env.scm.variables.keys())
        true_edges = set()
        for v in var_names:
            for p in env.scm.variables[v].parents:
                true_edges.add((p, v))

        # Use n_samples=500 for stable estimation at extreme noise
        _, direct = interventional_discovery(
            env.scm, var_names, n_samples=500, verbose=False,
        )
        result = evaluate_against_true(direct, true_edges, var_names)
        assert result["f1"] < 0.5, (
            f"At σ=8.0, F1 should collapse; got {result['f1']}"
        )

    def test_no_false_positives_at_moderate_noise(self):
        """Even at σ=2.0, algorithm should not hallucinate edges."""
        from env.block_world_fork import BlockWorldFork
        env = BlockWorldFork(noise=2.0)
        var_names = list(env.scm.variables.keys())
        true_edges = set()
        for v in var_names:
            for p in env.scm.variables[v].parents:
                true_edges.add((p, v))

        _, direct = interventional_discovery(
            env.scm, var_names, n_samples=500, verbose=False,
        )
        result = evaluate_against_true(direct, true_edges, var_names)
        # Algorithm is conservative: precision stays 1.0, errors are FN
        assert result["precision"] == pytest.approx(1.0, abs=1e-3)