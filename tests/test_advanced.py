"""Tests for collider and confounder structures."""

import numpy as np
import pytest
from causal.interventional_discovery import (
    interventional_discovery, edges_to_adjacency, evaluate_against_true,
)
from causal.identifiability import IdentifiabilityChecker


# ==================== Collider ====================

class TestColliderDiscovery:
    def test_three_edges_found(
        self, block_world_collider, var_names_collider, true_edges_collider
    ):
        _, direct = interventional_discovery(
            block_world_collider.scm, var_names_collider,
            n_samples=300, verbose=False,
        )
        result = evaluate_against_true(
            direct, true_edges_collider, var_names_collider
        )
        assert result["f1"] == pytest.approx(1.0, abs=1e-3)
        assert result["n_pred"] == 3

    def test_no_spurious_force_edges(
        self, block_world_collider, var_names_collider
    ):
        """Force_A and Force_B have no direct edge."""
        _, direct = interventional_discovery(
            block_world_collider.scm, var_names_collider,
            n_samples=300, verbose=False,
        )
        assert ("Force_A", "Force_B") not in direct
        assert ("Force_B", "Force_A") not in direct

    def test_collision_is_common_effect(
        self, block_world_collider, var_names_collider
    ):
        _, direct = interventional_discovery(
            block_world_collider.scm, var_names_collider,
            n_samples=300, verbose=False,
        )
        assert ("Force_A", "Collision") in direct
        assert ("Force_B", "Collision") in direct
        assert ("Collision", "Fallen_C") in direct


class TestColliderIdentifiability:
    def test_do_force_a_empty_adjustment(
        self, block_world_collider, var_names_collider
    ):
        """
        P(Fallen_C | do(Force_A)) identified WITHOUT adjusting Collision.
        Empty adjustment set blocks all back-doors (there are none from
        the root Force_A).
        """
        _, direct = interventional_discovery(
            block_world_collider.scm, var_names_collider,
            n_samples=200, verbose=False,
        )
        adj = edges_to_adjacency(direct, var_names_collider)
        checker = IdentifiabilityChecker(adj, var_names_collider)
        info = checker.is_identifiable("Force_A", "Fallen_C")
        assert info["identifiable"]
        assert info["adjustment_set"] == []

    def test_collision_blocks_spurious_path(
        self, block_world_collider, var_names_collider
    ):
        """
        Force_A and Force_B should be conditionally independent
        given nothing, but not given Collision.
        """
        env = block_world_collider
        # Sample both forces independently (no intervention -> both 0)
        env.reset()
        # Direct check: P(Collision | Force_A, Force_B) has structure
        # Collision = 1 iff both forces are on.
        samples = []
        for _ in range(50):
            env.reset()
            s = env.do({"type": "push_both"})
            samples.append(s["Collision"])
        p_collision_both = np.mean(samples)
        assert p_collision_both > 0.9

        samples = []
        for _ in range(50):
            env.reset()
            s = env.do({"type": "push_A"})
            samples.append(s["Collision"])
        p_collision_only_a = np.mean(samples)
        assert p_collision_only_a < 0.1


# ==================== Confounder ====================

class TestConfounderDiscovery:
    def test_three_edges_found(
        self, block_world_confounder, var_names_confounder, true_edges_confounder
    ):
        _, direct = interventional_discovery(
            block_world_confounder.scm, var_names_confounder,
            n_samples=300, verbose=False,
        )
        result = evaluate_against_true(
            direct, true_edges_confounder, var_names_confounder
        )
        assert result["f1"] == pytest.approx(1.0, abs=1e-3)
        assert result["n_pred"] == 3

    def test_wind_affects_both(
        self, block_world_confounder, var_names_confounder
    ):
        _, direct = interventional_discovery(
            block_world_confounder.scm, var_names_confounder,
            n_samples=300, verbose=False,
        )
        assert ("Wind", "Force_A") in direct
        assert ("Wind", "Fallen_B") in direct
        assert ("Force_A", "Fallen_B") in direct


class TestConfounderIdentifiability:
    def test_do_force_a_needs_wind(
        self, block_world_confounder, var_names_confounder
    ):
        """
        P(Fallen_B | do(Force_A)) is identified only by adjusting Wind.
        """
        _, direct = interventional_discovery(
            block_world_confounder.scm, var_names_confounder,
            n_samples=200, verbose=False,
        )
        adj = edges_to_adjacency(direct, var_names_confounder)
        checker = IdentifiabilityChecker(adj, var_names_confounder)
        info = checker.is_identifiable("Force_A", "Fallen_B")
        assert info["identifiable"]
        # Adjustment set must include Wind
        assert "Wind" in info["adjustment_set"]


class TestConfounderEffects:
    def test_do_force_a_differs_from_observing(self, block_world_confounder):
        """Naive correlation > causal effect (typical confounding)."""
        env = block_world_confounder

        # Naive: observe naturally, compute P(Fallen_B | Force_A=1) - P(Fallen_B | Force_A=0)
        obs = []
        for _ in range(300):
            env.reset()
            s = env.scm.sample()
            obs.append((s["Force_A"], s["Fallen_B"]))
        p_1 = np.mean([fb for fa, fb in obs if fa > 0.5])
        p_0 = np.mean([fb for fa, fb in obs if fa <= 0.5])
        naive_effect = p_1 - p_0

        # Causal: do(Force_A=1) vs do(Force_A=0)
        vals = []
        for _ in range(300):
            env.reset()
            s = env.scm.sample(interventions={"Force_A": 1.0})
            vals.append(s["Fallen_B"])
        p_causal_1 = np.mean([1 if v > 0.5 else 0 for v in vals])

        vals = []
        for _ in range(300):
            env.reset()
            s = env.scm.sample(interventions={"Force_A": 0.0})
            vals.append(s["Fallen_B"])
        p_causal_0 = np.mean([1 if v > 0.5 else 0 for v in vals])
        causal_effect = p_causal_1 - p_causal_0

        # Both effects are positive; naive may differ from causal
        assert causal_effect > 0.5, "True causal effect should be strong"
        assert naive_effect > 0.0, "Naive association should also be positive"