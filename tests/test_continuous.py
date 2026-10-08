"""Continuous SCM tests: linear-Gaussian chain + OLS fit."""

import numpy as np
import pytest
from causal.interventional_discovery import (
    interventional_discovery, evaluate_against_true,
)
from causal.linear_gaussian import LinearGaussianSCM


def _mixed_data(env, n=400):
    """Mix observational and interventional samples so that Force_A varies."""
    data = []
    for _ in range(n):
        f = float(np.random.choice([-1.0, 0.0, 1.0], p=[0.3, 0.4, 0.3]))
        data.append(env.scm.sample(interventions={"Force_A": f}))
    return data


class TestContinuousEnv:
    def test_positive_force_shifts_displaced(self, continuous_chain):
        vals = []
        for _ in range(80):
            continuous_chain.reset()
            s = continuous_chain.do({"type": "force_pos"})
            vals.append(s["Displaced_A"])
        assert np.mean(vals) == pytest.approx(1.5, abs=0.1)

    def test_negative_force_shifts_displaced(self, continuous_chain):
        vals = []
        for _ in range(80):
            continuous_chain.reset()
            s = continuous_chain.do({"type": "force_neg"})
            vals.append(s["Displaced_A"])
        assert np.mean(vals) == pytest.approx(-1.5, abs=0.1)

    def test_wait_gives_zero_mean(self, continuous_chain):
        vals = []
        for _ in range(80):
            continuous_chain.reset()
            s = continuous_chain.do({"type": "wait"})
            vals.append(s["Displaced_A"])
        assert abs(np.mean(vals)) < 0.1


class TestContinuousDiscovery:
    def test_f1_perfect(
        self, continuous_chain, var_names_cont, true_edges_cont
    ):
        _, direct = interventional_discovery(
            continuous_chain.scm, var_names_cont,
            n_samples=300, readout="continuous", verbose=False,
        )
        result = evaluate_against_true(
            direct, true_edges_cont, var_names_cont
        )
        assert result["f1"] == pytest.approx(1.0, abs=1e-2)
        assert result["n_pred"] == 5

    def test_no_spurious_edges(
        self, continuous_chain, var_names_cont
    ):
        _, direct = interventional_discovery(
            continuous_chain.scm, var_names_cont,
            n_samples=300, readout="continuous", verbose=False,
        )
        # Force_A should not directly edge Support_B / Velocity_B / Fallen_B
        assert ("Force_A", "Support_B") not in direct
        assert ("Force_A", "Fallen_B") not in direct
        assert ("Force_A", "Velocity_B") not in direct


class TestLinearGaussianSCM:
    def test_fit_recovers_coefficients(
        self, continuous_chain, var_names_cont, true_edges_cont
    ):
        structure = {v: [] for v in var_names_cont}
        for (cause, effect) in true_edges_cont:
            structure[effect].append(cause)

        # Mixed data: Force_A varies so OLS can recover coefficients
        data = _mixed_data(continuous_chain)

        lg = LinearGaussianSCM(structure, var_names_cont)
        lg.fit(data)

        # Force_A -> Displaced_A should be ≈ 1.5
        coef = lg.coeffs["Displaced_A"]["Force_A"]
        assert coef == pytest.approx(1.5, abs=0.1)

        # Displaced_A -> Support_B should be ≈ -0.8
        coef = lg.coeffs["Support_B"]["Displaced_A"]
        assert coef == pytest.approx(-0.8, abs=0.1)

        # Displaced_A -> Velocity_B should be ≈ 0.6
        coef = lg.coeffs["Velocity_B"]["Displaced_A"]
        assert coef == pytest.approx(0.6, abs=0.1)

    def test_query_matches_intervention(
        self, continuous_chain, var_names_cont, true_edges_cont
    ):
        structure = {v: [] for v in var_names_cont}
        for (cause, effect) in true_edges_cont:
            structure[effect].append(cause)

        data = _mixed_data(continuous_chain)
        lg = LinearGaussianSCM(structure, var_names_cont)
        lg.fit(data)

        # E[Fallen_B | do(Force_A=+1)] — true value ≈ 0.01 by hand
        predicted = lg.query_mean("Fallen_B", interventions={"Force_A": 1.0})

        vals = []
        for _ in range(300):
            continuous_chain.reset()
            s = continuous_chain.scm.sample(interventions={"Force_A": 1.0})
            vals.append(s["Fallen_B"])
        sampled = float(np.mean(vals))

        assert predicted == pytest.approx(sampled, abs=0.15)

    def test_query_zero_intervention(
        self, continuous_chain, var_names_cont, true_edges_cont
    ):
        structure = {v: [] for v in var_names_cont}
        for (cause, effect) in true_edges_cont:
            structure[effect].append(cause)

        data = _mixed_data(continuous_chain)
        lg = LinearGaussianSCM(structure, var_names_cont)
        lg.fit(data)

        # do(Force_A=0): Displaced_A=0, Support_B=0.5, Velocity_B=0
        # Fallen_B = 0.5*0.5 + 0.4*0 = 0.25
        predicted = lg.query_mean("Fallen_B", interventions={"Force_A": 0.0})
        assert predicted == pytest.approx(0.25, abs=0.1)

    def test_query_linearity(
        self, continuous_chain, var_names_cont, true_edges_cont
    ):
        """Linear-Gaussian: E[Fallen_B | do(Force_A=a)] is affine in a."""
        structure = {v: [] for v in var_names_cont}
        for (cause, effect) in true_edges_cont:
            structure[effect].append(cause)

        data = _mixed_data(continuous_chain)
        lg = LinearGaussianSCM(structure, var_names_cont)
        lg.fit(data)

        # E[Fallen_B | do(0)] and E[Fallen_B | do(1)]
        m0 = lg.query_mean("Fallen_B", interventions={"Force_A": 0.0})
        m1 = lg.query_mean("Fallen_B", interventions={"Force_A": 1.0})
        m2 = lg.query_mean("Fallen_B", interventions={"Force_A": 2.0})

        # Affine: m2 - m1 == m1 - m0
        assert (m2 - m1) == pytest.approx(m1 - m0, abs=1e-6)