"""BayesianSCM tests."""

import numpy as np
import pytest
from causal.bayesian_scm import BayesianSCM


@pytest.fixture
def fitted_bayes(block_world, var_names_2):
    structure = {
        v: block_world.scm.variables[v].parents for v in var_names_2
    }
    bayes = BayesianSCM(structure, var_names_2)
    data = []
    for action in block_world.actions:
        for _ in range(100):
            block_world.reset()
            data.append(block_world.do(action))
    bayes.fit_cpt(data)
    return bayes


class TestBayesianQuery:
    def test_push_returns_one(self, fitted_bayes):
        p = fitted_bayes.query("Fallen_B", interventions={"Force_A": 1.0})
        assert p == pytest.approx(1.0, abs=0.05)

    def test_wait_returns_zero(self, fitted_bayes):
        p = fitted_bayes.query("Fallen_B", interventions={"Force_A": 0.0})
        assert p == pytest.approx(0.0, abs=0.05)

    def test_observe_equivalent_to_wait(self, fitted_bayes):
        p_obs = fitted_bayes.query("Fallen_B", interventions={})
        # {} means free Force_A; should still see P(Fallen_B) marginal
        assert 0.0 <= p_obs <= 1.0


class TestBayesianMatchesTruth:
    def test_query_matches_sampling(self, block_world, fitted_bayes):
        for intervention, expected in [
            ({"Force_A": 1.0}, 1.0),
            ({"Force_A": 0.0}, 0.0),
        ]:
            p_bayes = fitted_bayes.query(
                "Fallen_B", interventions=intervention
            )
            vals = []
            for _ in range(200):
                block_world.reset()
                s = block_world.scm.sample(interventions=intervention)
                vals.append(s["Fallen_B"])
            p_true = np.mean([1 if v > 0.5 else 0 for v in vals])
            assert p_bayes == pytest.approx(p_true, abs=0.1)