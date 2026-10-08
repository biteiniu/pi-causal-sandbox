"""Unit tests for causal.info."""

import math
import pytest
from causal.info import (
    entropy, binary_entropy, gini,
    information_gain, mutual_information,
)


class TestEntropy:
    def test_uniform_binary(self):
        assert binary_entropy(0.5) == pytest.approx(1.0, abs=1e-6)

    def test_certain(self):
        assert binary_entropy(0.0) == 0.0
        assert binary_entropy(1.0) == 0.0

    def test_entropy_uniform_3(self):
        assert entropy([1/3, 1/3, 1/3]) == pytest.approx(
            math.log2(3), abs=1e-6
        )

    def test_entropy_empty(self):
        assert entropy([]) == 0.0


class TestGini:
    def test_balanced(self):
        assert gini([0, 1]) == pytest.approx(0.5, abs=1e-6)

    def test_pure(self):
        assert gini([1, 1]) == 0.0
        assert gini([0, 0]) == 0.0


class TestInformationGain:
    def test_perfect_split(self):
        data = []
        for _ in range(20):
            data.append(({"Force_A": 1.0}, {"Fallen_B": 1.0}))
            data.append(({"Force_A": 0.0}, {"Fallen_B": 0.0}))
        ig = information_gain(data, {"type": "push", "force": 1.0}, "Fallen_B")
        assert ig == pytest.approx(1.0, abs=1e-6)

    def test_no_intervention(self):
        data = [({"Force_A": 1.0}, {"Fallen_B": 1.0})] * 5
        ig = information_gain(data, {"type": "observe"}, "Fallen_B")
        assert ig == 0.0

    def test_insufficient_data(self):
        ig = information_gain([], {"type": "push", "force": 1.0}, "Fallen_B")
        assert ig == 0.0


class TestMutualInformation:
    def test_independent(self):
        x = [0, 0, 1, 1, 0, 0, 1, 1]
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        assert mutual_information(x, y) == pytest.approx(0.0, abs=1e-6)

    def test_identical(self):
        x = [0, 0, 1, 1, 0, 0, 1, 1]
        assert mutual_information(x, x) == pytest.approx(1.0, abs=1e-6)