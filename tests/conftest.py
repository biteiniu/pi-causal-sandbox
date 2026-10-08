"""Shared fixtures for pytest suite."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def seed_numpy():
    np.random.seed(42)


@pytest.fixture
def block_world():
    from env.block_world import BlockWorld
    return BlockWorld()


@pytest.fixture
def block_world_3():
    from env.block_world_3 import BlockWorld3
    return BlockWorld3()


@pytest.fixture
def block_world_fork():
    from env.block_world_fork import BlockWorldFork
    return BlockWorldFork(noise=0.05)


@pytest.fixture
def block_world_collider():
    from env.block_world_collider import BlockWorldCollider
    return BlockWorldCollider(noise=0.02)


@pytest.fixture
def block_world_confounder():
    from env.block_world_confounder import BlockWorldConfounder
    return BlockWorldConfounder(noise=0.02)


@pytest.fixture
def continuous_chain():
    from env.continuous_chain import ContinuousChain
    return ContinuousChain(noise=0.1)


def _edges_of(env, var_names):
    edges = set()
    for v in var_names:
        for p in env.scm.variables[v].parents:
            edges.add((p, v))
    return edges


@pytest.fixture
def var_names_2(block_world):
    return list(block_world.scm.variables.keys())


@pytest.fixture
def var_names_3(block_world_3):
    return list(block_world_3.scm.variables.keys())


@pytest.fixture
def var_names_fork(block_world_fork):
    return list(block_world_fork.scm.variables.keys())


@pytest.fixture
def var_names_collider(block_world_collider):
    return list(block_world_collider.scm.variables.keys())


@pytest.fixture
def var_names_confounder(block_world_confounder):
    return list(block_world_confounder.scm.variables.keys())


@pytest.fixture
def var_names_cont(continuous_chain):
    return list(continuous_chain.scm.variables.keys())


@pytest.fixture
def true_edges_2(block_world, var_names_2):
    return _edges_of(block_world, var_names_2)


@pytest.fixture
def true_edges_3(block_world_3, var_names_3):
    return _edges_of(block_world_3, var_names_3)


@pytest.fixture
def true_edges_fork(block_world_fork, var_names_fork):
    return _edges_of(block_world_fork, var_names_fork)


@pytest.fixture
def true_edges_collider(block_world_collider, var_names_collider):
    return _edges_of(block_world_collider, var_names_collider)


@pytest.fixture
def true_edges_confounder(block_world_confounder, var_names_confounder):
    return _edges_of(block_world_confounder, var_names_confounder)


@pytest.fixture
def true_edges_cont(continuous_chain, var_names_cont):
    return _edges_of(continuous_chain, var_names_cont)