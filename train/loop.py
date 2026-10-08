

"""π-Causal Sandbox v0.4: fork support"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.block_world import BlockWorld
from env.block_world_3 import BlockWorld3
from env.block_world_fork import BlockWorldFork
from policy.pi_core import PiCore
from causal.bayesian_scm import BayesianSCM
from causal.counterfactual_tree import CounterfactualTree
from causal.interventional_discovery import (
    interventional_discovery,
    edges_to_adjacency,
)
from causal.identifiability import IdentifiabilityChecker
from search.astar import astar_intervention


_BAYES_DATA = []


class SimpleLearner:
    def __init__(self):
        self.int_buffer = []
        self.data = self.int_buffer

    def add(self, intervention, sample):
        self.int_buffer.append((intervention, sample))

    def clear(self):
        self.int_buffer.clear()

    def __len__(self):
        return len(self.int_buffer)


def edges_to_structure(direct_edges, var_names):
    structure = {v: [] for v in var_names}
    for (cause, effect) in direct_edges:
        if effect in structure:
            structure[effect].append(cause)
    return structure


def fit_bayesian(env, structure, var_names, n_per_action=80, reset=False):
    global _BAYES_DATA
    if reset:
        _BAYES_DATA = []
    for action in env.actions:
        for _ in range(n_per_action):
            env.reset()
            _BAYES_DATA.append(env.do(action))
    bayes = BayesianSCM(structure, var_names)
    bayes.fit_cpt(_BAYES_DATA)
    return bayes


def train_v4(
    env_class=BlockWorldFork,
    target="Fallen_C",
    n_episodes=20,
    n_steps=6,
    discovery_every=10,
    verbose=True,
):
    env = env_class()
    actions = env.actions
    var_names = list(env.scm.variables.keys())
    learner = SimpleLearner()

    print(f"=== Environment: {env_class.__name__} ===")
    print(f"=== Target: {target} ===")
    print(f"=== Variables: {var_names} ===")

    print("\n=== Initial structure discovery ===")
    _, direct = interventional_discovery(
        env.scm, var_names, n_samples=300, verbose=True
    )
    structure = edges_to_structure(direct, var_names)
    print(f"  structure: {structure}")

    bayes = fit_bayesian(env, structure, var_names, reset=True)
    print(f"  CPT entries: {len(bayes.cpt)}")

    adj = edges_to_adjacency(direct, var_names)
    checker = IdentifiabilityChecker(adj, var_names)

    pi = PiCore(
        actions, learner=learner, bayes_scm=bayes,
        force_observe_steps=1, target=target,
    )
    cf_tree = CounterfactualTree(env.scm, max_depth=3, max_branch=3)

    reward_history = []
    action_counts = {"push": 0, "observe": 0, "wait": 0}

    for episode in range(n_episodes):
        pi.reset_exploration()
        state = env.reset()
        episode_reward = 0.0

        for step in range(n_steps):
            causal_state = {
                "observations": state,
                "uncertainty": {},
                "depth": step,
            }
            action = pi(causal_state)
            next_state = env.do(action)
            intervention = env._action_to_intervention(action)
            learner.add(intervention, next_state)

            if step > 0 and len(env.history) >= 2:
                cf_tree.expand(env.history, actions)

            ig = pi._info_gain(action, causal_state)
            gr = pi._goal_reward(action, causal_state)
            cost = 0.1 if action["type"] == "push" else 0.0
            reward = ig + gr - cost
            episode_reward += reward

            pi.update(reward, causal_state, action)
            action_counts[action["type"]] = action_counts.get(action["type"], 0) + 1
            state = next_state

        reward_history.append(episode_reward)

        if (episode + 1) % discovery_every == 0:
            print(f"\n--- Episode {episode + 1}: refresh structure ---")
            _, direct = interventional_discovery(
                env.scm, var_names, n_samples=200, verbose=False
            )
            structure = edges_to_structure(direct, var_names)
            bayes = fit_bayesian(env, structure, var_names, n_per_action=50)
            pi.bayes_scm = bayes
            adj = edges_to_adjacency(direct, var_names)
            checker = IdentifiabilityChecker(adj, var_names)
            print(f"  structure: {structure}")
            print(f"  CPT entries: {len(bayes.cpt)}")

        if verbose and episode % 5 == 0:
            print(
                f"Episode {episode:3d} | Reward {episode_reward:7.3f} | "
                f"Mode {pi.mode:8s} | Buffer {len(learner):4d}"
            )

    return {
        "env": env, "pi": pi, "learner": learner, "bayes": bayes,
        "checker": checker, "cf_tree": cf_tree,
        "rewards": reward_history, "action_counts": action_counts,
        "structure": structure, "target": target,
    }


def report(r):
    print("\n" + "=" * 60)
    print("Final Report")
    print("=" * 60)

    print(f"\n=== Training ===")
    print(f"  Target:        {r['target']}")
    print(f"  Avg reward:    {np.mean(r['rewards']):.3f}")
    print(f"  Action counts: {r['action_counts']}")
    print(f"  Mode:          {r['pi'].mode}")

    print(f"\n=== AHP ===")
    rpt = r["pi"].report()
    for k, v in rpt["ahp"]["weights"].items():
        print(f"  {k:15s} {v:.4f}")
    print(f"  CR = {rpt['ahp']['CR']} (consistent={rpt['ahp']['consistent']})")

    print(f"\n=== Discovered structure ===")
    for v, parents in r["structure"].items():
        print(f"  {v:12s} <- {parents}")

    print(f"\n=== Identifiability ===")
    target = r["target"]
    for t in ["Force_A", "Support_C", "Fallen_B"]:
        info = r["checker"].is_identifiable(t, target)
        print(
            f"  P({target}|do({t})): identifiable={info['identifiable']}, "
            f"adj={info['adjustment_set']}"
        )

    print(f"\n=== A* planning ===")
    env = r["env"]
    init_state = env.reset()
    plan = astar_intervention(
        env, init_state, bayes_scm=r["bayes"],
        target=target, max_depth=4, max_nodes=200,
    )
    print(f"  n_expanded = {plan['n_expanded']}")
    print(f"  cost       = {plan['cost']:.3f}")
    if plan["path"]:
        for i, step in enumerate(plan["path"]):
            print(f"    {i+1}. {dict(step)}")
    else:
        print("  no path found")

    print(f"\n=== Counterfactual demo ===")
    env.reset()
    factual = env.do({"type": "push", "force": 1.0})
    print(f"  Factual: push(1.0)  -> {target} = {factual[target]:.2f}")
    for action in [{"type": "wait"}, {"type": "push", "force": 0.3}]:
        cf = env.counterfactual(action)
        print(f"  CF:      {action['type']:8s} -> {target} = {cf[target]:.2f}")


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("π-Causal Sandbox v0.4")
    print("=" * 60)

    # env_class: BlockWorld / BlockWorld3 / BlockWorldFork
    result = train_v4(
        env_class=BlockWorldFork,
        target="Fallen_C",
        n_episodes=20,
        n_steps=6,
        discovery_every=10,
    )
    report(result)
    print("\nDone.")