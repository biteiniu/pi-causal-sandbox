import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.block_world import BlockWorld
from causal.interventional_discovery import interventional_discovery, edges_to_adjacency
from causal.bayesian_scm import BayesianSCM
from causal.identifiability import IdentifiabilityChecker
from search.astar import astar_intervention

np.random.seed(42)
env = BlockWorld()
var_names = list(env.scm.variables.keys())

print("=" * 60)
print("Identifiability + A* test")
print("=" * 60)

# ---------- 1. Discover structure ----------
print("\n=== 1. Interventional discovery ===")
potential, direct = interventional_discovery(
    env.scm, var_names, n_samples=300,
    edge_threshold=0.15, residual_threshold=0.15, verbose=True,
)
adj = edges_to_adjacency(direct, var_names)

# ---------- 2. Identifiability ----------
print("\n=== 2. Identifiability check ===")
checker = IdentifiabilityChecker(adj, var_names)
for treatment, outcome in [
    ("Force_A", "Fallen_B"),
    ("Displaced_A", "Fallen_B"),
    ("Support_B", "Fallen_B"),
]:
    r = checker.is_identifiable(treatment, outcome)
    print(f"  P({outcome}|do({treatment})):")
    print(f"    identifiable = {r['identifiable']}")
    print(f"    reason       = {r['reason']}")
    print(f"    adj set      = {r['adjustment_set']}")

# ---------- 3. BayesianSCM ----------
print("\n=== 3. Fit BayesianSCM ===")
structure = {v: env.scm.variables[v].parents for v in var_names}
bayes = BayesianSCM(structure, var_names)
int_data = []
for action in env.actions:
    for _ in range(100):
        env.reset()
        int_data.append(env.do(action))
bayes.fit_cpt(int_data)
print(f"  CPT entries: {len(bayes.cpt)}")

# ---------- 4. A* planning ----------
print("\n=== 4. A* intervention planning ===")
init_state = env.reset()
result = astar_intervention(
    env, init_state, bayes_scm=bayes,
    target="Fallen_B", max_depth=3, max_nodes=200,
)
print(f"  n_expanded = {result['n_expanded']}")
print(f"  cost       = {result['cost']:.3f}")
if result["path"]:
    print(f"  path ({len(result['path'])} steps):")
    for i, step in enumerate(result["path"]):
        print(f"    {i+1}. {dict(step)}")
else:
    print("  no path found")

print("\nDone.")