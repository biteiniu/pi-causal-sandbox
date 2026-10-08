"""
Interventional causal discovery.

Stage 1 (potential edges): under multiple backgrounds, test if do(X_i)
changes the distribution of X_j.

Stage 2 (direct edges): if some conditioning subset makes the effect
vanish, drop the edge as indirect.
"""

import numpy as np
from itertools import combinations, product
from typing import Dict, List, Tuple


def _sample_mean_bin(scm, intervention, target, n):
    vals = []
    for _ in range(n):
        s = scm.sample(interventions=intervention)
        vals.append(s[target])
    return float(np.mean([1 if v > 0.5 else 0 for v in vals]))


def _make_backgrounds(var_names, x_i):
    """Background interventions: empty + each other var fixed to 0/1"""
    bgs = [{}]
    for v in var_names:
        if v == x_i:
            continue
        bgs.append({v: 0.0})
        bgs.append({v: 1.0})
    return bgs


def _conditional_effect(scm, x_i, x_j, backgrounds, n):
    """Max |P(X_j|do(X_i=1)) - P(X_j|do(X_i=0))| across backgrounds"""
    max_eff = 0.0
    for bg in backgrounds:
        p1 = _sample_mean_bin(scm, {x_i: 1.0, **bg}, x_j, n)
        p0 = _sample_mean_bin(scm, {x_i: 0.0, **bg}, x_j, n)
        max_eff = max(max_eff, abs(p1 - p0))
    return max_eff


def _all_fixed_eliminate(scm, x_i, x_j, subset, n, residual_threshold):
    """True if for every assignment of `subset`, do(X_i) no longer affects X_j"""
    size = len(subset)
    for fixed_vals in product([0.0, 1.0], repeat=size):
        fixed = dict(zip(subset, fixed_vals))
        p1 = _sample_mean_bin(scm, {x_i: 1.0, **fixed}, x_j, n)
        p0 = _sample_mean_bin(scm, {x_i: 0.0, **fixed}, x_j, n)
        if abs(p1 - p0) >= residual_threshold:
            return False
    return True


def interventional_discovery(
    scm,
    var_names,
    n_samples=300,
    edge_threshold=0.15,
    residual_threshold=0.15,
    verbose=True,
):
    """Return (potential_edges, direct_edges)"""
    # ---------- Stage 1 ----------
    potential = {}
    for x_i in var_names:
        bgs = _make_backgrounds(var_names, x_i)
        for x_j in var_names:
            if x_i == x_j:
                continue
            eff = _conditional_effect(scm, x_i, x_j, bgs, n_samples)
            if eff > edge_threshold:
                potential[(x_i, x_j)] = eff

    if verbose:
        print(f"  Stage1 (potential edges): {len(potential)}")

    # ---------- Stage 2 ----------
    direct = {}
    for (x_i, x_j), eff in potential.items():
        mediators = [
            v for v in var_names
            if v not in (x_i, x_j)
            and (x_i, v) in potential
            and (v, x_j) in potential
        ]

        is_indirect = False
        if mediators:
            for size in range(1, len(mediators) + 1):
                if is_indirect:
                    break
                for subset in combinations(mediators, size):
                    if _all_fixed_eliminate(
                        scm, x_i, x_j, subset, n_samples // 2, residual_threshold
                    ):
                        is_indirect = True
                        break

        if not is_indirect:
            direct[(x_i, x_j)] = eff

    if verbose:
        print(f"  Stage2 (direct edges): {len(direct)}")
        print(f"  Filtered as indirect: {len(potential) - len(direct)}")

    return potential, direct


def edges_to_adjacency(edges, var_names):
    n = len(var_names)
    idx = {v: i for i, v in enumerate(var_names)}
    A = np.zeros((n, n))
    for (c, e), w in edges.items():
        A[idx[c], idx[e]] = w
    return A


def evaluate_against_true(discovered, true_edges, var_names):
    pred = set(discovered.keys())
    tp = len(true_edges & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true_edges) if true_edges else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": p, "recall": r, "f1": f1,
        "n_pred": len(pred), "n_true": len(true_edges), "tp": tp,
        "false_positives": sorted(pred - true_edges),
        "false_negatives": sorted(true_edges - pred),
    }