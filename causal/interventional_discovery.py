"""
Interventional causal discovery (binary and continuous readout).

Stage 1: multiple backgrounds, test do(X_i) effect on X_j.
Stage 2: drop indirect edges via conditioning subsets.
"""

import numpy as np
from itertools import combinations, product


def _sample_mean(scm, intervention, target, n, readout="binary"):
    vals = []
    for _ in range(n):
        s = scm.sample(interventions=intervention)
        vals.append(s[target])
    if readout == "binary":
        return float(np.mean([1 if v > 0.5 else 0 for v in vals]))
    return float(np.mean(vals))


def _make_backgrounds(var_names, x_i, pos_val, neg_val):
    bgs = [{}]
    for v in var_names:
        if v == x_i:
            continue
        bgs.append({v: pos_val})
        bgs.append({v: neg_val})
    return bgs


def _conditional_effect(scm, x_i, x_j, backgrounds, n, readout, pos_val, neg_val):
    max_eff = 0.0
    for bg in backgrounds:
        p1 = _sample_mean(scm, {x_i: pos_val, **bg}, x_j, n, readout)
        p0 = _sample_mean(scm, {x_i: neg_val, **bg}, x_j, n, readout)
        max_eff = max(max_eff, abs(p1 - p0))
    return max_eff


def _all_fixed_eliminate(
    scm, x_i, x_j, subset, n, residual_threshold, readout, pos_val, neg_val
):
    size = len(subset)
    for fixed_vals in product([pos_val, neg_val], repeat=size):
        fixed = dict(zip(subset, fixed_vals))
        p1 = _sample_mean(scm, {x_i: pos_val, **fixed}, x_j, n, readout)
        p0 = _sample_mean(scm, {x_i: neg_val, **fixed}, x_j, n, readout)
        if abs(p1 - p0) >= residual_threshold:
            return False
    return True


def interventional_discovery(
    scm,
    var_names,
    n_samples=300,
    edge_threshold=0.15,
    residual_threshold=0.15,
    readout="binary",
    verbose=True,
):
    """
    readout = "binary"     : do(X_i=1) vs do(X_i=0)
    readout = "continuous" : do(X_i=+1) vs do(X_i=-1)
    """
    if readout == "continuous":
        pos_val, neg_val = 1.0, -1.0
    else:
        pos_val, neg_val = 1.0, 0.0

    potential = {}
    for x_i in var_names:
        bgs = _make_backgrounds(var_names, x_i, pos_val, neg_val)
        for x_j in var_names:
            if x_i == x_j:
                continue
            eff = _conditional_effect(
                scm, x_i, x_j, bgs, n_samples,
                readout, pos_val, neg_val,
            )
            if eff > edge_threshold:
                potential[(x_i, x_j)] = eff

    if verbose:
        print(f"  Stage1 (potential edges): {len(potential)}")

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
                        scm, x_i, x_j, subset,
                        n_samples // 2, residual_threshold,
                        readout, pos_val, neg_val,
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