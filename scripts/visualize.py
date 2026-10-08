"""Generate visualization charts for the pi-causal-sandbox project.

Outputs:
    docs/dags.png           -- causal DAGs for 4 environments
    docs/noise_sweep.png    -- F1 vs noise curve (n_samples=500, seed=42)
    docs/counterfactual.png -- factual vs counterfactual bars
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"
)
os.makedirs(DOCS, exist_ok=True)

N_SAMPLES = 500
SEED = 42


# ============================================================
# 1. DAG visualization
# ============================================================

def draw_dag(ax, pos, edges, title):
    ax.set_title(title, fontsize=11, fontweight="bold")
    for (u, v) in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        L = np.sqrt(dx * dx + dy * dy) + 1e-6
        off = 0.15
        ax.annotate(
            "",
            xy=(x2 - dx / L * off, y2 - dy / L * off),
            xytext=(x1 + dx / L * off, y1 + dy / L * off),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.4),
        )
    for name, (x, y) in pos.items():
        ax.scatter([x], [y], s=900, c="lightsteelblue",
                   edgecolors="navy", zorder=3, linewidths=1.5)
        ax.text(x, y, name, ha="center", va="center",
                fontsize=7, zorder=4, fontweight="bold")
    ax.set_xlim(-0.7, 4.7)
    ax.set_ylim(-0.7, 3.7)
    ax.axis("off")


def make_dags_figure():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    pos = {
        "Force_A":    (0, 1.5),
        "Displaced_A": (1.5, 1.5),
        "Support_B":  (3.0, 2.5),
        "Velocity_B": (3.0, 0.5),
        "Fallen_B":   (4.3, 1.5),
    }
    edges = [
        ("Force_A", "Displaced_A"),
        ("Displaced_A", "Support_B"),
        ("Displaced_A", "Velocity_B"),
        ("Support_B", "Fallen_B"),
        ("Velocity_B", "Fallen_B"),
    ]
    draw_dag(axes[0, 0], pos, edges, "Chain (2 blocks)")

    pos = {
        "Force_A":    (0, 1.5),
        "Displaced_A": (1.4, 1.5),
        "Support_B":  (2.8, 2.5),
        "Support_C":  (2.8, 0.5),
        "Fallen_B":   (4.1, 2.5),
        "Fallen_C":   (4.1, 1.0),
    }
    edges = [
        ("Force_A", "Displaced_A"),
        ("Displaced_A", "Support_B"),
        ("Displaced_A", "Support_C"),
        ("Support_B", "Fallen_B"),
        ("Support_C", "Fallen_C"),
        ("Fallen_B", "Fallen_C"),
    ]
    draw_dag(axes[0, 1], pos, edges, "Fork + Diamond")

    pos = {
        "Force_A":   (0, 2.3),
        "Force_B":   (0, 0.7),
        "Collision": (2.0, 1.5),
        "Fallen_C":  (3.8, 1.5),
    }
    edges = [
        ("Force_A", "Collision"),
        ("Force_B", "Collision"),
        ("Collision", "Fallen_C"),
    ]
    draw_dag(axes[1, 0], pos, edges, "Collider (A -> C <- B)")

    pos = {
        "Wind":     (1.0, 2.5),
        "Force_A":  (0.0, 1.0),
        "Fallen_B": (2.5, 1.0),
    }
    edges = [
        ("Wind", "Force_A"),
        ("Wind", "Fallen_B"),
        ("Force_A", "Fallen_B"),
    ]
    draw_dag(axes[1, 1], pos, edges, "Confounder (U -> A, U -> B)")

    plt.tight_layout()
    out = os.path.join(DOCS, "dags.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ============================================================
# 2. Noise sweep curve (deterministic per-σ seed)
# ============================================================

def make_noise_curve():
    from env.block_world_fork import BlockWorldFork
    from causal.interventional_discovery import (
        interventional_discovery, evaluate_against_true,
    )

    sigmas = [0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]
    f1_scores = []

    print(f"  running noise sweep (n_samples={N_SAMPLES}, seed={SEED} per σ)...")
    for s in sigmas:
        # Reset seed INSIDE the loop: every σ is independently reproducible
        np.random.seed(SEED)

        env = BlockWorldFork(noise=s)
        var_names = list(env.scm.variables.keys())
        true_edges = set()
        for v in var_names:
            for p in env.scm.variables[v].parents:
                true_edges.add((p, v))

        _, direct = interventional_discovery(
            env.scm, var_names, n_samples=N_SAMPLES, verbose=False,
        )
        r = evaluate_against_true(direct, true_edges, var_names)
        f1_scores.append(r["f1"])
        print(f"    σ={s:>4.2f} → F1={r['f1']:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, f1_scores, "o-", color="steelblue", lw=2, markersize=8)
    ax.axhline(0.5, color="orange", ls="--", lw=1,
               label="F1 = 0.5 (collapse)")
    ax.axvline(2.5, color="red", ls=":", lw=1,
               label="σ = 2.5 (first failure)")
    ax.set_xscale("log")
    ax.set_xlabel("Noise σ (log scale)", fontsize=11)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_title(
        f"Robustness: F1 vs noise σ (n_samples={N_SAMPLES})",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_ylim(-0.05, 1.05)

    out = os.path.join(DOCS, "noise_sweep.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ============================================================
# 3. Counterfactual comparison
# ============================================================

def make_counterfactual():
    from env.block_world_fork import BlockWorldFork

    np.random.seed(SEED)
    env = BlockWorldFork(noise=0.02)

    env.reset()
    factual = env.do({"type": "push", "force": 1.0})
    target_factual = factual["Fallen_C"]

    cf_actions = [
        ("wait",      {"type": "wait"}),
        ("push(0.3)", {"type": "push", "force": 0.3}),
        ("push(0.5)", {"type": "push", "force": 0.5}),
        ("push(1.0)", {"type": "push", "force": 1.0}),
    ]

    labels = ["Factual\npush(1.0)"]
    values = [target_factual]
    for name, act in cf_actions:
        cf = env.counterfactual(act)
        labels.append(f"CF:\n{name}")
        values.append(cf["Fallen_C"])

    colors = ["steelblue"] + ["lightcoral"] * 4
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors,
                  edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02,
            f"{v:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("P(Fallen_C = 1)", fontsize=11)
    ax.set_title(
        "Counterfactual reasoning on BlockWorldFork",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    out = os.path.join(DOCS, "counterfactual.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    print("\n[1/3] DAG structures")
    make_dags_figure()

    print("\n[2/3] Noise sweep curve")
    make_noise_curve()

    print("\n[3/3] Counterfactual comparison")
    make_counterfactual()

    print("\nDone. Charts saved to docs/")