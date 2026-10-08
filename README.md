# π-Causal Sandbox
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23237590.svg)](https://doi.org/10.5281/zenodo.23237590)

**A reproducible wind tunnel for active causal inference.**

[![Tests](https://img.shields.io/badge/tests-78%20passed-green)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## The Core Observation

In systems governed by **hard-threshold mechanisms**, purely observational
structure-learning methods (NOTEARS, DAG-GNN) **fail** — achieving **F1 < 0.2**
across chain, fork, collider, and confounder structures.

**Intervention-driven discovery succeeds** — achieving **F1 = 1.000** on the
same structures, with **zero false positives** across the entire noise sweep
(σ ≤ 2.0).

This asymmetry is the central finding. It deserves more attention in the
causal ML community than it has received.

---

## What It Does

Given a set of variables and observational/interventional data:

1. **Find who causes whom** — structure discovery
2. **Predict "what if I do X"** — interventional effects
3. **Check "is this estimable from data"** — identifiability
4. **Answer "what if it had been different"** — counterfactual reasoning
5. **Find the cheapest intervention plan** — active discovery under budget

---

## Quick Start

```bash
pip install -r requirements.txt
pytest                                          # 78 tests
python scripts/visualize.py                     # regenerate 3 charts
python train/loop.py                            # end-to-end demo
