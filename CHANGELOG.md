# CHANGELOG

## v1.0.1 (2026-09-21)

### Added
- `docs/TECHNICAL_REPORT.md`: 7-section academic report
  - Abstract, Introduction, Formal Framework
  - Methods (4 algorithms), Experiments (5 sections)
  - Discussion, Related Work, Conclusion
  - Appendices A-C (reproduction, layout, version history)
- `scripts/build_report.sh`: pandoc PDF build (Linux/macOS)
- `scripts/build_report.ps1`: pandoc PDF build (Windows)

### Report Highlights
- **Central thesis**: hard-threshold mechanisms → observational
  structure learning fails (F1<0.2), interventions succeed (F1=1.0)
- **Four insights (I1-I4)**: interventional superiority, multi-background
  AND-gate capture, collider minimal adjustment, active near-optimality
- **Five limitations**: threshold bias, linearity assumption, no latents,
  no time, seed-sensitivity at σ≈2.5
- **Related work**: Pearl, Peters/Schölkopf, NOTEARS, DAG-GNN, Zhang

### Test Status
- 78 passed in 33.45s (unchanged)

---

## v1.0 (2026-09-21)

### Added
- `causal/active_discovery.py`: budget-limited active discovery
- `tests/test_active.py`: 12 tests

### Fixed
- Tie-break in `select_next()` uses declaration order, not alphabetical

---

## v0.9 / v0.8 / v0.7 / v0.6.1 / v0.6 / v0.5.3 / v0.5 / v0.4 / v0.3 / v0.2 / v0.1
(see earlier entries)