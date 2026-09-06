# Public analysis scripts

This directory contains only non-core, manuscript-facing statistical summaries.

- `run_public_analysis.py` recomputes selected summary statistics from the derived
  public analysis files in `../data/`.
- `check_release.py` checks row counts and a small set of frozen headline quantities.

These scripts do **not** implement the synthetic demand generator, distribution-fitting
routines, tariff optimizer, Monte Carlo orchestration, or external-data fitting pipeline.
