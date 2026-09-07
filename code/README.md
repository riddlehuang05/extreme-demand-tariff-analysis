# Public verification scripts

This directory contains the scripts that recompute and verify the manuscript-facing
numerical summaries distributed in `../data/`.

- `run_public_analysis.py` — regenerates the public summary tables (q0.99-error summaries,
  TAIL-minus-baseline regret by oracle region, and Spearman associations between each
  diagnostic and full-action regret).
- `check_release.py` — verifies row counts and a set of headline quantities against the
  values reported in the manuscript.
- `recompute_roc_auc.py` — recomputes the tariff-mode-error ROC curves and point AUC values
  from `../data/mechanism_decision_rows_5400.csv`, writing
  `results/diagnostic_auc_recomputed.csv` and `results/roc_curves_recomputed.csv`.

All scripts run on the derived files in `../data/` and require only the dependencies in
`../requirements.txt`. They do not require the research pipeline that produced the raw
experiments.
