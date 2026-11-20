# Result guide

Run `python scripts/reproduce_results.py` from the repository root. The command checks the released files and reproduces the numerical summaries listed below.

| Article result | Source record |
|---|---|
| Primary GEV–KDE and TAIL–EVENT-EMP contrasts | `primary/paired_regret_contrasts_10000.csv` and `primary/focal_maxt_intervals.csv` |
| Analytic oracle modes, cost gaps, and tariff radii | `primary/oracle_mode_values_9_cells.csv` and `primary/tariff_pair_perturbation_radii.csv` |
| Adaptive Wasserstein bounds and numerical checks | `adaptive_w1/` |
| Figure S3 inputs and numerical summary | `figure_s3/` |
| Event-level excess-law ablation | `tail_ablation/` |
| Common high-p* contract cells | `high_pstar/` |
| Contract quantile and action errors | `contract_action/` |
| External illustrative-charge summaries for Figure 4 | `external/` |

The records are organized by experiment and use the names cited in the article. Complete simulation histories and model-fitting checkpoints are included in the journal submission data files. The external charge records are calculations under a shared illustrative tariff applied to observed loads.
| Full solver-allowance recomputation for Table S13 | `solver_allowance/` |
