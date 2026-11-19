# Result index

Run python scripts/verify_results.py from the repository root to check the released records against SHA256SUMS.csv and reproduce the key summaries.

| Paper result | Released records | Verification |
|---|---|---|
| Primary GEV–KDE and TAIL–EVENT-EMP contrasts; simultaneous intervals | primary/paired_regret_contrasts_10000.csv; primary/focal_maxt_intervals.csv | Compare regional means and interval signs |
| Analytic oracle modes, cost gaps, and tariff radii | primary/oracle_mode_values_9_cells.csv; primary/tariff_pair_perturbation_radii.csv | The script checks focal contrasts; the article tables display the oracle values and radii |
| Adaptive Wasserstein bounds, Table S6, Table S14, Figure S3 | adaptive_w1/adaptive_w1_by_law.parquet; adaptive_w1/adaptive_w1_decision_checks.parquet; adaptive_w1/adaptive_w1_by_method.csv; adaptive_w1/adaptive_w1_coverage.csv; figure_s3/ | Recompute method means, GEV bracket width, pass counts, and zero observed violations |
| Event-level excess-law ablation, Table S18 | tail_ablation/history_cell_metrics.parquet; tail_ablation/paired_summary.csv | Recompute history-paired regional and nine-cell means and 95% intervals |
| Common high-p* contract cells, Table S19 | high_pstar/high_pstar_common_contract_cells.csv | Check four shared cells and six paired method comparisons |
| Contract quantile and action errors | contract_action/contract_action_error_by_method_cell.csv | Check method-by-cell coverage and source manifest |
| External illustrative charges, Figure 4 | external/Figure4_external_selected.csv; external/external_realized_*.csv | Inspect plotted values and charge summaries |

The article PDFs and LaTeX sources are in article/. The result files are the archived numerical records used for the manuscript. Full simulation histories and model-fitting scripts are in the journal submission archives. The external charge files represent a shared illustrative tariff applied to observed loads.
