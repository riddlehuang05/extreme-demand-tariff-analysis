# Manuscript summary tables

These five CSV files summarize the primary experiment with 1,000 training
histories, five methods, and nine tariff settings.

| File | Contents | Manuscript location |
| --- | --- | --- |
| `method_fidelity_summary.csv` | Five-method quantile and mean fidelity | Table 2 |
| `cell_decision_summary.csv` | 45 method-by-cell summaries from 1,000 histories each | Table S1 |
| `regional_paired_contrasts.csv` | Regional paired regret contrasts on absolute and normalized scales | Table S2 |
| `capacity_leave_one_cell_out.csv` | Capacity-region cell-deletion sensitivity | Table S2 |
| `diagnostic_spearman_history_bootstrap.csv` | Within-method associations and history-bootstrap intervals | SI Section S1.7 |

The [source-data directory](../data/README.md) contains the broader collection
of manuscript results, including history-level records and external prediction
scores. To regenerate these summaries, follow the primary simulation and
analysis commands in [Running the analyses](../docs/REPRODUCING.md).
