# Distributional functionals and tariff decisions

The additional analyses connect forecast errors to the quantities used by the
tariff decision. They use the primary 1,000 histories and nine tariff settings.

## Functional errors and economic loss

```bash
python scripts/analyze_decision_functionals.py
```

This command reads the supplied `data/derived/primary_rows.csv.gz`. It averages
absolute relative errors and regret over the nine cells within each history
and method. Mean and quantile errors are shared across cells; stop-loss errors
are evaluated at each cell's oracle contract threshold.

Spearman correlations are calculated separately for each method. The mean-
and stop-loss-error correlations with regret are compared with the quantile-
error correlation using the same outcome. Each of 5,000 bootstrap draws
resamples complete histories, preserving all variables and methods. Ranks are
recomputed in every resample. The seed is 2026092501; intervals are pointwise
95% percentile intervals.

## Gaussian reference and contract optimization

```bash
python scripts/run_gaussian_comparison.py
```

The Gaussian reference uses the original history stream
`[20260915, 2, 0, replication_id, 10]`. Each regenerated history is matched to
the supplied EMP mean and decision records before comparison. The script fits
Gaussian location and scale to the 24 monthly maxima by maximum likelihood,
then conditions that fitted predictive law on nonnegative demand. All tariff
functionals are evaluated analytically. The same tariff optimizer and
analytic oracle produce 9,000 additional decisions.

The regional comparisons average three cells within each history before
forming paired differences. They use 5,000 history resamples with seed
2026092502. Negative GAUSSIAN-minus-comparator differences favour GAUSSIAN.

The script also compares the contract rule with independent bounded
minimization, boundary candidates, and grids. The first 100 histories give
1,800 EMP and GAUSSIAN cases. Seven specified laws add 63 cases covering
KDE, GEV, the oracle, two tail-count constructions, a point mass, and a flat
optimal set. Objective values are compared because several contract levels
can have the same minimum cost. The full grid sizes and objective differences
are supplied in the results.

## Output files

All files below are written to `data/extensions/`. The functional analysis also
updates the stop-loss columns in `tables/method_fidelity_summary.csv`.

| File | Contents | Manuscript location |
| --- | --- | --- |
| `functional_history_summary.csv` | History-level errors and regret | Table 2 and Section 5.2 |
| `functional_fidelity.csv` | Median and mean absolute relative errors, in percent | Table 2 |
| `functional_correlation_contrasts.csv` | Paired differences in Spearman correlations | Table S11 |
| `gaussian_decisions.csv.gz` | All 9,000 Gaussian decisions | Section 5.2 and Table S12 |
| `gaussian_regional_contrasts.csv` | Fifteen paired regional comparisons | Table S12 |
| `gaussian_fidelity.csv` | Gaussian absolute relative errors, as proportions | Section S1.15 |
| `mode_stability_radii.csv` | Sufficient Wasserstein radii by tariff cell | Table S13 |
| `contract_solver_comparison.csv` | Independent objective comparisons | Table S14 |
| `worked_tariff_example.csv` | Costs and actions for the first history | Section S1.15 |

The stability radii are `oracle_gap / (2 * demand_rate * kappa)` for the
primary coefficients. They express the oracle mode separation in kW and
are sufficient radii, rather than estimated deployment guarantees. Correlation
contrasts describe associations within the controlled experiment.
