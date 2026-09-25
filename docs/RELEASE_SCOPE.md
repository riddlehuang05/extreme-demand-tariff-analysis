# Workflow coverage

This repository accompanies the manuscript's simulation and external
prediction studies. The table below identifies the available analysis entry
points and result data. Follow [Running the analyses](REPRODUCING.md) for
command order and input preparation.

## Available workflows

| Analysis | Entry point |
| --- | --- |
| Primary simulation and five predictive methods | `scripts/run_revision_experiment.py --mode full` |
| Event-frequency sensitivity | `scripts/run_revision_experiment.py --mode frequency` |
| Crossed tariff geometry | `scripts/run_gap_crossed_experiment.py` |
| Operating-regime sensitivity | `scripts/run_regime_shift_sensitivity.py --mode full` |
| External data preprocessing | `scripts/prepare_external_inputs.py` |
| External rolling prediction | `scripts/run_external_rolling_validation.py` |
| Paired result summaries | `scripts/analyze_revision_results.py` |
| Cell-deletion sensitivity | `scripts/analyze_capacity_leave_one_cell_out.py` |
| Within-method correlation intervals | `scripts/analyze_spearman_history_bootstrap.py` |
| Functional fidelity and paired correlation differences | `scripts/analyze_decision_functionals.py` |
| Gaussian reference and contract-objective comparisons | `scripts/run_gaussian_comparison.py` |
| Decision diagnostics | `scripts/run_diagnostic_adjustment_analysis.py` |

Shared estimators and tariff calculations are in `src/extreme_demand/`.
Experiment parameters and random seeds are in `configs/`.

## Data supplied

The [data directory](../data/README.md) contains 67 indexed files, including
synthetic history-level results, aggregate summaries, derived external scores,
and source records for Supplementary Figures S4–S6. Five selected manuscript
tables are also available in [tables/](../tables/README.md).

The raw manufacturing time series must be downloaded from the
[cited Figshare release](https://doi.org/10.6084/m9.figshare.14822256.v9).
The external records included here contain derived results and anonymized
factory identifiers.

## Supplementary experiment coverage

Parameter recovery, clipping sensitivity, and furnace misspecification
(Figures S4–S6) are supplied as result data with panel mappings. Supporting
generator and simulation modules are present, but the complete original run
configurations and standalone workflows for these experiments are not included.
The result files support inspection and plotting of the reported outcomes;
they do not provide a complete workflow for regenerating those experiments.

The optional historical comparisons in `analyze_revision_results.py` also
require three inputs that are not distributed. Their filenames and the
script's skip behavior are described in the [reproduction guide](REPRODUCING.md).

## Computing environment

Python dependencies are listed in `requirements.txt`. The functional analyses
and the complete 1,000-history Gaussian comparison were run from this repository.
The original five-method results are included; their complete
computing-environment records are unavailable. The pinned environment
also supports the primary smoke workflow.
