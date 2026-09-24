# Running the analyses

Run the commands below from the repository root with Python 3.12.
The included [source data](../data/README.md) and [summary tables](../tables/README.md)
can be read directly if you do not need to regenerate results.

## Environment

```bash
python -m pip install -r requirements.txt
```

The dependency versions record an environment in which the primary smoke run
succeeded. The complete set of original production environments was not
recorded, so these pins do not establish byte-for-byte reproducibility for
every full experiment.

## Primary simulation and frequency sensitivity

```bash
python scripts/run_revision_experiment.py --mode smoke
python scripts/run_revision_experiment.py --mode full
python scripts/run_revision_experiment.py --mode frequency
python scripts/analyze_revision_results.py
python scripts/analyze_capacity_leave_one_cell_out.py
python scripts/analyze_spearman_history_bootstrap.py
```

The full and frequency runs require a successful smoke run with the same
source signature. They write results to `outputs/full/` and
`outputs/frequency/`. The analysis scripts read those outputs and write to
`tables/` and `reports/`; running them replaces the corresponding summary
tables. Use a separate checkout to retain the supplied tables for comparison.

Two optional historical comparison tables require all three files below in
`reference_inputs/r1/`:

- `01_mechanism_cell_design.csv`
- `02_mechanism_results_full.csv`
- `05_exact_dgp_recovery_full.csv`

These inputs are not included. The analysis script skips these comparisons
when they are absent and records `legacy_r1_comparison_included` in its report.
The primary and frequency summaries use the generated outputs above.

## Tariff geometry and diagnostics

After the primary full run:

```bash
python scripts/run_gap_crossed_experiment.py
python scripts/summarize_gap_crossed_experiment.py
python scripts/run_diagnostic_adjustment_analysis.py
python scripts/audit_replication_convergence.py
```

After the frequency run, `python scripts/audit_tail_quadrature.py` provides
an additional numerical integration check.

## Operating-regime sensitivity

```bash
python scripts/run_regime_shift_sensitivity.py --mode smoke
python scripts/run_regime_shift_sensitivity.py --mode full
```

Results are written to `ablations/ASMBI_REGIME_SHIFT_V1/`, using the settings
in `configs/regime_shift_sensitivity.yaml`.

## External manufacturing loads

Download [Figshare Version 9](https://doi.org/10.6084/m9.figshare.14822256.v9).
Place `Factories/*.csv` and `DR_information/Industy DR Information.xlsx`
under `inputs/korean_source/`, retaining the source filenames, including
the spelling of `Industy`.

```bash
python scripts/prepare_external_inputs.py
python scripts/run_external_rolling_validation.py --variant natural_tail
python scripts/run_external_rolling_validation.py --variant all_observations
python scripts/summarize_external_variant_sensitivity.py
```

If the dataset is stored elsewhere, pass `--source-root PATH` to
`prepare_external_inputs.py`. Prediction outputs are written to
`outputs/external_rolling_cv/` and
`outputs/external_rolling_cv_all_observations/`.
Use `python scripts/audit_kblock_continuity.py` for the optional K-BLOCK
continuity check.

## Seeds and output sizes

Random seeds are listed in [configs/seed_manifest.yaml](../configs/seed_manifest.yaml)
and the experiment configuration files. The primary full run produces 45,000
decision records; frequency sensitivity produces 28,000; crossed tariff
geometry produces 380,000; and operating-regime sensitivity produces 36,000.
The primary external analysis produces 224 prediction-score records.

The [data index](../data/SUPPLEMENTARY_DATA_INDEX.csv) describes the included
scientific data. See [Workflow coverage](RELEASE_SCOPE.md) for supplementary
analyses whose result data are available but whose complete original run
workflows are not included. Full experiments have not been rerun from the
reorganized public repository; the recorded smoke result predates that
reorganization.
