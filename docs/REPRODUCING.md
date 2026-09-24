# Reproducing the analyses

Run all commands from the repository root using Python 3.12.
Install dependencies with `python -m pip install -r requirements.txt`.
The pinned versions were used in a previously successful smoke check; they are
not a verified freeze of every original production run.

The final packaging and directory cleanup did not rerun simulations, fitting,
or production analyses.

## Primary controlled experiment

```bash
python scripts/run_revision_experiment.py --mode smoke
python scripts/run_revision_experiment.py --mode full
python scripts/run_revision_experiment.py --mode frequency
python scripts/analyze_revision_results.py
python scripts/analyze_capacity_leave_one_cell_out.py
python scripts/analyze_spearman_history_bootstrap.py
```

The smoke run is required by the production runner's checks. The primary and
frequency runs write to `outputs/full/` and `outputs/frequency/`.
The analysis scripts read those generated outputs and write summaries to
`tables/` and `reports/`. They may overwrite the included aggregate tables;
commit or copy any results you wish to retain before regenerating them.

The two historical R1 comparison tables are optional. To generate them, supply
all three original files under `reference_inputs/r1/`:

- `01_mechanism_cell_design.csv`
- `02_mechanism_results_full.csv`
- `05_exact_dgp_recovery_full.csv`

Those files are not distributed. Their absence does not prevent generation of
the current primary and frequency summaries. The analysis report records
`legacy_r1_comparison_included`.

## Tariff geometry and diagnostics

After the primary full run:

```bash
python scripts/run_gap_crossed_experiment.py
python scripts/summarize_gap_crossed_experiment.py
python scripts/run_diagnostic_adjustment_analysis.py
python scripts/audit_replication_convergence.py
```

After the frequency run, optional numerical auditing is available through
`python scripts/audit_tail_quadrature.py`.

## Regime-shift sensitivity

```bash
python scripts/run_regime_shift_sensitivity.py --mode smoke
python scripts/run_regime_shift_sensitivity.py --mode full
```

Results are written under `ablations/ASMBI_REGIME_SHIFT_V1/`.
The configuration is `configs/regime_shift_sensitivity.yaml`.

## External manufacturing loads

Obtain [Figshare v9](https://doi.org/10.6084/m9.figshare.14822256.v9).
Place its original `Factories/*.csv` and
`DR_information/Industy DR Information.xlsx` under
`inputs/korean_source/`, preserving their names.

```bash
python scripts/prepare_external_inputs.py
python scripts/run_external_rolling_validation.py --variant natural_tail
python scripts/run_external_rolling_validation.py --variant all_observations
python scripts/summarize_external_variant_sensitivity.py
```

Use `--source-root PATH` with the preprocessing script when the source dataset
is stored elsewhere. The primary and sensitivity outputs are generated under
`outputs/external_rolling_cv/` and
`outputs/external_rolling_cv_all_observations/`.
The optional continuity audit is `python scripts/audit_kblock_continuity.py`.

## Seeds and verification

See `configs/seed_manifest.yaml` and the experiment YAML files for seeds.
[Frozen output targets](FROZEN_OUTPUT_TARGETS.md) documents expected result
counts and hashes. Source signatures depend on repository paths and therefore
change with the directory cleanup; they are distinct from result-file hashes.
Compare regenerated results with the frozen targets before treating a new run
as reproducing the published numerical results.
