# Extreme-demand tariff analysis: code and reproducibility

This release contains the V1.1 simulation and model-fitting source for the V1.4
manuscript, including the analytic oracle, TAIL selection and fallback, other
predictive methods, external rolling validation, sensitivity analyses, and
seed-bearing configurations. Five aggregate tables are included. The cited
external load dataset and row-level Monte Carlo outputs are not redistributed.
The prior public companion remains available in the Git history at
`650db19b0632ecfc7f6222b1dd218774582afd8b` (tag `v1.0-companion`).

## Layout and provenance

- `scripts/` contains simulation, fitting, analysis, and audit entry points.
  `vendor/` and `vendor_track_a/` contain the model and decision modules.
  Archive-specific handoff verification and PDF repair scripts are omitted;
  three sensitivity scripts have portable root paths.
- `configs/revision.yaml`, `configs/residual_experiments.yaml`,
  `configs/regime_shift_sensitivity.yaml`, and
  `configs/external_rolling_validation.yaml` are the frozen V1.1 configurations.
- `DESIGN.md` records the frozen experiment design.
- `configs/external_korea.yaml` and `scripts/prepare_external_inputs.py` provide
  a portable entry point to reconstruct the external input from the cited
  Figshare v9 release. This entry point is adapted from the original Gate 6A
  reconstruction code. A source-data check matched the frozen
  `fixed15.parquet` (410,880 rows), eligibility table (280 rows), and input
  manifest (11 rows).
- `seed_manifest.yaml` records root seeds and later sensitivity seeds.
- `FROZEN_OUTPUT_TARGETS.md` lists expected row counts and hashes without
  distributing row-level outputs.
- `tables/` contains five selected aggregate tables from the frozen results.
  Run `python code/check_public_tables.py` to check that they are present
  and nonempty.
- `SHA256SUMS.csv` gives SHA-256 hashes for the release files.

No original review documents, manuscript drafts, personal filesystem paths,
raw load files, or row-level outputs are distributed here. The source code and
configurations disclose the complete synthetic generator and fitted methods.

## Environment

Python 3.12.10 was recorded by the original external-data reconstruction.
`requirements.txt` lists versions used in a subsequent successful smoke check;
it is not a verified freeze of every original production run. Install with
`python -m pip install -r requirements.txt`.

## Reproduction sequence

Run commands from this directory. The controlled synthetic analyses do not
need external data:

```text
python scripts/run_revision_experiment.py --mode smoke
python scripts/run_revision_experiment.py --mode full
python scripts/run_revision_experiment.py --mode frequency
python scripts/run_gap_crossed_experiment.py
python scripts/run_regime_shift_sensitivity.py --mode smoke
python scripts/run_regime_shift_sensitivity.py --mode full
python scripts/analyze_revision_results.py
python scripts/analyze_capacity_leave_one_cell_out.py
python scripts/analyze_spearman_history_bootstrap.py
```

The full experiment is computationally substantial. Seed roots and parameters
are in `configs/` and `seed_manifest.yaml`; full outputs are created under
`outputs/`, `ablations/`, `tables/`, and `reports/`. The `full` and `frequency`
runs should be completed before downstream summaries. For the two historical
R1 comparison tables, place `01_mechanism_cell_design.csv`,
`02_mechanism_results_full.csv`, and `05_exact_dgp_recovery_full.csv` under
`reference_inputs/r1/`. The current V1.1 summaries do not require those files.

A standalone 4-history smoke run passed all five built-in checks and produced
the frozen smoke-result SHA-256
`70a7aa30cd1341e01d4c445c2f7dc3ac8e9f74e8a86ef816802eb020a6838c39`.
The 1,000-history production run was not repeated for this release preparation.

For the external analysis, obtain the dataset cited in the manuscript
(Figshare v9, DOI 10.6084/m9.figshare.14822256.v9), place the original
`Factories/*.csv` and `DR_information/Industy DR Information.xlsx` under
`inputs/korean_source/`, then run:

```text
python scripts/prepare_external_inputs.py
python scripts/run_external_rolling_validation.py --variant natural_tail
python scripts/run_external_rolling_validation.py --variant all_observations
```

The original input files are deliberately excluded. Use `--source-root` if the
Figshare files are outside `inputs/korean_source/`. The reconstruction script
uses the original inclusion and demand-response rules from the archived Gate
6A implementation. The external path must be checked against the frozen input
manifest and the 8-factory/32-origin/224-score design before publication.

## Scope and remaining dependencies

The controlled generator and model-fitting pipeline are included. The original
external data must be obtained from its cited release. The three historical R1
comparison inputs are not distributed. A fresh source download, exact original
production dependency versions, and a new full-scale run were outside the
release-preparation check. Compare regenerated outputs with
`FROZEN_OUTPUT_TARGETS.md` and the manuscript before using them as a new
source of published numerical results.
