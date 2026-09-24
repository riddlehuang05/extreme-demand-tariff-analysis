# Release scope

The release is based on the V1.1 authoritative implementation used for the V1.4
manuscript. It is a code release with selected aggregate results, not a copy
of the full research archive.

## Included

| Analysis | Entry point or material |
| --- | --- |
| Primary simulation and five predictive constructions | `scripts/run_revision_experiment.py` |
| Tail-frequency stress | The same runner with `--mode frequency` |
| Crossed tariff geometry | `scripts/run_gap_crossed_experiment.py` |
| Regime shifts | `scripts/run_regime_shift_sensitivity.py` |
| External preprocessing and rolling prediction | `scripts/prepare_external_inputs.py`, `scripts/run_external_rolling_validation.py` |
| Paired summaries, cell deletion, and Spearman intervals | `scripts/analyze_*.py` |
| Secondary diagnostic analysis | `scripts/run_diagnostic_adjustment_analysis.py` |
| Shared estimation and tariff implementations | `src/extreme_demand/` |
| Frozen parameters and seeds | `configs/` |
| Selected final aggregate results | `tables/` |

The repeated estimator copies in the original project were byte-equivalent
and have been consolidated into one source package. Packaging changes include
portable paths, the descriptive package name, optional historical comparison
inputs, and organized documentation. They do not intentionally alter the
scientific algorithms or parameter settings.

## Scope limits

- External source data must be downloaded from its cited release.
- Per-history results, raw factory records, manuscript drafts, and reviewer
  correspondence are not distributed.
- Historical recovery, clipping, and furnace-misspecification experiments
  described in the supplementary material are distinct from the current
  V1.1 primary/stress/rolling pipeline. Supporting generator and simulation
  modules are retained in the source, but this release does not supply their
  complete original run configurations and standalone reproduction workflow.
  The optional R1 recovery comparison additionally requires the three inputs
  listed in [the reproduction guide](REPRODUCING.md).
- Production experiments were not rerun during this release preparation or
  directory cleanup. A previously recorded primary smoke check matched its
  frozen result hash; that check predates the directory cleanup.
- Dependency pins record the earlier smoke environment, not every original
  production environment.

The earlier public release and its previously published data remain accessible
in Git history under `v1.0-companion`.
