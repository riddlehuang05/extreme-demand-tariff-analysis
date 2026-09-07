# Decision-Relevant Modelling of Extreme Demand for Industrial Electricity Tariffs

Data and verification code accompanying the manuscript by Mingyu Huang, Jiangweixi
Wang, Zheng Gao, Yihang Yuan, and Zhengjun Yang.

This repository is a focused release of the derived numerical outputs behind the
manuscript's reported results, together with small scripts that recompute and verify
them. It is the data-and-verification companion of the manuscript; the manuscript has
not yet been published.

## What this repository gives you

- `python code/check_release.py` — re-checks the headline quantities reported in the
  manuscript (unique q0.99-error predictions, paired regional regret differences,
  diagnostic-unit structure, and the strongest regret association);
- `python code/run_public_analysis.py` — regenerates the public summary tables
  (q0.99-error summaries, TAIL-minus-baseline regret by oracle region, Spearman
  associations) into a local `results/` directory;
- `python code/recompute_roc_auc.py` — recomputes the tariff-mode-error ROC curves and
  point AUC values from the 5,400-row decision table.

The derived CSV files in `data/` mirror the corresponding results sections of the
manuscript, so the reported numbers can be traced to these files and reproduced with
the scripts above.

## Repository layout

```text
code/
  run_public_analysis.py   Regenerate the public summary tables
  check_release.py         Verify row counts and headline quantities
  recompute_roc_auc.py     Recompute tariff-mode-error ROC curves and point AUC values
  README.md

data/
  q99_errors.csv                     600 replication--method q0.99 errors
  regret_differences.csv             Paired TAIL-minus-baseline regret, by region
  diagnostic_units.csv               600 replication--method diagnostic units
  mechanism_decision_rows_5400.csv   All 5,400 replication--method--cell decision rows
  confirmatory_regret_summary.csv    Confirmatory contrasts with bootstrap intervals
  diagnostic_summary.csv             Diagnostic AUC and regret-association summary
  clipping_summary.csv               Physical-clipping robustness cells
  misspecification_summary.csv       T4--T6 misspecification robustness cells
  external_crps.csv                  External Korean-factory CRPS values by variant
```

## Requirements

- Python >= 3.10
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

## Reproducing the summaries

```bash
python code/check_release.py
python code/run_public_analysis.py
```

`recompute_roc_auc.py` writes `results/diagnostic_auc_recomputed.csv` and
`results/roc_curves_recomputed.csv`.

## Data provenance and scope

The CSV files are derived, manuscript-facing analysis outputs. The original South Korean
manufacturing-load dataset is publicly available from the data source cited in the
manuscript and is not redistributed here.

The full research pipeline that generated the underlying Monte Carlo and external
outputs — the synthetic demand generator, the TAIL/KDE/EMP fitting internals, the
tariff-action optimization engine, and the external-data fitting pipeline — is outside
the scope of this release.

## License

See `LICENSE`.
