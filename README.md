# Decision-Relevant Modelling of Extreme Demand for Industrial Electricity Tariffs

Data and verification code for the manuscript:

**Mingyu Huang, Jiangweixi Wang, Zheng Gao, Yihang Yuan, and Zhengjun Yang.**  
*Decision-Relevant Modelling of Extreme Demand for Industrial Electricity Tariffs.*  
Applied Stochastic Models in Business and Industry.

## What this repository gives you

This release contains the manuscript-facing numerical outputs behind the paper's headline
results and two small scripts that recompute and verify them:

- `python code/check_release.py` — re-checks, in a few seconds, the headline quantities
  reported in the article (unique q0.99-error predictions, paired regional regret
  differences, diagnostic-unit structure, and the strongest regret association);
- `python code/run_public_analysis.py` — regenerates the public summary tables (q0.99-error
  summaries, TAIL-minus-baseline regret by oracle region, and Spearman associations between
  each diagnostic and full-action regret) into a local `results/` directory;
- `python code/recompute_roc_auc.py` — recomputes the tariff-mode-error ROC curves and
  point AUC values from the 5,400 decision-row table into a local `results/` directory.

Each derived CSV corresponds directly to a result reported in the paper, so the numbers in
the article can be traced to the files below and reproduced with the two scripts above.
Figure and table presentation is provided by the article and its Supporting Information.

## Repository layout

```text
code/
  run_public_analysis.py   Regenerate the public summary tables
  check_release.py         Verify row counts and headline quantities against the article
  recompute_roc_auc.py     Recompute tariff-mode-error ROC curves and point AUC values
  README.md

data/
  q99_errors.csv                       600 replication--method q0.99 errors
  regret_differences.csv               Paired TAIL-minus-baseline regret, by region
  diagnostic_units.csv                 600 replication--method diagnostic units (regret and
                                       diagnostics averaged over the nine tariff cells)
  mechanism_decision_rows_5400.csv     All 5,400 replication--method--cell decision rows,
                                       one row per tariff decision (ROC/AUC base table)
  confirmatory_regret_summary.csv      Frozen confirmatory contrasts with bootstrap intervals
  diagnostic_summary.csv               Diagnostic AUC and regret-association summary
  clipping_summary.csv                 Physical-clipping robustness cells
  misspecification_summary.csv         T4--T6 misspecification robustness cells
  external_crps.csv                    External Korean-factory CRPS values by variant
```

## Requirements

- Python >= 3.10
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

## Run the verification

```bash
python code/check_release.py
python code/run_public_analysis.py
```

## Data provenance and scope

The CSV files are the derived, manuscript-facing analysis outputs used for the numerical
results in the article. They are organized to mirror the corresponding results sections and
supporting tables. The original South Korean manufacturing-load dataset is publicly
available from the data source cited in the manuscript and is not redistributed here.

Two analysis levels appear in `data/` and should not be confused:

- `diagnostic_units.csv` contains the 600 replication--method units used for the regret
  prioritization analysis (Table 3 and Figure 3b of the main text): 200 replications times
  three methods. Within each unit, full-action regret and the diagnostic scores are averaged
  over the nine tariff cells before ranking, so one unit is not a single cell-level row.
- `mechanism_decision_rows_5400.csv` contains all 5,400 replication--method--cell decision
  rows (200 replications, three methods, nine cells). It is the base table for the
  tariff-mode-error ROC/AUC analysis (Table 3 and Figure 3a): each row is one tariff
  decision, the outcome is tariff-mode error (`tariff_mode_correct`), and the diagnostics
  are the four absolute-error or mode-value/gap scores on the row. Because the same fitted
  distribution is reused across the nine cells within a replication, the 5,400 rows are not
  5,400 independent forecasts; the article's inferential unit is the replication cluster.

This repository is a focused data-and-verification release for the reported numerical
results. The full research pipeline that generated them—the synthetic demand generator,
TAIL/KDE/EMP fitting internals, tariff-action optimization engine, Monte Carlo
orchestration, robustness simulation engines, and the external-data fitting pipeline—is
described in the article's Data and Code Availability statement and is outside the scope of
this release.

## License

See `LICENSE`.

## Citation

Please cite the accompanying article. Add the final DOI and bibliographic details after
publication.
