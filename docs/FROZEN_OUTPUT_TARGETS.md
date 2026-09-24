# Frozen output targets

These values come from the authoritative V1.1 output summaries in the private
archive. They identify what a production rerun should be checked against;
the row-level outputs themselves are not in this release.

| Output relative to package root | Expected rows | Frozen SHA-256 |
| --- | ---: | --- |
| `outputs/smoke/revision_results.csv` | 180 | `70a7aa30cd1341e01d4c445c2f7dc3ac8e9f74e8a86ef816802eb020a6838c39` |
| `outputs/full/revision_results.csv` | 45,000 | `3fe410c2313d1df116d6f59e0bd75a585b14bb974c634498757cd04a1c64aaee` |
| `outputs/frequency/revision_results.csv` | 28,000 | `003e6436f8accf3cd6d1752d4e3aed9b1836129809d2c9119fe275b60133a42f` |
| `outputs/gap_crossed/gap_crossed_results.csv` | 380,000 | `48625c63ee4b735c0320e9be8df72c94fdef6cc1377ece0fd9ef452bc005138c` |
| `ablations/ASMBI_REGIME_SHIFT_V1/full/regime_shift_results.csv` | 36,000 | `36de66d14ff8d006abbefe522f9c4af59a53cffbfc39d0c1516455f2b4f1e064` |
| `outputs/external_rolling_cv/prediction_scores.csv` | 224 | `884e8de5df1cb52d7bd2cf45a5907e710ed63ef4d368079963508c833f5718ee` |

The local standalone smoke run matched its frozen result hash exactly.
Production hashes have not been independently regenerated in this review.
The external-input reconstruction from a local retained copy matched the
frozen `fixed15.parquet`, `monthly_eligibility.csv`, and `input_manifest.csv`
as data frames; a fresh source download was not checked.
