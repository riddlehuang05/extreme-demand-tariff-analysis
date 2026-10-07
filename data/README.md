# Data included in the final repository

The data tree is organized around the published figures, tables, and stability calculations.

| Path | Contents |
|---|---|
| `plot/` | Machine-readable inputs for final main and supplementary figures, including paired primary distributions, crossed-grid profiles, and Figure S3 panels |
| `source/primary/` | Final 10,000-history oracle, paired-contrast, regret-contribution, precision, and signed-functional summaries |
| `source/high_pstar/` | Final high-p* design, cell summaries, and same-information contrasts |
| `source/s6_alpha120/` | Physical-pipeline tariff design, oracle grid, and method-by-region summaries |
| `source/high_density_v2/` | Compact summaries for the negative-binomial–Weibull challenge, physical-pipeline run record, and high-p* challenge |
| `source/dgp_external_support/` | External predictive-score sources and fitting-support sensitivity records |
| `source/crossed_grid_no_upper_1000/` | Tariff design and cell/region summaries for the 76-cell crossed analysis |
| `numerical_audit/` | Saved final decision and competitor-pair records used by the direct tariff-contrast numerical audit |
| `derived_publication/tariff_contrast_wasserstein/` | Analytic radii, numerical-condition coverage, smoke checks, identity checks, final Figure S3 outputs, and exact input hashes |
| `reported_separate_experiments/` | Source records for parameter recovery, physical clipping, and threshold-fixed bootstrap coverage |
| `../tables/` | Machine-readable final main and supplementary table values |

The primary design contains 10,000 histories, five predictive methods, nine tariff cells, and 450,000 method–history–cell decisions. The first 1,000 primary histories appear only in the paired fitting-support sensitivity; the crossed tariff design is a separate 76-cell analysis.

`SOURCE_DATA_INDEX.csv` lists every data file included here with its size, SHA256, scientific owner, and use. The direct contrast audit input list identifies 404 exact source files and can be checked independently before running the script.

The repository emphasizes plot-ready data, final tabular summaries, and all inputs for the deterministic tariff-contrast audit. The companion source-data archive supplied with the manuscript contains the complete row-level records for the other simulation designs.
