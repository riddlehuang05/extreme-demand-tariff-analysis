# Supplementary source data

This package contains 58 existing research-output files accompanying the manuscript: 32 formal outputs, 23 derived files, and three supplementary figure-source files. The root-level `SUPPLEMENTARY_DATA_INDEX.csv` lists each file, its role, record count, byte size, provenance and SHA-256 checksum. Resolve every `file` entry relative to this directory.

## Contents

- `formal/`: frozen summaries, convergence records, origin-level external scores and fitting diagnostics, crossed-tariff design, and history-bootstrap correlation intervals.
- `derived/`: descriptive plotting data, synthetic history-level records, regional contrasts and external summaries.
- `supplementary/`: the frozen source records for Figures S4–S6.
- `FIGURE_SOURCE_MAP.csv`: fields, filters, series labels, and display transformations for all 23 panels of Figures S4–S6. Paths in this map are relative to this directory.

The files are unchanged copies of the frozen V1.1 analysis outputs and the V1.4 narrative figure-source snapshot. No simulations, model fitting or inferential analyses were rerun when assembling this package. The 45,000 primary decision records and 5,000 fit-level records agree with the final authoritative results. All six source hashes recorded by the figure-data preparation script match the authoritative data.

## Relationship to the manuscript

The final main figures use the same underlying data as the narrative source package. Main Figures 1 and 3 and Supplementary Figures S1–S6 match that package's PDF assets byte for byte. Main Figure 2 has a later layout correction. Main Figure 4 has a later score-panel revision; its four loss measures agree with the final Table S7 to the reported six decimal places. `figure1_cell_plot.csv` and `figure2_crossed_plot.csv` are retained plot helpers from this source package; they are not the sole data source for the final narrative figures.

Some frozen external files contain fields named `q95_exceedance_brier` or `q99_exceedance_brier`, or metric rows with those names. These are method-specific exceedance diagnostics retained in the source records. The final manuscript compares CRPS, tail-weighted CRPS and pinball losses across methods, and reports method-specific exceedance counts in Table S7. The retained diagnostic fields do not define an additional common-event scoring comparison.

External losses are calculated after dividing demand by the mean monthly maximum in the corresponding training period. Some external files include anonymized factory labels, rolling-origin month labels, normalization constants and derived observed maxima; they contain no raw high-frequency factory time series. The public source dataset is Lee et al., Scientific Data (2022), https://doi.org/10.1038/s41597-022-01357-8.

## Integrity

`sha256` in the index applies to each file as stored, including the compressed bytes of `primary_rows.csv.gz`. The `provenance` field identifies the frozen source collection rather than a second file path to resolve. UTF-8 CSV files can be read directly; `primary_rows.csv.gz` is a gzip-compressed CSV.

## Figures S4–S6

| Figure | File under `supplementary/` | Records |
| --- | --- | ---: |
| S4: parameter recovery | `figure_s4_parameter_recovery.csv` | 600 |
| S5: clipping sensitivity | `figure_s5_clipping_sensitivity.csv` | 3,200 |
| S6: physical misspecification | `figure_s6_physical_misspecification.csv` | 5,200 |

These are the actual frozen inputs to the published figure assets. Their
checksums match both the narrative figure-source snapshot and the earlier
frozen source manifest. S4 reports 200 histories at each of 12, 24, and 60
training months. S5 contains paired regret differences for two baselines,
four clipping settings, and two training lengths. S6 contains 200 histories
per scenario, training length, and available method. The No-decl. method is
present in T6_CLUSTERED; it is not a missing series in T4 or T5.

The `delta` and `regret_scaled` columns already include the factor of 1,000
used in the figure axes. The mapping records this convention to prevent
double scaling. No numerical values were changed when these files were
renamed and packaged. The data index and figure map are metadata and are
not counted among the 58 research-output files.
