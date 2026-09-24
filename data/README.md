# Source data

This directory contains 58 scientific data files accompanying the manuscript:
32 analysis outputs, 23 derived files, and three supplementary figure-source
files. [SUPPLEMENTARY_DATA_INDEX.csv](SUPPLEMENTARY_DATA_INDEX.csv) lists each
file's purpose, dimensions, and explanatory notes. File paths
in the index are relative to this directory.

## Contents

| Directory or file | Contents |
| --- | --- |
| `formal/` | Analysis summaries, convergence records, external prediction scores and fitting diagnostics, crossed-tariff design, and correlation intervals |
| `derived/` | Plotting data, synthetic history-level records, regional contrasts, and external summaries |
| `supplementary/` | Source records for Figures S4–S6 |
| `FIGURE_SOURCE_MAP.csv` | Fields, filters, series labels, and display transformations for the 23 panels of Figures S4–S6 |

The primary simulation data include 45,000 tariff decision records and 5,000
model-fit records. Selected aggregate results are also described in
[manuscript tables](https://github.com/riddlehuang05/extreme-demand-tariff-analysis/blob/main/tables/README.md). The plot helper files
`figure1_cell_plot.csv` and `figure2_crossed_plot.csv` contain only part of
the data used in the corresponding main figures; consult the full data index
for the decision records and crossed-tariff summaries.

## External prediction data

External losses use demand divided by the mean monthly maximum in the
corresponding training period. The derived records include anonymized factory
labels, rolling-origin months, normalization constants, and observed maxima.
Raw high-frequency time series are available from
[Lee et al., Scientific Data (2022)](https://doi.org/10.1038/s41597-022-01357-8),
using the [Version 9 data release](https://doi.org/10.6084/m9.figshare.14822256.v9).

Some files retain `q95_exceedance_brier` and `q99_exceedance_brier` fields or
metric rows. These are method-specific exceedance diagnostics with different
events across methods. The manuscript's cross-method score comparisons use
CRPS, tail-weighted CRPS, and pinball losses; Table S7 additionally reports
method-specific exceedance counts. The retained exceedance diagnostic fields
are not common-event scores for comparing methods.

## Supplementary Figures S4–S6

| Figure | File under `supplementary/` | Records |
| --- | --- | ---: |
| S4: parameter recovery | `figure_s4_parameter_recovery.csv` | 600 |
| S5: clipping sensitivity | `figure_s5_clipping_sensitivity.csv` | 3,200 |
| S6: physical misspecification | `figure_s6_physical_misspecification.csv` | 5,200 |

S4 includes 200 histories at each of 12, 24, and 60 training months. S5 contains
paired regret differences for two baselines, four clipping settings, and two
training lengths. S6 contains 200 histories per scenario, training length,
and available method. The No-decl. method is included for `T6_CLUSTERED` only;
it is not an omitted series in T4 or T5.

The `delta` and `regret_scaled` columns already include the factor of 1,000
used on the figure axes. Apply the transformations in
[FIGURE_SOURCE_MAP.csv](FIGURE_SOURCE_MAP.csv) without scaling these columns again.
The [workflow coverage](https://github.com/riddlehuang05/extreme-demand-tariff-analysis/blob/main/docs/RELEASE_SCOPE.md) describes the available
code and the original configurations that are not included for these experiments.

## File formats

CSV files use UTF-8. `primary_rows.csv.gz` is a gzip-compressed CSV.
Paths in the figure map are relative to this directory.

This README, the data index, and the figure map are documentation files in
addition to the 58 scientific data files.
