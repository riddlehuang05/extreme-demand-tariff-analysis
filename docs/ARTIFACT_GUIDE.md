# Artifact guide

This guide maps the article's reported outputs to the files supplied in this repository.

| Reported material | Repository location |
|---|---|
| Main article | `manuscript/main/Manuscript.pdf` |
| Main figures | `manuscript/main/figures/Figure_1.pdf`–`Figure_4.pdf` |
| Supporting Information | `manuscript/supplementary/Supporting_Information.pdf` |
| Supplementary figures | `manuscript/supplementary/figures/FigureS1.pdf`–`FigureS8.pdf` |
| Supplementary tables | `manuscript/supplementary/tables/` and `tables/` |
| Primary and robustness analysis records | `data/source/` and `data/numerical_audit/` |
| Figure-ready data | `data/plot/` |
| Tariff-contrast stability audit | `code/build_contrast_stability_package.py`; see [`REPRODUCING.md`](REPRODUCING.md) |

The controlled analysis contains 10,000 paired histories, five predictive methods, nine tariff cells, and 450,000 method–history–cell decisions. Regional regret contrasts use history-paired Student t intervals (9,999 degrees of freedom). External predictive-score comparisons use paired Student t intervals across eight factory-level differences (7 degrees of freedom).

The external analysis evaluates predictive scores and hypothetical tariff charges using observed manufacturing-load records. It does not evaluate realized savings from actual tariff bills. The repository's external-data files contain score summaries and paired comparisons, not the underlying facility time series.

The repository-root manifest and checksum file support verification of the supplied materials.
