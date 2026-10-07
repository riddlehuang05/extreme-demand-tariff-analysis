# Extreme-demand tariffs: final manuscript and reproducibility package

This repository accompanies *Decision-Relevant Modelling of Extreme Demand for Industrial Electricity Tariffs*.

Industrial electricity tariffs translate uncertainty in monthly peak demand into different economic actions. This study links predictive distributions to the tariff functionals they affect, the optimized tariff mode, and the resulting full-action regret.

## Research design and findings

The final controlled analysis uses 10,000 shared 24-month histories, five predictive methods, and nine tariff settings, yielding 450,000 method–history–cell decisions. The five methods are TAIL, EVENT-EMP, GEV, KDE, and EMP.

The theory characterizes contract demand through a tariff-implied constrained Expected Shortfall, separates full-action regret into mode-selection loss and within-contract excess, and gives competitor-specific and direct tariff-contrast stability bounds. A scalar mode-value certificate provides a compact stability summary; the contrast formulation retains the shared effect of a predictive distribution across competing tariff costs.

The analysis reports signed functional associations using simultaneous familywise intervals, compares methods within matched information sets, and examines the findings under high-p* tariffs, fitting-support sensitivity, a joint negative-binomial–Weibull challenge, and physical-pipeline stress. In the monthly-maxima information set, the GEV–KDE mean-regret contrast is −3.887 thousand CNY/month in the contract-demand region and +5.470 thousand CNY/month in the capacity region.

Rolling forecasts from eight eligible manufacturing facilities provide an external comparison of predictive scores. Economic tariff regret is evaluated in the controlled tariff experiments.

## What is included

- The final main manuscript and Supporting Information LaTeX source, their review-facing PDFs, and the final Figure 1–4 and Figure S1–S8 PDF assets.
- The final upload tables, machine-readable table sources, figure-ready data, and compact source summaries.
- The deterministic tariff-contrast stability audit, with the saved source inputs required to rerun its numerical checks and Figure S3 outputs.
- A filtered data index and a file-level SHA256 manifest for this release.

The package is arranged so the manuscript, figures, tables, and supporting data can be checked together. The companion source-data archive distributed with the submission retains the full decision-level records for the other reported experiments.

## Reproduce the deterministic stability audit

Use Python 3.12 and install the dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m py_compile code/build_contrast_stability_package.py
python code/build_contrast_stability_package.py --root .
```

The script reads saved oracle values, numerical-audit rows, and plotting inputs. It draws no random numbers, generates no histories, and fits no predictive laws. It writes the tariff radii, numerical coverage summaries, smoke checks, Expected-Shortfall identity audit, and Figure S3 files under `data/derived_publication/tariff_contrast_wasserstein/`.

`data/derived_publication/tariff_contrast_wasserstein/source_input_hashes.csv` records the relative paths, sizes, and hashes of all 404 script inputs. The filtered `data/SOURCE_DATA_INDEX.csv` describes the data included in this repository.

## Manuscript files

- Main article: `manuscript/main/Manuscript.tex`
- Supporting Information: `manuscript/supplementary/Supporting_Information.tex`
- Main figure PDFs: `manuscript/main/figures/`
- Supplementary figure PDFs: `manuscript/supplementary/figures/`
- Main upload tables: `manuscript/main/upload_tables/`
- Supporting Information table sources: `manuscript/supplementary/tables/`

See [the alignment note](docs/FINAL_ALIGNMENT.md) for the manuscript-to-data crosswalk and review criteria reflected in the final release.

## Citation and license

Cite the accompanying manuscript and this repository when using the materials. Code and data files are distributed under the MIT License; third-party data retain their original terms.
