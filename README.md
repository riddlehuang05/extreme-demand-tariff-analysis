# Data and code for *Decision-Relevant Modelling of Extreme Demand for Industrial Electricity Tariffs*

This repository accompanies the article with machine-readable supplementary tables, analysis data, code, and reproducibility documentation. The manuscript and Supporting Information are supplied through the journal submission materials.

## Study at a glance

The controlled study uses 10,000 paired 24-month histories, five predictive methods, and nine tariff settings, for 450,000 method–history–cell decisions. The theory expresses contract-demand optimization through a constrained Expected Shortfall, separates mode-selection loss from residual contract-level loss, and derives direct Wasserstein bounds for competing tariff-cost contrasts. An external rolling-forecast analysis covers eight eligible manufacturing facilities and compares predictive scores and hypothetical charges; it does not estimate savings from observed tariff bills.

## Repository contents

| Path | Contents |
|---|---|
| `manuscript/supplementary/tables/`, `tables/` | Machine-readable numeric and plotting data for the reported tables |
| `data/` | Analysis records, experiment summaries, and plot data; see [`data/SOURCE_DATA_INDEX.csv`](data/SOURCE_DATA_INDEX.csv) |
| `code/` | Deterministic post-processing for the tariff-contrast stability audit |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | Environment, command, inputs, outputs, and scope of the reproducible audit |
| `FINAL_REPOSITORY_MANIFEST.json`, `FINAL_REPOSITORY_SHA256.csv` | File inventory and SHA256 checksums |

## Reproduce the deterministic audit

From the repository root, using Python 3.12.10:

```bash
python -m pip install -r requirements.txt
python code/build_contrast_stability_package.py
```

The script reads the supplied analysis records and writes audit tables and Figure S3 outputs in PNG, SVG, EPS, and TIFF formats under `data/derived_publication/tariff_contrast_wasserstein/`. It checks decision rows, computes analytic tariff radii and numerical coverage summaries, and runs deterministic smoke and Expected Shortfall identity checks. This command reproduces the stability audit; it does not regenerate the simulation histories, refit predictive distributions, or rerun the primary tariff decisions. The solver allowance is a numerical bound, not an interval-arithmetic certificate. See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the command's inputs and outputs.

## Data and integrity

The source-data index identifies the supplied analysis records and their role. External-facility materials in this repository are predictive-score summaries and paired comparisons; the underlying facility time series are not redistributed here. Use the repository manifest and checksum file to verify the supplied files.

## Citation and license

Please cite the accompanying article when using these materials. The repository includes an MIT license; any third-party data remain subject to their source terms.
