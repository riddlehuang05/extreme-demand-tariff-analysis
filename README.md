# Extreme-demand tariff analysis

Code accompanying **Decision-Relevant Modelling of Extreme Demand for Industrial
Electricity Tariffs**, by Mingyu Huang, Jiangweixi Wang, Zheng Gao, Yihang Yuan,
and Zhengjun Yang.

This repository provides the V1.1 computational pipeline used in the V1.4
manuscript: simulation, distribution fitting, tariff decisions, stress
experiments, external predictive evaluation, and random seeds. The primary
design uses **1,000 training histories, five methods, and 45,000 decisions**.
The frozen source data underlying the main and supplementary results are
available in [data/](data/README.md), alongside five convenient aggregate tables.

The manuscript snapshot is tagged
[`v1.4-manuscript`](https://github.com/riddlehuang05/extreme-demand-tariff-analysis/tree/v1.4-manuscript).
Its data index contains 58 result files, including the source records and
panel mappings for Supplementary Figures S4–S6.

## Repository layout

```text
configs/                 Experiment settings and seed_manifest.yaml
data/                    Frozen result data, figure mappings, and checksums
docs/                    Design, reproduction guide, and release scope
scripts/                 Runnable experiments, summaries, and checks
src/extreme_demand/       Shared models, estimators, and tariff calculations
tables/                  Five published aggregate result tables
requirements.txt         Python dependencies
SHA256SUMS.csv            Release file hashes
LICENSE                  MIT license
```

## Getting started

Use Python 3.12 and run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/check_public_tables.py
```

The table check verifies presence and nonempty content; it does not rerun the
experiments or independently validate their numerical results.

For simulation and model fitting, follow the ordered commands in
[Reproducing the analyses](docs/REPRODUCING.md). Full experiments require
substantial computation.

## Data and results

The five files in [tables/](tables/README.md) contain the final primary
performance summaries, paired contrasts, and sensitivity summaries.
The [source-data index](data/SUPPLEMENTARY_DATA_INDEX.csv) links to frozen
synthetic history-level records, formal summaries, external derived scores,
and figure-source data. File paths in that index are relative to `data/`.
Synthetic histories can also be regenerated from the supplied code and seeds.

The external load data must be obtained from the cited
[Figshare Version 9 release](https://doi.org/10.6084/m9.figshare.14822256.v9).
The preprocessing entry point is `scripts/prepare_external_inputs.py`.
The original dataset is not redistributed here.

## Documentation

- [Experiment design](docs/DESIGN.md)
- [Reproduction commands and dependencies](docs/REPRODUCING.md)
- [Included analyses and limitations](docs/RELEASE_SCOPE.md)
- [Expected output sizes and frozen hashes](docs/FROZEN_OUTPUT_TARGETS.md)
- [Random seeds](configs/seed_manifest.yaml)

The earlier public companion is preserved under the
[`v1.0-companion` tag](https://github.com/riddlehuang05/extreme-demand-tariff-analysis/tree/v1.0-companion).
Use the current release for the 1,000-history manuscript.
