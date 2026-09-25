# Extreme-demand tariff analysis

Simulation, model fitting, and source data for **Decision-Relevant Modelling of
Extreme Demand for Industrial Electricity Tariffs**, by Mingyu Huang,
Jiangweixi Wang, Zheng Gao, Yihang Yuan, Zhengjun Yang, and Huiqiong Li.

This study connects errors in demand distributions to the costs and stability
of industrial tariff decisions. The framework identifies the mean and stop-loss
functionals used by competing tariffs, establishes a sufficient condition for
mode stability, and bounds full-action regret. Paired simulations examine these
relationships; manufacturing-load forecasts provide an external assessment of
predictive performance.

The primary experiment uses **1,000 simulated training histories, five methods,
and nine tariff settings**, producing **45,000 tariff decisions**. The methods
include an event-level tail model (TAIL), an event-level empirical model
(EVENT-EMP), and GEV, kernel density (KDE), and empirical (EMP) models of
monthly maxima. Additional analyses vary event frequency, tariff geometry,
and operating regimes. A Gaussian reference and paired functional-error
analyses connect the forecast comparison to the tariff cost functionals.

## Explore the results

- [Summary tables](tables/README.md): model fidelity, tariff decisions,
  paired regret contrasts, and sensitivity analyses.
- [Source data](data/README.md): 67 indexed files covering simulation results,
  external prediction scores, and supplementary figure data.
- [Functional analyses](docs/FUNCTIONAL_ANALYSES.md): paired associations,
  a Gaussian reference, and independent contract-objective comparisons.
- [Experiment design](docs/DESIGN.md): estimation, tariff settings,
  uncertainty calculations, and sensitivity analyses.

The result files can be used without running the experiments. The
[data index](data/SUPPLEMENTARY_DATA_INDEX.csv) describes each file, and the
[figure map](data/FIGURE_SOURCE_MAP.csv) identifies the fields and display
transformations for Supplementary Figures S4–S6.

## Run the analyses

Use Python 3.12. From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the primary workflow with its small simulation run:

```bash
python scripts/run_revision_experiment.py --mode smoke
```

The [reproduction guide](docs/REPRODUCING.md) gives the commands, required
inputs, and output locations for each analysis. Full experiments require
substantial computation. Experiment settings and random seeds are supplied
in [configs/](configs/), with a [seed index](configs/seed_manifest.yaml).
See [workflow coverage](docs/RELEASE_SCOPE.md) for the available code and the
supplementary experiments supplied as result data without a complete run workflow.

For the external analysis, download the original manufacturing load data from
[Figshare Version 9](https://doi.org/10.6084/m9.figshare.14822256.v9).
The guide describes preprocessing with `scripts/prepare_external_inputs.py`.

## Repository structure

```text
configs/                 Experiment settings and random seeds
data/                    Source data, file index, and figure mappings
docs/                    Study design and instructions for running analyses
scripts/                 Experiment runners and result summaries
src/extreme_demand/       Demand models, estimators, and tariff calculations
tables/                  Selected manuscript result tables
requirements.txt         Python dependencies
LICENSE                  MIT license
```

## Citation

When using this work, cite the accompanying manuscript by its title and
authors above, and include the repository URL. The [data index](data/README.md)
and [seed manifest](configs/seed_manifest.yaml) identify the supplied results
and random-number streams.

## License and contact

The repository is distributed under the [MIT license](LICENSE). The external
manufacturing dataset has its own terms at the linked source.

Corresponding author: Professor Huiqiong Li, Yunnan University
([lihuiqiong@ynu.edu.cn](mailto:lihuiqiong@ynu.edu.cn)).
