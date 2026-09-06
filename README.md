# Decision-Relevant Modelling of Extreme Demand for Industrial Electricity Tariffs

Selected public analysis code and derived numerical outputs for the manuscript:

**Mingyu Huang, Jiangweixi Wang, and Yihang Yuan.**  
*Decision-Relevant Modelling of Extreme Demand for Industrial Electricity Tariffs.*  
Applied Stochastic Models in Business and Industry.

## Overview

The paper studies when improved modelling of industrial electricity-demand extremes
becomes relevant to downstream tariff decisions. The public repository is intentionally
compact. It provides selected derived analysis outputs and non-core statistical summary
scripts supporting the manuscript-facing numerical results.

The repository does **not** contain rendered manuscript figures. The published article
and Supporting Information are the authoritative sources for figure and table presentation.

## Repository layout

```text
code/
  run_public_analysis.py   Recompute selected public statistical summaries
  check_release.py         Check row counts and headline quantities
  README.md

data/
  q99_errors.csv
  regret_differences.csv
  diagnostic_units.csv
  confirmatory_regret_summary.csv
  diagnostic_summary.csv
  clipping_summary.csv
  misspecification_summary.csv
  external_crps.csv
```

## Requirements

- Python >= 3.10
- See `requirements.txt`

Install with:

```bash
pip install -r requirements.txt
```

## Run the public analysis

```bash
python code/check_release.py
python code/run_public_analysis.py
```

The second command creates a local `results/` directory containing compact CSV/JSON
summaries. No PNG/PDF figure files are generated or distributed by this repository.

## Public data scope

The CSV files are derived, manuscript-facing analysis outputs rather than raw research
inputs. They cover selected q0.99-error results, paired regional regret differences,
diagnostic/regret analysis units, confirmatory regret summaries, robustness summaries,
and external-factory CRPS values.

The original South Korean manufacturing-load dataset is not redistributed here; it is
publicly available from the source cited in the manuscript.

## Implementation scope

Core research implementation is not part of this public package. In particular, the
repository does not contain the synthetic demand generator, TAIL/KDE/EMP fitting
internals, tariff-action optimization engine, full Monte Carlo orchestration, robustness
simulation engines, or the external-data fitting pipeline.

Accordingly, this repository should be described as a **selected analysis/data release**,
not as a full end-to-end reproduction archive.

## License

See `LICENSE`.

## Citation

Please cite the accompanying article. Add the final DOI and bibliographic details after
publication.
