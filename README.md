# Extreme-demand models and tariff decisions

This repository accompanies *When Do Extreme-Demand Models Improve Tariff Decisions? A Computational Comparison*, prepared for *Communications in Statistics—Simulation and Computation*. The study compares five predictive distributions on paired demand histories and evaluates how their differences affect tariff-mode selection, contract demand, and full-action regret.

The [main manuscript](article/main/main.pdf) and [Supporting Information](article/supplement/supporting_information.pdf) are available as PDFs. Their LaTeX sources, tables, and figures are in [article/](article/).

## Reproduce the reported summaries

Python 3.12 is required. From the repository root:

    python -m pip install -r requirements.txt
    python scripts/reproduce_results.py

The command reads the released result records and reproduces the method-level Wasserstein summaries, numerical decision-check counts, paired event-level ablation estimates and intervals, and focal GEV–KDE comparisons. It also checks the released file hashes. A successful run prints a JSON report with status PASS.

The [result guide](results/README.md) maps the manuscript's main tables and figures to their source files. The [reproduction directory](reproduction/README.md) contains the estimators, simulation runners, fixed-seed history generator, external validation runner, and audit scripts. A [versioned release](https://github.com/riddlehuang05/extreme-demand-tariff-analysis/releases/tag/cssc-submission-v1) provides the analysis archive. The released result records allow the reported numerical summaries to be checked directly; the full Monte Carlo run can be regenerated when the history generator and primary runner are used together. For Table S13, use `results/solver_allowance/full_allowance_rows.parquet` and `results/solver_allowance/coverage.csv`; for Table S15, use `results/tail_ablation/paired_summary.csv`. The [reproduction instructions](reproduction/README.md) map these records to the analysis scripts.

## Repository contents

| Path | Contents |
|---|---|
| [article/main/](article/main/) | Manuscript PDF, LaTeX source, bibliography, and figures |
| [article/supplement/](article/supplement/) | Supporting Information PDF, source, tables, and figures |
| [results/](results/README.md) | Analysis records organized by experiment |
| [scripts/reproduce_results.py](scripts/reproduce_results.py) | Reproduction command |
| [reproduction/](reproduction/) | Fixed-seed simulation and audit code |
| [requirements.txt](requirements.txt) | Python dependencies |
| [SHA256SUMS.csv](SHA256SUMS.csv) | File checksums |

The manufacturing-load example calculates charges under an illustrative common tariff. Code in this repository is released under the [MIT License](LICENSE). Third-party data retain their original terms.
The full solver-allowance recomputation and its source hashes are in [results/solver_allowance/](results/solver_allowance/).
