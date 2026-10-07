# Reproducing the deterministic tariff-contrast audit

## Environment

The supplied environment is pinned in `requirements.txt` and was prepared for Python 3.12.10. Run the commands below from the repository root.

```bash
python -m pip install -r requirements.txt
python code/build_contrast_stability_package.py
```

To render main Figure 3 from the supplied analysis records, run:

```bash
python code/plot_main_figure3.py
```

## What the command does

The script uses saved analysis records to reproduce the deterministic audit of tariff-contrast stability. It reads:

- the nine-cell oracle mode-value table;
- the saved primary decision and competitor-level numerical-audit records;
- the saved Wasserstein, primitive-perturbation, and support-sensitivity summaries.

It verifies the expected primary design dimensions, calculates the analytic tariff radii and numerical sufficient-condition coverage, writes the competitor-pair summary, and runs smoke checks for contrast bounds and the constrained Expected Shortfall identity. It also regenerates the Figure S3 plotting outputs from the supplied source tables. The inputs used and their SHA256 values are recorded in `data/derived_publication/tariff_contrast_wasserstein/source_input_hashes.csv`.

The generated audit tables and figure files are written under:

```text
data/derived_publication/tariff_contrast_wasserstein/
```

## Scope and interpretation

This is deterministic post-processing of saved records. It generates no histories, refits no predictive distributions, and reruns no primary tariff decisions. It therefore reproduces the stated stability audit, not the complete simulation and fitting pipeline for every result in the article.

The solver allowance is reported as a numerical objective-suboptimality bound. Its floating-point evaluation is not certified using directed rounding or interval arithmetic; the audit must not be interpreted as a machine-certified bound.

## Main Figure 3

`code/plot_main_figure3.py` reads the primary method-by-cell summary, the nine-cell oracle table, and the two supplied conditional-regret tables. It renders the four-panel main Figure 3 without rerunning the controlled experiment. The files are written to `data/derived_publication/main_figure3/` by default; an alternate directory can be supplied with `--output-dir`.

## Data and checksums

`data/SOURCE_DATA_INDEX.csv` describes the supplied source and analysis files. `FINAL_REPOSITORY_MANIFEST.json` lists the release contents, and `FINAL_REPOSITORY_SHA256.csv` provides file-level checksums. The script also records hashes for its direct inputs in the generated output directory.

External-facility results are supplied as predictive-score summaries and paired comparisons. The underlying facility time series are not included in this repository.
