# Reproduction guide

## Final paper assets

The `manuscript/` tree contains the final LaTeX source, bibliography, Wiley class and style files, upload tables, canonical figure PDFs, and review-facing manuscript/SI PDFs. The source trees compile from their own directories using the supplied dependencies.

## Deterministic tariff-contrast audit

From the repository root, install the pinned environment and run:

```bash
python -m pip install -r requirements.txt
python code/build_contrast_stability_package.py --root .
```

This is deterministic post-processing of saved final analysis records. The script verifies the 450,000 primary decisions, checks both competitor-specific gaps for every decision, computes analytic tariff radii, and writes numerical coverage and smoke-audit outputs. It does not generate histories, refit predictive laws, or rerun primary tariff decisions.

The script consumes 404 saved files listed in `data/derived_publication/tariff_contrast_wasserstein/source_input_hashes.csv`. It also uses the final Figure S3 source tables in `data/plot/r5/` and compact source tables under `data/source/`.

## Figure, table, and inference sources

- Main Figures 1–4 are supplied as canonical PDFs; machine-readable plot data are in `data/plot/`.
- Supplementary Figures S1–S8 and Tables S1–S14 are supplied with their LaTeX source. Numeric and display data are in `tables/`.
- Primary regional contrasts use history-paired Student t intervals (df = 9,999); external score comparisons use factory-level paired Student t intervals (df = 7). The separate simultaneous-correlation and threshold-fixed bootstrap procedures remain documented in the final article and Supporting Information.

The package manifest and SHA256 list make file-level integrity checks straightforward. The full source-data archive accompanying the submission provides the complete decision-level records for the reported simulation designs.
