# Extreme-demand tariff decisions: CSSC reproducibility record

This repository accompanies *When Do Extreme-Demand Models Improve Tariff Decisions? A Computational Comparison*. It contains the submitted article, the result records needed to audit its main numerical claims, and one verification command.

| Folder | Contents |
|---|---|
| [article/main/](article/main/) | Main manuscript PDF and LaTeX source |
| [article/supplement/](article/supplement/) | Supporting Information PDF, source, figures, and tables |
| [results/](results/README.md) | Primary comparisons, Wasserstein checks, event-level ablation, contract diagnostics, and external charge summaries |
| [scripts/](scripts/) | Independent checks of record integrity and reported summaries |

## Verify the released results

Python 3.12 was used for the analysis. From the repository root:

    python -m pip install -r requirements.txt
    python scripts/verify_results.py

The script checks file hashes; recomputes the five-method Wasserstein means and numerical pass counts from 50,000 law records and 450,000 decision records; recomputes paired event-level ablation means and 95% intervals from 540,000 rows; and cross-checks the focal GEV–KDE contrasts, common high-p* cells, and contract-action diagnostics. It prints a JSON report with status PASS when these checks agree.

The complete Monte Carlo histories, fitted-law checkpoints, and full experiment scripts are supplied in the journal submission data archives. This compact repository supports independent verification of the released results; rerunning model fitting from the first simulated history requires those archives. The external manufacturing-load analysis uses an illustrative common tariff to calculate hypothetical charges.

## Article files and provenance

The [main manuscript](article/main/main.pdf) contains the GitHub link in its Data and Code Availability statement. The [Supporting Information](article/supplement/supporting_information.pdf) contains the derivations and experimental specifications. SHA256SUMS.csv records hashes for the article PDFs, result records, script, and environment file. The release is linked to the CSSC submission prepared on 19 November 2026.

The repository uses the MIT license for its original code. Third-party data retain their original terms.
