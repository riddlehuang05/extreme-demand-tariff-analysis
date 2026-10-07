# Alignment with the final manuscript and review criteria

This release presents one coherent publication state: the main article, Supporting Information, canonical figures, table sources, plotting data, and deterministic stability audit share the same final numerical baseline.

## Manuscript-to-data crosswalk

| Manuscript component | Final source in this repository | Alignment |
|---|---|---|
| Main text and references | `manuscript/main/Manuscript.tex`, `introduction_references.bib` | Extracted from the final main LaTeX source package |
| Main Figures 1–4 | `manuscript/main/figures/Figure_1.pdf`–`Figure_4.pdf` | SHA256 matches the final submission manifest |
| Main Tables 1–3 | `manuscript/main/upload_tables/` | Copied from the final upload table set |
| Supporting Information and Figures S1–S8 | `manuscript/supplementary/` | Extracted from the final SI source package |
| Supporting Tables S1–S14 | `manuscript/supplementary/tables/` and `tables/` | LaTeX and machine-readable sources accompany the final table ownership |
| Direct tariff-contrast stability results | `code/build_contrast_stability_package.py`, `data/numerical_audit/`, `data/derived_publication/tariff_contrast_wasserstein/` | Deterministic source inputs and final numerical outputs are included |

## Theory reflected in the final text

- Contract demand is represented by a constrained Expected-Shortfall functional at the tariff-implied probability level, with the ordinary Expected-Shortfall identity applying when the quantile set intersects the feasible threshold interval.
- Full-action regret is decomposed exactly into tariff-mode selection loss and within-contract excess, including boundary optima and nonunique contract solutions.
- Mode-preservation theory retains competitor-specific gaps and errors, alongside the scalar (d_{MV}) summary.
- Mode-specific Wasserstein radii distinguish the capacity, actual-demand, and contract-demand values. Direct tariff-contrast bounds preserve the common distributional perturbation across competing optimized costs.
- Numerical optimization enters through an objective-suboptimality allowance. The numerical audit reports its implementation as a floating-point bound that is not interval-certified.

## Statistical and empirical alignment

- The primary controlled analysis uses 10,000 paired histories, five methods, and nine tariff cells (450,000 decisions). Regional mean-regret comparisons use history-paired Student t 95% intervals with df = 9,999.
- The functional-error analysis compares mean and tariff-specific stop-loss errors with the q0.99 comparator using ten simultaneous familywise intervals. All ten intervals for the signed correlation differences are positive.
- Same-information method comparisons are distinguished from complete-procedure comparisons. The GEV–KDE contrast within the monthly-maxima information set changes sign between the contract-demand and capacity regions.
- The final robustness evidence includes high-p* tariff stress, fitting-support sensitivity, the joint negative-binomial–Weibull challenge, and physical-pipeline stress. The expanded crossed tariff design evaluates 76 tariff cells.
- External results describe predictive scores across eight eligible manufacturing facilities. Their uncertainty is summarized at the factory level; tariff economic regret is assessed in the controlled experiments.

## Teacher and Sonnet review crosswalk

| Review topic | Final manuscript treatment represented here |
|---|---|
| Contract boundaries, atoms, and the tariff-implied quantile | Constrained optimization statements distinguish interior quantile conditions from boundary solutions; the exact within-contract loss is stated separately. |
| Mode errors versus continuous contract-level loss | Exact regret decomposition and contract-correct residual analysis separate the discrete and continuous parts of the decision. |
| Certificate interpretation and numerical accuracy | Pairwise mode-value and Wasserstein checks are distinguished; numerical bounds and the solver allowance are described with their actual certification status. |
| Ranking changes and information sets | Regional regret contrasts are reported for matched information sets where available, and whole-procedure comparisons are identified as such. |
| Baseline design and robustness | The baseline controlled DGP is complemented by targeted count–severity and physical-pipeline challenges, plus high-p* and support sensitivity analyses. |
| Replication uncertainty and rare tariff regions | Primary inference uses 10,000 paired histories; the 76-cell crossed design shows the broader tariff geometry. |
| External-data interpretation | Factory-level score comparisons are reported as predictive evidence, with no conversion into observed tariff savings. |

The reported numerical checks describe observed coverage in the final analysis records. They are presented as numerical sufficient-condition checks, with the corresponding implementation uncertainty stated alongside them.
