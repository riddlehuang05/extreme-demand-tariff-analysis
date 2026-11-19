"""Compose a CSSC review manuscript from the frozen V2.0 source and verified addendum."""
from __future__ import annotations

import shutil
import zipfile
import json
from pathlib import Path
import subprocess
import sys
from compress_cssc_main import compress
from polish_cssc_narrative import polish

ROOT = Path(__file__).resolve().parents[2]
ORIG = ROOT / "WILEY_SUBMISSION_FINAL-V1.2改" / "02_Manuscript"
OUT = ROOT / "CSSC_SUBMISSION_20261112" / "Manuscript"
TEMPLATE = ROOT / "exp1" / "data" / "InteractAPALaTeX"


def replace_exact(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected one source anchor; found {text.count(old)}: {old[:80]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ("interact.cls",):
        shutil.copy2(TEMPLATE / filename, OUT / filename)
    shutil.copy2(ORIG / "introduction_references.bib", OUT / "references.bib")
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    for path in (ORIG / "figures").glob("Figure_*.pdf"):
        shutil.copy2(path, figures / path.name)

    source = (ORIG / "Manuscript.tex").read_text(encoding="utf-8")
    body_start = source.index("\\section{Introduction}")
    body = source[body_start:]
    body = replace_exact(body, "\\bibliographystyle{unsrt}\n\\bibliography{introduction_references}", "\\bibliographystyle{apalike}\n\\bibliography{references}")
    body = body.replace("\\cite{", "\\citep{")

    old = ("We study this problem through a three-mode industrial tariff. We\n"
           " characterize contract demand through a constrained Expected Shortfall\n"
           "representation at a tariff-implied probability level and decompose\n"
           "full-action regret exactly into mode-selection loss and any residual\n"
           "within-contract excess. We then analyze the cost differences that\n"
           "determine mode selection and derive contrast-specific Wasserstein\n"
           "stability margins that retain the shared\n"
           "response of competing tariff costs to the predictive distribution. We\n"
           "examine this structure in paired controlled experiments, targeted\n"
           "stress designs, and rolling forecasts from manufacturing-load data.")
    new = ("We construct a controlled comparison of distribution estimators under a\n"
           "mixed tariff action. The experiment pairs fitted predictive laws within\n"
           "each demand history, fixes the tariff cells before method comparison,\n"
           "and evaluates both distributional functionals and the complete action\n"
           "under an analytic oracle. The tariff representation identifies the\n"
           "mean, stop-loss curve, and constrained Expected Shortfall as the\n"
           "functionals that control the action; exact regret decomposition and\n"
           "contrast-specific Wasserstein margins explain why estimator rankings\n"
           "change across tariff regions. We test those rankings across\n"
           "alternative demand mechanisms, sample lengths, and rolling forecasts\n"
           "from manufacturing-load data.")
    body = replace_exact(body, old, new)

    old = ("Economic rankings change with tariff geometry, including in\n"
           "comparisons where methods use the same information. Rankings also vary\n"
           "across demand mechanisms and predictive targets. The external\n")
    new = ("Economic rankings change with tariff geometry, including in\n"
           "comparisons where methods use the same information. The external\n")
    body = replace_exact(body, old, new)
    fidelity_start = body.index("\\captionof{table}{Distributional and tariff-functional fidelity")
    fidelity_start = body.rfind("\\clearpage", 0, fidelity_start)
    fidelity_end = body.index("\\clearpage", fidelity_start + len("\\clearpage"))
    fidelity_table = body[fidelity_start:fidelity_end]
    fidelity_table = fidelity_table.replace("\\label{tab:table_2}", "\\label{tab:si-fidelity}")
    body = compress(body)
    body = polish(body)
    body = replace_exact(
        body,
        "contains machine-readable analysis records and deterministic code\nfor the tariff-contrast stability analysis.",
        "contains machine-readable analysis records and deterministic code\nfor the tariff-contrast stability analysis. The accompanying submission\narchive contains the primary history inputs, comparison rows, and\nexecutable code for the revised primary experiment."
    )
    body = replace_exact(
        body,
        "The \\textcolor{blue}{\\textbf{GitHub repository}}\n"
        "\\textcolor{blue}{\\url{https://github.com/riddlehuang05/extreme-demand-tariff-analysis}}\n"
        "contains machine-readable analysis records and deterministic code\n"
        "for the tariff-contrast stability analysis. The accompanying submission\n"
        "archive contains the primary history inputs, comparison rows, and\n"
        "executable code for the revised primary experiment.",
        "The versioned manuscript sources, scripts, result summaries, and audit\n"
        "manifests are available at\\newline\n"
        "\\url{https://github.com/riddlehuang05/extreme-demand-tariff-analysis}.\n"
        "The accompanying submission archive contains the 10,000 primary\n"
        "history inputs, revised decision and numerical-audit records,\n"
        "complete experiment scripts, external rolling-validation records,\n"
        "and SHA-256 manifests for verifying the reported results."
    )
    body = replace_exact(
        body,
        "choices, analyses, interpretation of results, and final manuscript\ncontent were reviewed and approved by the authors, who take full\nresponsibility for the work.",
        "choices, analyses, interpretation of results, and final submitted\ncontent remain the responsibility of the authors."
    )
    body = replace_exact(
        body,
        "OpenAI Codex was used as a supporting tool for coding, code checking,\nlanguage editing, and manuscript preparation. All methodological\nchoices, analyses, interpretation of results, and final submitted\ncontent remain the responsibility of the authors.",
        "OpenAI Codex assisted with computational scripts and manuscript\nrevision. The authors are responsible for the final content."
    )
    body = replace_exact(body, "\\section*{AI-Assisted Preparation}", "\\noindent\\textit{AI use.}")
    body = replace_exact(body, "\\noindent\\textit{AI use.}\n\nOpenAI Codex", "\\noindent\\textit{AI use.} OpenAI Codex")

    preamble = r"""\documentclass[suppldata]{interact}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx,booktabs,array,longtable,placeins,xurl}
\usepackage{xcolor,hyperref,caption}
\usepackage[authoryear,sort]{natbib}
\bibpunct[, ]{(}{)}{;}{a}{,}{,}
\usepackage[mathlines]{lineno}
\usepackage{setspace}
\doublespacing
\renewcommand\linenumberfont{\normalfont\scriptsize\sffamily}
\setlength\linenumbersep{10pt}
\setlength{\emergencystretch}{2em}
\providecommand{\tablebodyfont}{\small}
\providecommand{\tablefootnotefont}{\small}
\newtheorem{proposition}{Proposition}[section]
\newtheorem{corollary}[proposition]{Corollary}
\hypersetup{hidelinks}
\raggedbottom
\begin{document}
\articletype{RESEARCH ARTICLE}
\title{When Do Extreme-Demand Models Improve Tariff Decisions? A Computational Comparison}
\author{\name{Mingyu Huang\textsuperscript{a}\thanks{CONTACT Mingyu Huang. Department of Statistics and Data Science, School of Mathematics and Statistics, Yunnan University, Chenggong Campus, South Section of East Outer Ring Road, Chenggong District, Kunming, Yunnan 650500, China. Email: huangmingyu@stu.ynu.edu.cn}, Jiangweixi Wang\textsuperscript{a}, Zheng Gao\textsuperscript{a}, Yihang Yuan\textsuperscript{a}, and Zhengjun Yang\textsuperscript{b}}
\affil{\textsuperscript{a}Department of Statistics and Data Science, School of Mathematics and Statistics, Yunnan University, Kunming 650500, China; \textsuperscript{b}Department of Mathematics, School of Mathematics and Statistics, Yunnan University, Kunming 650500, China}}
\maketitle
\noindent\textbf{Short title:} Extreme-Demand Tariff Decisions
\begin{abstract}
Forecast comparisons for extreme demand can reverse after the predictive distribution is passed through a tariff decision. We develop a controlled computational comparison that evaluates distribution estimators at three linked levels: tariff-relevant functionals, the chosen charging mode and contract level, and full-action economic regret. A constrained Expected Shortfall representation and a contrast-specific stability analysis identify the distributional features governing these outcomes. Across 10,000 paired simulated histories, methods based on the same monthly maxima reverse their regret ranking across tariff regions, even though the data available to each method are identical. Rolling forecasts from eight manufacturing facilities illustrate how predictive-score rankings and hypothetical tariff-charge rankings vary with the evaluation target. The results provide a computational protocol for comparing extreme-demand estimators by the decisions they support.
\end{abstract}
\begin{keywords}
Monte Carlo comparison; extreme-value estimation; probabilistic forecasting; decision loss; electricity demand tariffs
\end{keywords}
\linenumbers
"""
    (OUT / "Manuscript.tex").write_text(preamble + body, encoding="utf-8")

    si_out = OUT.parent / "Supporting_Information"
    si_out.mkdir(parents=True, exist_ok=True)
    si_zip = ROOT / "WILEY_SUBMISSION_FINAL-V1.2改" / "07_LaTeX_Supplementary_File" / "Supporting_Information_LaTeX.zip"
    with zipfile.ZipFile(si_zip) as archive:
        for item in archive.infolist():
            if item.is_dir() or item.filename.startswith("wiley_template/"):
                continue
            destination = si_out / item.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(item))
    shutil.copy2(
        ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "data"
        / "derived_publication" / "tariff_contrast_wasserstein" / "FigureS3.pdf",
        si_out / "figures" / "FigureS3.pdf",
    )
    s10 = si_out / "tables" / "Table_S10.tex"
    s10_text = s10.read_text(encoding="utf-8")
    s10_text = replace_exact(s10_text,
        "p{68.49pt}>{\\raggedright\\arraybackslash}p{91.31pt}>{\\raggedright\\arraybackslash}p{114.14pt}>{\\raggedright\\arraybackslash}p{182.63pt}",
        "p{48pt}>{\\raggedright\\arraybackslash}p{64pt}>{\\raggedright\\arraybackslash}p{76pt}>{\\raggedright\\arraybackslash}p{132pt}")
    s10.write_text(s10_text, encoding="utf-8")
    s14 = si_out / "tables" / "Table_S14.tex"
    s14_text = s14.read_text(encoding="utf-8")
    s14_text = replace_exact(s14_text, "\\begin{tabular}{@{}lrrr@{}}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{@{}lrrr@{}}")
    s14_text = replace_exact(s14_text, "\\end{tabular}\n\\par\\smallskip", "\\end{tabular}}\n\\par\\smallskip")
    s14.write_text(s14_text, encoding="utf-8")
    s3 = si_out / "tables" / "Table_S3.tex"
    s3_text = s3.read_text(encoding="utf-8")
    s3_text = replace_exact(s3_text, "\\fontsize{8.5}{10.5}\\selectfont\\setlength{\\tabcolsep}{3.0pt}", "\\fontsize{7.5}{9.5}\\selectfont\\setlength{\\tabcolsep}{2.5pt}")
    s3_text = replace_exact(s3_text, "\\begin{tabular}{@{}llllllll@{}}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{@{}llllllll@{}}")
    s3_text = replace_exact(s3_text, "\\end{tabular}\n\\par\\smallskip", "\\end{tabular}}\n\\par\\smallskip")
    s3.write_text(s3_text, encoding="utf-8")
    shutil.copy2(TEMPLATE / "interact.cls", si_out / "interact.cls")
    si_source = (si_out / "Supporting_Information.tex").read_text(encoding="utf-8")
    si_source = replace_exact(si_source, si_source[:si_source.index("\\begin{document}")], r"""\documentclass[]{interact}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs,amsmath,amssymb,graphicx,url,placeins,enumitem,array,longtable,caption}
\providecommand{\tablebodyfont}{\small}
\providecommand{\tablefootnotefont}{\small}
\usepackage[mathlines]{lineno}
\renewcommand\linenumberfont{\normalfont\scriptsize\sffamily}
\setlength\linenumbersep{10pt}
\graphicspath{{./}{figures/}{figures/si/}}
\setlist[itemize]{leftmargin=1.5em,itemsep=2pt,topsep=3pt}
\setlength{\parskip}{1.5pt}
\raggedbottom
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}
""")
    si_source = replace_exact(si_source, "Decision-Relevant Modelling of Extreme Demand for\nIndustrial Electricity Tariffs", "When Do Extreme-Demand Models Improve Tariff Decisions?\nA Computational Comparison")
    si_source = replace_exact(si_source, "Applied Stochastic Models in Business and Industry", "Communications in Statistics -- Simulation and Computation")
    si_source = replace_exact(
        si_source,
        "\\text{Scenario} & \\text{Clusters/month} &\n\\text{Headroom fraction} & \\text{Consecutive-heat probability}",
        "\\text{Scenario} & \\text{Clusters/month} &\n\\text{Headroom} & \\text{Consecutive probability}"
    )
    si_source = replace_exact(
        si_source,
        "u\\in\\{0.300,0.350,0.375,0.400,0.425,0.450,0.475,\n0.500,0.550,0.600,0.625,0.650,0.700,0.750\\}.",
        "\\begin{aligned}u\\in\\{&0.300,0.350,0.375,0.400,0.425,0.450,0.475,\\\\\n&0.500,0.550,0.600,0.625,0.650,0.700,0.750\\}.\\end{aligned}"
    )
    si_source = replace_exact(
        si_source,
        "All methodological\nchoices, analyses, interpretation of results, and final manuscript\ncontent were reviewed and approved by the authors, who take full\nresponsibility for the work.",
        "Methodological choices, analyses, interpretation of results, and\nfinal submitted content remain the responsibility of the authors."
    )
    si_source = replace_exact(
        si_source,
        "OpenAI Codex was used as a supporting tool for coding, code checking,\nlanguage editing, and manuscript preparation. Methodological choices, analyses, interpretation of results, and\nfinal submitted content remain the responsibility of the authors.",
        "OpenAI Codex assisted with computational scripts and manuscript\nrevision. The authors are responsible for the final content."
    )
    si_source = replace_exact(si_source, "\\section*{AI-Assisted Preparation}", "\\noindent\\textit{AI use.}")
    si_source = replace_exact(si_source, "\\noindent\\textit{AI use.}\n\nOpenAI Codex", "\\noindent\\textit{AI use.} OpenAI Codex")
    si_source = replace_exact(si_source,
        "\\subsection*{S1.15 Numerical solver allowance}\n",
        "\\subsection*{S1.15 Numerical solver allowance}\n\n"
        "A separate deterministic smoke audit checks interior as well as\n"
        "boundary contract optima. For a Uniform(5,85) demand law,\n"
        "$\\alpha=1.2$, $\\kappa=2$, and $D\\in[20,50]$, the fitted-optimal\n"
        "contract level is $D^*=43.0556$, obtained from the constrained\n"
        "$p^*=7/12$ quantile. The reproducibility script evaluates this\n"
        "interior candidate explicitly; it also integrates the absolute\n"
        "difference of piecewise-linear CDFs exactly in its Uniform-law\n"
        "$W_1$ check.\n")
    revised_decisions = (ROOT / "CSSC_REVIEW_REVISION_20261112"
                         / "primary_cv_crps_analysis_20261115" / "analysis"
                         / "primary_decisions_enriched_10000.csv.gz")
    ties_path = ROOT / "CSSC_REVIEW_REVISION_20261112" / "event_emp_tie_audit.json"
    subprocess.run([
        sys.executable, str(ROOT / "CSSC_REVIEW_REVISION_20261112" / "audit_event_emp_ties.py"),
        "--input", str(revised_decisions), "--output", str(ties_path),
    ], check=True)
    ties = json.loads(ties_path.read_text(encoding="utf-8"))
    addendum = rf"""
\section*{{Appendix S2. Empirical-quantile agreement and action separation}}

The empirical 0.99 quantile of 24 observed monthly maxima is the
sample maximum. The EVENT-EMP construction, which uses the empirical
event-value law and empirical monthly event counts, also reaches the
largest observed event maximum at this probability. In the primary
archive, their estimated 0.99 quantiles agree exactly in
{ties['q99_exact_equal_rows']:,} of {ties['paired_decision_rows']:,}
paired method--history--cell rows. The two procedures select the same
tariff mode in {ties['same_mode_rows']:,} rows
({100 * ties['same_mode_fraction']:.2f}\%). Their estimated means
agree in {ties['mean_equal_rows']:,} rows and their full-action regrets
agree in {ties['regret_equal_rows']:,} rows. Thus the upper-quantile
equality is a property of the finite empirical supports, not an
identity of the fitted distributions or economic actions.

The stop-loss diagnostic divides the absolute error in
$H(\alpha D^*_{{F,\mathrm{{con}}}})$ by the oracle value at the same
tariff-specific threshold. These nine oracle denominators range from
4,425.687 to 14,122.007 kW in the primary design; all are positive.
The complete cellwise values and the paired agreement audit are in
the accompanying source-data package.
"""
    addendum += "\n" + fidelity_table.removeprefix("\\clearpage\n") + "\n"
    si_source = replace_exact(si_source, "\\begin{thebibliography}{9}", addendum + "\n\\begin{thebibliography}{9}")
    (si_out / "Supporting_Information.tex").write_text(si_source, encoding="utf-8")
    figures_tex = si_out / "FIGURES.tex"
    figures_text = figures_tex.read_text(encoding="utf-8")
    figures_text = replace_exact(
        figures_text,
        "Numerical distribution-to-decision bounds and\nfitting-support sensitivity.",
        "Numerical distribution-to-decision bounds.")
    old_s1_caption_start = figures_text.index("\\caption{High-contract-quantile stress experiment")
    old_s1_caption_end = figures_text.index("\n\\label{fig:final-figures1}", old_s1_caption_start)
    figures_text = (figures_text[:old_s1_caption_start]
                    + "\\caption{High-contract-quantile stress using the 10,000 revised primary fitted laws and 1,500,000 tariff decisions. Columns show $p^*=0.90$, $0.95$, and $0.99$. (a--c) Mean full-action regret across the ten utilization cells for each method. (d--f) Exact-zero-regret percentages by method and cell. (g--i) History-paired, equal-cell-weighted regional mean differences for GEV minus KDE and TAIL minus EVENT-EMP, with pointwise 95\\% paired $t$ intervals. Negative values favour the first method. The three oracle modes are actual demand, contract demand, and capacity.}"
                    + figures_text[old_s1_caption_end:])
    support_caption_start = figures_text.index("\\textbf{(d,e)} Paired mean regret changes")
    support_caption_end = figures_text.index("\n}", support_caption_start)
    figures_text = figures_text[:support_caption_start].rstrip() + figures_text[support_caption_end:]
    figures_tex.write_text(figures_text, encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised" / "build_cssc_w1_figure_s3.py"),
        "--bounds", str(ROOT / "CSSC_REVIEW_REVISION_20261112" / "primary_cv_crps_primitive_bounds_20261115"),
        "--source-data", str(ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data"),
        "--supplement", str(si_out),
    ], check=True)
    published_w1 = (ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "data"
                    / "derived_publication" / "tariff_contrast_wasserstein")
    current_w1_files = {"audit_summary.json", "competitor_pair_summary.csv",
                        "numerical_condition_coverage.csv", "FigureS3_W1_input.csv",
                        "FigureS3_primitive_cell_input.csv", "FigureS3_coverage_input.csv",
                        "FigureS3.pdf", "FigureS3.png", "FigureS3.svg"}
    if not all((published_w1 / name).is_file() for name in current_w1_files):
        raise ValueError("Revised Figure S3 source files are incomplete")
    for old_w1_file in published_w1.iterdir():
        if old_w1_file.is_file() and old_w1_file.name not in current_w1_files:
            old_w1_file.unlink()
    stale_support_plot = (ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "data"
                          / "derived_publication" / "tariff_contrast_wasserstein"
                          / "FigureS3_support_input.csv")
    if stale_support_plot.is_file():
        stale_support_plot.unlink()
    bounds_run = ROOT / "CSSC_REVIEW_REVISION_20261112" / "primary_cv_crps_primitive_bounds_20261115"
    published_bounds = ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "data" / "source" / "r5_primitive_bounds"
    published_bounds.mkdir(parents=True, exist_ok=True)
    for old_bound_file in published_bounds.iterdir():
        if old_bound_file.is_file():
            old_bound_file.unlink()
    for name in ("primitive_bounds_manifest.json", "primitive_bounds_summary.json",
                 "w1_summary_by_method.csv", "primitive_summary_by_method_cell_oracle.csv",
                 "validation_summary_by_chunk.csv", "README.md"):
        shutil.copy2(bounds_run / name, published_bounds / name)
    published_numerical = ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "data" / "numerical_audit"
    published_numerical.mkdir(parents=True, exist_ok=True)
    for old_numerical_file in published_numerical.glob("*.parquet"):
        old_numerical_file.unlink()
    numerical_files = list(bounds_run.glob("primitive_decisions_*.parquet")) + list(bounds_run.glob("primitive_distances_*.parquet"))
    if len(numerical_files) != 800:
        raise ValueError("Revised numerical-bound parquet archive is incomplete")
    for source_file in numerical_files:
        shutil.copy2(source_file, published_numerical / source_file.name)
    (si_out / "README.txt").write_text(
        "Supporting Information source package\n\n"
        "This package contains the current supplementary LaTeX source, "
        "Figures S1-S8, Tables S1-S16, and files needed to compile the document. "
        "Tables S14-S16 are embedded in the supplementary source.\n\n"
        "Build from this directory with:\n"
        "  latexmk -pdf -interaction=nonstopmode -halt-on-error Supporting_Information.tex\n\n"
        "The main manuscript is maintained separately.\n",
        encoding="utf-8",
    )
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised" / "refresh_high_pstar_outputs.py"),
    ], check=True)
    high_run = ROOT / "CSSC_REVIEW_REVISION_20261112" / "high_pstar_cv_crps_20261115"
    published_high = ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "data" / "source" / "high_pstar"
    published_high.mkdir(parents=True, exist_ok=True)
    for old_high_file in published_high.iterdir():
        if old_high_file.is_file():
            old_high_file.unlink()
    for name in ("high_pstar_run_manifest.json", "high_pstar_checks.json",
                 "high_pstar_cell_summary_10000.csv", "high_pstar_mode_confusion_10000.csv",
                 "high_pstar_oracle_region_summary_10000.csv", "high_pstar_oracle_design_cells.csv",
                 "high_pstar_same_information_paired_contrasts.csv",
                 "figure_s1_table_s1_validation.json"):
        shutil.copy2(high_run / name, published_high / name)
    high_checkpoints = sorted((high_run / "checkpoints").glob("chunk_*_regrets.npz"))
    if len(high_checkpoints) != 400:
        raise ValueError("Revised high-p* history-level regret checkpoints are incomplete")
    published_checkpoints = published_high / "regret_checkpoints"
    published_checkpoints.mkdir(exist_ok=True)
    for old_file in published_checkpoints.iterdir():
        if old_file.is_file():
            old_file.unlink()
    for source_file in high_checkpoints:
        shutil.copy2(source_file, published_checkpoints / source_file.name)
    from apply_cssc_review_round import main as apply_review_revisions
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised" / "summarize_primary_fidelity.py"),
        "--decisions", str(ROOT / "CSSC_REVIEW_REVISION_20261112"
                            / "primary_cv_crps_analysis_20261115" / "analysis"
                            / "primary_decisions_enriched_10000.csv.gz"),
        "--out", str(ROOT / "CSSC_REVIEW_REVISION_20261112"
                      / "primary_cv_crps_analysis_20261115" / "analysis"
                      / "primary_functional_fidelity.csv"),
    ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised"
                            / "analyze_primary_functional_associations.py"),
        "--data", str(ROOT / "CSSC_REVIEW_REVISION_20261112"
                       / "primary_cv_crps_analysis_20261115" / "analysis"
                       / "primary_decisions_enriched_10000.csv.gz"),
        "--out", str(ROOT / "CSSC_REVIEW_REVISION_20261112"
                      / "primary_cv_crps_analysis_20261115" / "associations"),
        "--expected-n", "10000", "--bootstrap-replicates", "2000",
    ], check=True)
    apply_review_revisions()
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised" / "refresh_nb_weibull_tables.py"),
        "--run", str(ROOT / "CSSC_REVIEW_REVISION_20261112" / "nb_weibull_cv_crps_20261115"),
        "--supplement", str(si_out),
        "--report", str(ROOT / "CSSC_REVIEW_REVISION_20261112"
                        / "nb_weibull_cv_crps_20261115" / "tables_refresh_audit.json"),
    ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised" / "build_cssc_nb_figure_s7.py"),
        "--run", str(ROOT / "CSSC_REVIEW_REVISION_20261112" / "nb_weibull_cv_crps_20261115"),
        "--supplement", str(si_out),
        "--report", str(ROOT / "CSSC_REVIEW_REVISION_20261112"
                        / "nb_weibull_cv_crps_20261115" / "figure_s7_audit.json"),
    ], check=True)
    nb_run = ROOT / "CSSC_REVIEW_REVISION_20261112" / "nb_weibull_cv_crps_20261115"
    published_nb = ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "data" / "source" / "nonpoisson_gpd_challenge"
    published_nb.mkdir(parents=True, exist_ok=True)
    for old_nb_file in published_nb.iterdir():
        if old_nb_file.is_file():
            old_nb_file.unlink()
    for source_file in [nb_run / "run_manifest.json", *(nb_run / "summary").glob("*.csv"),
                        nb_run / "summary" / "validation.json"]:
        shutil.copy2(source_file, published_nb / source_file.name)
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised" / "refresh_primary_tables.py"),
        "--analysis", str(ROOT / "CSSC_REVIEW_REVISION_20261112"
                          / "primary_cv_crps_analysis_20261115" / "analysis"),
        "--source-data", str(ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data"),
        "--supplement", str(si_out),
        "--manuscript", str(OUT / "Manuscript.tex"),
        "--associations", str(ROOT / "CSSC_REVIEW_REVISION_20261112"
                               / "primary_cv_crps_analysis_20261115" / "associations"),
        "--report", str(ROOT / "CSSC_REVIEW_REVISION_20261112"
                        / "primary_cv_crps_analysis_20261115" / "refreshed_tables_manifest.json"),
    ], check=True)
    crossed = ROOT / "CSSC_REVIEW_REVISION_20261112" / "crossed_grid_cv_crps_20261115"
    published_crossed = ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "data" / "source" / "crossed_grid_no_upper_1000"
    manifest = json.loads((crossed / "raw" / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("rows") != 380000:
        raise ValueError("Revised crossed-grid experiment has not passed")
    published_crossed.mkdir(parents=True, exist_ok=True)
    for old_crossed_file in published_crossed.iterdir():
        if old_crossed_file.is_file():
            old_crossed_file.unlink()
    for source_file in (
        crossed / "raw" / "run_manifest.json",
        crossed / "raw" / "crossed_grid_no_upper_decisions.csv.gz",
        crossed / "raw" / "crossed_grid_no_upper_cell_summary.csv",
        crossed / "design" / "crossed_tariff_design.csv",
    ):
        shutil.copy2(source_file, published_crossed / source_file.name)
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised" / "build_cssc_primary_figures.py"),
    ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised" / "build_cssc_figure2_crossed.py"),
    ], check=True)
    validation_dir = ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for stale_validation in validation_dir.iterdir():
        if stale_validation.is_file():
            stale_validation.unlink()
    shutil.copy2(ROOT / "CSSC_REVIEW_REVISION_20261112" / "primary_cv_crps_analysis_20261115"
                 / "analysis" / "primary_analysis_checks.json",
                 validation_dir / "primary_analysis_checks.json")
    shutil.copy2(ROOT / "CSSC_REVIEW_REVISION_20261112" / "primary_cv_crps_primitive_bounds_20261115"
                 / "primitive_bounds_summary.json",
                 validation_dir / "primitive_bounds_summary.json")
    stale_plot = (ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data" / "data"
                  / "plot" / "FigureS_new_fidelity_maxT.csv")
    if stale_plot.is_file():
        stale_plot.unlink()
    for stale_table in (si_out / "tables").glob("*_numeric.csv"):
        stale_table.unlink()
    for stale_table in (si_out / "tables").glob("*_display.csv"):
        stale_table.unlink()
    stale_support_table = si_out / "tables" / "Table_S11.tex"
    if stale_support_table.is_file():
        stale_support_table.unlink()
    # Preserve the saved external charge-analysis records used by Figure 4(e,f).
    # These are held-out hypothetical-charge evaluations, not observed bills.
    external_source = (ROOT.parent / "！初稿V1.0" / "修改V1.2" / "修改代码+数据"
                       / "data" / "final_requirements_audit_20261002")
    external_target = (ROOT / "CSSC_REVIEW_REVISION_20261112" / "source_data"
                       / "data" / "source" / "dgp_external_support")
    external_target.mkdir(parents=True, exist_ok=True)
    for name in (
        "external_realized_factory_values.csv",
        "external_realized_factory_paired_differences.csv",
        "external_realized_method_cell_summary.csv",
        "external_realized_mode_counts.csv",
        "external_realized_paired_t_exact_signflip.csv",
        "external_representative_tariff_design.csv",
        "external_representative_tariff_realized_rows.csv",
    ):
        source_file = external_source / name
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing external charge record: {source_file}")
        shutil.copy2(source_file, external_target / name)
    figure4_source = (ROOT.parent / "！初稿V1.0" / "修改V1.2"
                      / "ASMBI_Main_SI_Figures_Tables_20261006_R5"
                      / "ASMBI_FINAL_FIGURES_TABLES_20261006_R5" / "data"
                      / "plot" / "Figure4_external_selected.csv")
    shutil.copy2(figure4_source, ROOT / "CSSC_REVIEW_REVISION_20261112"
                 / "source_data" / "data" / "plot" / figure4_source.name)
    subprocess.run([
        sys.executable, str(ROOT / "exp1" / "code" / "revised" / "refresh_cssc_addenda.py"),
    ], check=True)
    print(OUT / "Manuscript.tex")


if __name__ == "__main__":
    main()
