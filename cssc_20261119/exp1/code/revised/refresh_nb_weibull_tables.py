"""Refresh CSSC NB--Weibull supplementary tables from the revised full run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signed(value: float) -> str:
    return f"{value:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = manifest.get("summary_validation", {})
    if (manifest.get("status") != "PASS" or manifest.get("replications_requested") != 10000
            or checks.get("status") != "PASS" or checks.get("decision_rows") != 450000
            or checks.get("unique_histories") != 10000):
        raise ValueError("The revised 10,000-history NB--Weibull run is incomplete")
    summary = run / "summary"
    cell_path = summary / "five_method_by_cell.csv"
    contrast_path = summary / "paired_method_contrasts_by_region.csv"
    design_path = summary / "oracle_tariff_design.csv"
    cells = pd.read_csv(cell_path)
    contrasts = pd.read_csv(contrast_path)
    design = pd.read_csv(design_path)
    methods = ("TAIL", "EVENT-EMP", "GEV", "KDE", "EMP")
    regions = ("actual", "contract", "capacity")
    if (len(cells) != 45 or len(contrasts) != 30 or len(design) != 9
            or set(cells.method_id) != set(methods) or set(cells.cell_id) != set(design.cell_id)
            or not cells.n_history_rows.eq(10000).all()
            or not contrasts.paired_history_observations_equal_cell_weighted.eq(10000).all()):
        raise ValueError("NB--Weibull summary tables have incomplete keys or history counts")
    policy_alpha = {"BASE_ALPHA105": 1.05, "CONTRACT_ALPHA120": 1.20}
    if set(design.policy_id) != set(policy_alpha):
        raise ValueError("Unexpected NB--Weibull tariff policy identifiers")
    design["alpha"] = design.policy_id.map(policy_alpha)
    merged = cells.merge(design[["cell_id", "alpha", "utilization"]], on="cell_id", validate="many_to_one")
    merged = merged.sort_values(["method_id", "alpha", "utilization"])
    rows4 = []
    for row in merged.itertuples(index=False):
        mean = row.regret_cny_per_month_mean / 1000
        low = row.regret_cny_per_month_ci95_low / 1000
        high = row.regret_cny_per_month_ci95_high / 1000
        if not pd.notna([mean, low, high]).all():
            raise ValueError("Nonfinite cellwise confidence interval")
        rows4.append(f"{row.oracle_region} & {row.alpha:.2f} / {row.utilization:.3f} & {row.method_id} & "
                     f"{mean:.2f} & [{low:.2f}, {high:.2f}] & {100 * row.action_accuracy:.1f}\\\\")
    table4 = r"""\begingroup
\fontsize{8.5}{10.5}\selectfont\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}llllll@{}}
\caption{Cell-specific NB--Weibull challenge results.}\label{tab:final-si-4}\\
\toprule
Region & Alpha / $u$ & Method & Mean regret & 95\% CI & Accuracy (\%)\\\midrule\endfirsthead
Region & Alpha / $u$ & Method & Mean regret & 95\% CI & Accuracy (\%)\\\midrule\endhead
""" + "\n".join(rows4) + r"""
\bottomrule
\end{longtable}
\endgroup
"""
    rows5 = []
    for region in regions:
        for row in contrasts[contrasts.oracle_region.eq(region)].itertuples(index=False):
            mean = row.paired_mean_regret_difference_a_minus_b_cny / 1000
            low = row.ci95_low_cny / 1000
            high = row.ci95_high_cny / 1000
            if not pd.notna([mean, low, high]).all():
                raise ValueError("Nonfinite regional paired interval")
            rows5.append(f"{region} & {row.method_a} -- {row.method_b} & {signed(mean)} & "
                         f"[{signed(low)}, {signed(high)}]\\\\")
    table5 = r"""\begingroup
\fontsize{8.5}{10.5}\selectfont\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}llll@{}}
\caption{Complete regional paired NB--Weibull contrasts.}\label{tab:final-si-5}\\
\toprule
Region & First -- second & Mean difference & 95\% CI\\\midrule\endfirsthead
Region & First -- second & Mean difference & 95\% CI\\\midrule\endhead
""" + "\n".join(rows5) + r"""
\bottomrule
\end{longtable}
\endgroup
"""
    table_dir = args.supplement / "tables"
    (table_dir / "Table_S4.tex").write_text(table4, encoding="utf-8")
    (table_dir / "Table_S5.tex").write_text(table5, encoding="utf-8")
    report = {"status": "PASS", "run_manifest_sha256": sha256(manifest_path),
              "source_sha256": {p.name: sha256(p) for p in (cell_path, contrast_path, design_path)},
              "table_sha256": {name: sha256(table_dir / name) for name in ("Table_S4.tex", "Table_S5.tex")},
              "cell_rows": len(rows4), "paired_contrast_rows": len(rows5)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
