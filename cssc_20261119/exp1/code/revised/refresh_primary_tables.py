"""Refresh primary result tables from the corrected 10,000-history analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


def signed(number: float) -> str:
    return f"{number:+.2f}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    old_dir = args.source_data / "data" / "source" / "primary"
    names = [
        "method_cell_precision_10000.csv",
        "oracle_mode_values_9_cells.csv",
        "paired_regret_contrasts_10000.csv",
        "regret_contributions_by_method_oracle_region_10000.csv",
        "regret_tail_summary_by_method_cell_10000.csv",
        "signed_functional_shifts_and_crossings_10000.csv",
        "primary_analysis_checks.json",
    ]
    for name in names:
        source = args.analysis / name
        target = old_dir / name
        if not source.is_file():
            raise FileNotFoundError(name)
        target.write_bytes(source.read_bytes())
    enriched_name = "primary_decisions_enriched_10000.csv.gz"
    enriched_source = args.analysis / enriched_name
    if not enriched_source.is_file():
        raise FileNotFoundError(enriched_source)
    (args.source_data / "data" / "source" / enriched_name).write_bytes(enriched_source.read_bytes())
    association_target = args.source_data / "data" / "source" / "teacher_sonnet"
    association_target.mkdir(parents=True, exist_ok=True)
    association_files = list(args.associations.glob("*.csv")) + list(args.associations.glob("*.json"))
    if len(association_files) < 9:
        raise ValueError("revised functional-association outputs are incomplete")
    for source in association_files:
        (association_target / source.name).write_bytes(source.read_bytes())
    method_cell = pd.read_csv(args.analysis / "method_cell_precision_10000.csv")
    main = args.manuscript.read_text(encoding="utf-8")
    start = main.index("\\midrule\nActual-demand & TAIL &", main.index("\\label{tab:table_3}"))
    end = main.index("\\bottomrule", start)
    region_order = [
        ("actual_maximum_demand", "Actual-demand"),
        ("contracted_maximum_demand", "Contract-demand"),
        ("capacity", "Capacity"),
    ]
    method_order = ["TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"]
    main_rows = []
    for mode, label in region_order:
        for method in method_order:
            cells = method_cell[(method_cell.oracle_mode == mode)
                                & (method_cell.method_id == method)]
            if len(cells) != 3 or not cells.histories.eq(10000).all():
                raise AssertionError(f"incomplete table 3 group: {mode}/{method}")
            regret = cells.regret_cny_per_month_mean.mean() / 1000
            normalized = 100 * cells.regret_over_oracle_value_mean.mean()
            accuracy = 100 * cells.accuracy.mean()
            main_rows.append(f"{label} & {method} & {regret:.2f} & {normalized:.3f} & {accuracy:.1f}\\\\")
    main = main[:start] + "\\midrule\n" + "\n".join(main_rows) + "\n" + main[end:]
    args.manuscript.write_text(main, encoding="utf-8")
    contrasts = pd.read_csv(args.analysis / "paired_regret_contrasts_10000.csv")
    table = args.supplement / "tables" / "Table_S12.tex"
    contents = table.read_text(encoding="utf-8")
    regions = (
        ("actual_maximum_demand", "Actual"),
        ("contracted_maximum_demand", "Contract"),
        ("capacity", "Capacity"),
    )
    methods = [
        ("TAIL", "EVENT-EMP"), ("GEV", "KDE"), ("KDE", "EMP"),
        ("TAIL", "KDE"), ("TAIL", "GEV"), ("TAIL", "EMP"),
        ("GEV", "EMP"), ("EVENT-EMP", "GEV"),
        ("EVENT-EMP", "KDE"), ("EVENT-EMP", "EMP"),
    ]
    updated = []
    for first, second in methods:
        cells = []
        for mode, _ in regions:
            data = contrasts[(contrasts.scope == "oracle_region")
                             & (contrasts.oracle_mode == mode)
                             & (contrasts.method_a == min(first, second))
                             & (contrasts.method_b == max(first, second))]
            if len(data) != 1:
                raise AssertionError(f"missing contrast: {first}, {second}, {mode}")
            row = data.iloc[0]
            scale = 1 if first < second else -1
            mean = scale * row.paired_regret_difference_a_minus_b / 1000
            low = (row.t95_low if scale == 1 else -row.t95_high) / 1000
            high = (row.t95_high if scale == 1 else -row.t95_low) / 1000
            cells.append(r"\shortstack{" + signed(mean) + r"\\{}[" + signed(low) + ", " + signed(high) + "]}")
        old = next((line for line in contents.splitlines() if line.startswith(f"{first} -- {second} &")), None)
        if old is None:
            raise AssertionError(f"missing table row {first}--{second}")
        new = f"{first} -- {second} & " + " & ".join(cells) + r"\\"
        contents = contents.replace(old, new)
        updated.append({"contrast": f"{first}--{second}", "new_row": new})
    table.write_text(contents, encoding="utf-8")
    correlation = pd.read_csv(args.associations / "original_10_correlation_family_maxT.csv")
    if len(correlation) != 10 or not correlation.histories.eq(10000).all():
        raise ValueError("ten revised correlation contrasts are incomplete")
    s7 = args.supplement / "tables" / "Table_S7.tex"
    s7_text = s7.read_text(encoding="utf-8")
    for method in ("TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"):
        for key, label in (("abs_mean_error_minus_abs_q99_error", "Mean -- q99"),
                           ("abs_stoploss_error_minus_abs_q99_error", "Stop-loss -- q99")):
            match = correlation[(correlation.method_id == method) & (correlation.contrast == key)]
            if len(match) != 1:
                raise ValueError(f"missing revised correlation {method}/{key}")
            row = match.iloc[0]
            old = next((line for line in s7_text.splitlines()
                        if line.startswith(f"{method} & {label} &")), None)
            if old is None:
                raise ValueError(f"missing Table S7 row {method}/{label}")
            new = (f"{method} & {label} & {row.estimate:+.3f} & "
                   f"[{row.simultaneous_maxT95_low:+.3f}, {row.simultaneous_maxT95_high:+.3f}]\\\\")
            s7_text = s7_text.replace(old, new, 1)
    s7.write_text(s7_text, encoding="utf-8")
    certificate_sources = (
        ("d_mv_lt_1", r"$d_{MV}$", "teacher_fig3d_dmv_certificate_pooled.csv"),
        ("pairwise_gap", "Pairwise-gap", "teacher_fig3d_pairwise_certificate_pooled.csv"),
    )
    s2_lines = [
        r"\begingroup",
        r"\fontsize{8.5}{10.5}\selectfont\setlength{\tabcolsep}{3.0pt}\renewcommand{\arraystretch}{1.23}",
        r"\begin{table}[tbp]\centering",
        r"\caption{Within-contract regret by mode-stability condition.}\label{tab:si-stability-condition}",
        r"\begin{tabular}{@{}lllllll@{}}\toprule",
        r"Condition & Method & Satisfied & Decisions & Histories & Mean & 95\% CI\\\midrule",
    ]
    certificate_hashes = {}
    for cert_type, label, filename in certificate_sources:
        source = args.associations / filename
        pooled = pd.read_csv(source)
        certificate_hashes[filename] = file_sha256(source)
        if (len(pooled) != 10 or set(pooled.method_id) != set(method_order)
                or set(pooled.certificate_status) != {"certified", "uncertified"}
                or not pooled.certificate_type.eq(cert_type).all()
                or pooled.decision_rows.le(0).any()
                or pooled.independent_histories.le(0).any()
                or pooled.mode_selection_regret_cny_mean.abs().max() > 1e-5
                or (pooled.regret_cny_per_month_mean
                    - pooled.within_contract_regret_cny_mean).abs().max() > 1e-5):
            raise ValueError(f"Incomplete revised stability-condition summary: {filename}")
        for method in method_order:
            for status, satisfied in (("certified", "Yes"), ("uncertified", "No")):
                match = pooled[(pooled.method_id == method)
                               & (pooled.certificate_status == status)]
                if len(match) != 1:
                    raise ValueError(f"Missing revised stability row: {cert_type}/{method}/{status}")
                row = match.iloc[0]
                s2_lines.append(
                    f"{label} & {method} & {satisfied} & {int(row.decision_rows):,} & "
                    f"{int(row.independent_histories):,} & {row.regret_cny_per_month_mean/1000:.2f} & "
                    f"[{row.t95_low/1000:.2f}, {row.t95_high/1000:.2f}]\\\\")
    s2_lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\par\smallskip\noindent\begin{minipage}{\linewidth}\footnotesize The subset contains decisions for which both oracle and fitted-law selected modes are contract. The mean is the equally weighted mean of eligible history means, with a pointwise $t$ interval using the displayed number of histories. Loss units are $10^3$ CNY/month. Different subsets overlap across methods and condition strata; their counts must not be added to obtain the number of independent histories.\end{minipage}",
        r"\end{table}", r"\endgroup",
    ]
    s2 = args.supplement / "tables" / "Table_S2.tex"
    s2.write_text("\n".join(s2_lines) + "\n", encoding="utf-8")
    report = {"status": "PASS", "source": str(args.analysis), "tables_refreshed": names + ["Manuscript Table 3", "Table S7"],
              "supplement_table": str(table), "contrast_rows": updated,
              "stability_source_sha256": certificate_hashes,
              "stability_table_sha256": file_sha256(s2)}
    report["tables_refreshed"].append("Table S2")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "data_tables": len(names),
                      "supplement_contrasts": len(updated)}, indent=2))


if __name__ == "__main__":
    main()
