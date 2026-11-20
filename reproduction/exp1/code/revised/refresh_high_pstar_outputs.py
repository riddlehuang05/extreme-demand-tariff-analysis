#!/usr/bin/env python3
"""Rebuild high-p* SI outputs from the verified revised primary-law stress run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "CSSC_REVIEW_REVISION_20261112" / "high_pstar_cv_crps_20261115"
SI = ROOT / "CSSC_SUBMISSION_20261112" / "Supporting_Information"
METHODS = ("TAIL", "EVENT-EMP", "GEV", "KDE", "EMP")
P_STARS = (0.90, 0.95, 0.99)
REGIONS = ("actual_maximum_demand", "contracted_maximum_demand", "capacity")
REGION_NAMES = ("Actual", "Contract", "Capacity")
PAIRS = (("GEV", "KDE"), ("TAIL", "EVENT-EMP"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def signed(x: float) -> str:
    return f"{x:+.2f}"


def main() -> None:
    manifest = json.loads((RUN / "high_pstar_run_manifest.json").read_text(encoding="utf-8"))
    checks = json.loads((RUN / "high_pstar_checks.json").read_text(encoding="utf-8"))
    records = sorted(manifest.get("chunk_records", []), key=lambda r: r["start"])
    if (manifest.get("status") != "PASS" or checks.get("status") != "PASS"
            or len(records) != 400 or checks.get("decision_rows_reanalyzed") != 1_500_000):
        raise ValueError("Full revised high-p* stress has not passed")
    cubes = []
    for i, row in enumerate(records):
        start, end = 25 * i, 25 * (i + 1)
        if row["start"] != start or row["end"] != end or row["status"] != "PASS":
            raise ValueError("High-p* history chunks do not cover 0:10000 exactly")
        checkpoint = RUN / "checkpoints" / f"chunk_{start:05d}_{end:05d}.json"
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        regret_file = checkpoint.with_name(payload["regrets_file"])
        if (sha256(checkpoint) != row["output_sha256"]
                or sha256(regret_file) != row["regrets_sha256"]
                or payload["regrets_sha256"] != row["regrets_sha256"]):
            raise ValueError(f"High-p* checkpoint hash mismatch: {start}")
        with np.load(regret_file) as npz:
            ids = npz["replication_id"]
            cube = npz["regret_cny"]
        if cube.shape != (25, 5, 3, 10) or not np.array_equal(ids, np.arange(start, end)):
            raise ValueError(f"High-p* checkpoint shape/id mismatch: {start}")
        cubes.append(cube)
    regrets = np.concatenate(cubes, axis=0)
    if regrets.shape != (10000, 5, 3, 10) or not np.isfinite(regrets).all() or regrets.min() < -1e-5:
        raise ValueError("High-p* regret array incomplete")
    cell = pd.read_csv(RUN / "high_pstar_cell_summary_10000.csv")
    oracle = pd.read_csv(RUN / "high_pstar_oracle_design_cells.csv")
    if len(cell) != 150 or len(oracle) != 30:
        raise ValueError("High-p* summary/design row count incorrect")
    u = sorted(cell.utilization.unique())
    if len(u) != 10:
        raise ValueError("High-p* utilization grid incorrect")
    for mi, method in enumerate(METHODS):
        for pi, p in enumerate(P_STARS):
            for ui, util in enumerate(u):
                row = cell[(cell.method_id == method) & np.isclose(cell.p_star, p)
                           & np.isclose(cell.utilization, util)]
                if len(row) != 1 or abs(regrets[:, mi, pi, ui].mean() - row.iloc[0].mean_regret_cny) > 1e-5:
                    raise ValueError(f"High-p* cell mean mismatch: {method}/{p}/{util}")
    masks = {}
    for pi, p in enumerate(P_STARS):
        for region in REGIONS:
            cells = oracle[np.isclose(oracle.p_star, p) & (oracle.oracle_mode == region)]
            indices = [u.index(float(v)) for v in cells.utilization]
            if not indices:
                raise ValueError(f"Missing oracle region: {p}/{region}")
            masks[(pi, region)] = indices
    rows = []
    for pi, p in enumerate(P_STARS):
        for a, b in PAIRS:
            ai, bi = METHODS.index(a), METHODS.index(b)
            for region, label in zip(REGIONS, REGION_NAMES):
                indices = masks[(pi, region)]
                paired = (regrets[:, ai, pi, :][:, indices].mean(axis=1)
                          - regrets[:, bi, pi, :][:, indices].mean(axis=1))
                if paired.shape != (10000,):
                    raise ValueError("High-p* paired regional comparison lost history alignment")
                mean = float(paired.mean())
                half = float(t.ppf(.975, 9999) * paired.std(ddof=1) / np.sqrt(10000))
                rows.append({"p_star": p, "method_a": a, "method_b": b,
                             "oracle_region": region, "region_label": label,
                             "design_cells_equal_weighted": len(indices), "paired_histories": 10000,
                             "paired_mean_regret_difference_cny": mean,
                             "paired_ci_low_cny": mean - half, "paired_ci_high_cny": mean + half})
    contrasts = pd.DataFrame(rows)
    contrast_path = RUN / "high_pstar_same_information_paired_contrasts.csv"
    contrasts.to_csv(contrast_path, index=False, float_format="%.12g")

    lines = [r"\begingroup", r"\fontsize{8.5}{10.5}\selectfont\setlength{\tabcolsep}{3.0pt}\renewcommand{\arraystretch}{1.23}",
             r"\begin{table}[tbp]\centering", r"\caption{Same-information comparisons under high contract quantiles.}\label{tab:si-high-pstar}",
             r"\begin{tabular}{@{}llllll@{}}\toprule", r"$p^*$ & First -- second & Oracle & Cells & Difference & 95\% CI\\\midrule"]
    for row in rows:
        lines.append(f"{row['p_star']:.2f} & {row['method_a']} -- {row['method_b']} & {row['region_label']} & "
                     f"{row['design_cells_equal_weighted']} & {signed(row['paired_mean_regret_difference_cny']/1000)} & "
                     f"[{signed(row['paired_ci_low_cny']/1000)}, {signed(row['paired_ci_high_cny']/1000)}]\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\noindent\begin{minipage}{\linewidth}\footnotesize Loss differences are in $10^3$ CNY/month. Each comparison uses the same 10,000 fitted primary histories and equal weights over the indicated region cells; intervals are pointwise paired $t$ intervals. Alpha is 2.20.\end{minipage}",
              r"\end{table}", r"\endgroup"]
    table_path = SI / "tables" / "Table_S1.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.0), constrained_layout=True)
    grays = ("#111111", "#555555", "#999999", "#333333", "#777777")
    x = np.arange(10)
    for pi, p in enumerate(P_STARS):
        ax = axes[0, pi]
        for mi, method in enumerate(METHODS):
            values = regrets[:, mi, pi, :]
            mean = values.mean(axis=0) / 1000
            ax.plot(x, mean, marker="o", markersize=2.3, linewidth=1.0,
                    color=grays[mi], linestyle=("-", "--", ":", "-.", (0, (4, 2)))[mi], label=method)
        ax.set_title(f"$p^*={p:.2f}$")
        ax.set_ylabel("Mean regret ($10^3$ CNY/month)" if pi == 0 else "")
        ax.set_xticks(x, [f"{v:.3g}" for v in u], rotation=45)
        if pi == 0:
            ax.legend(ncol=2, fontsize=7, frameon=False)
        ax = axes[1, pi]
        zero = np.mean(np.abs(regrets[:, :, pi, :]) <= 1e-8, axis=0) * 100
        image = ax.imshow(zero, vmin=0, vmax=100, cmap="Greys", aspect="auto")
        ax.set_yticks(np.arange(5), METHODS if pi == 0 else [""] * 5)
        ax.set_xticks(x, [f"{v:.3g}" for v in u], rotation=45)
        ax.set_xlabel("Utilization")
        if pi == 2:
            fig.colorbar(image, ax=ax, label="Exact-zero regret (%)", shrink=.8)
        ax = axes[2, pi]
        for pair_i, (a, b) in enumerate(PAIRS):
            sub = contrasts[(np.isclose(contrasts.p_star, p))
                            & (contrasts.method_a == a) & (contrasts.method_b == b)]
            sub = sub.set_index("oracle_region").loc[list(REGIONS)]
            y = sub.paired_mean_regret_difference_cny.to_numpy() / 1000
            lo = sub.paired_ci_low_cny.to_numpy() / 1000
            hi = sub.paired_ci_high_cny.to_numpy() / 1000
            ax.errorbar(np.arange(3) + (pair_i - .5) * .12, y,
                        yerr=np.vstack((y - lo, hi - y)), fmt="o" if pair_i == 0 else "s",
                        markersize=4, capsize=2, color=grays[pair_i],
                        label=f"{a} -- {b}")
        ax.axhline(0, color="black", linewidth=.6)
        ax.set_xticks(range(3), REGION_NAMES)
        ax.set_ylabel("Paired difference ($10^3$ CNY/month)" if pi == 0 else "")
        if pi == 0:
            ax.legend(fontsize=7, frameon=False)
    for label, ax in zip("abcdefghi", axes.flat):
        ax.text(-.12, 1.04, f"({label})", transform=ax.transAxes, weight="bold", va="bottom")
    figure_dir = SI / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(figure_dir / f"FigureS1.{suffix}", dpi=240)
    plt.close(fig)
    audit = {"status": "PASS", "histories": 10000, "methods": len(METHODS),
             "p_star_levels": list(P_STARS), "decisions": int(regrets.size),
             "regret_min_cny": float(regrets.min()),
             "regret_max_cny": float(regrets.max()),
             "source_manifest_sha256": sha256(RUN / "high_pstar_run_manifest.json"),
             "source_cell_summary_sha256": sha256(RUN / "high_pstar_cell_summary_10000.csv"),
             "outputs": {str(p): sha256(p) for p in (contrast_path, table_path, figure_dir / "FigureS1.pdf")}}
    (RUN / "figure_s1_table_s1_validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
