"""Draw Figure S7 from the revised 10,000-history NB--Weibull experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

METHODS = ("TAIL", "EVENT-EMP", "GEV", "KDE", "EMP")
REGIONS = ("actual", "contract", "capacity")
COLORS = dict(zip(METHODS, ("#176B87", "#985D88", "#D17828", "#39836C", "#666666")))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "PASS" or manifest.get("replications_requested") != 10000
            or manifest.get("summary_validation", {}).get("decision_rows") != 450000):
        raise ValueError("Revised NB--Weibull full run is incomplete")
    paths = sorted((run / "chunks/decisions").glob("part_*.parquet"))
    if len(paths) != 400:
        raise ValueError("Expected 400 revised NB--Weibull decision chunks")
    frame = pd.concat((pd.read_parquet(path, columns=["replication_id", "cell_id", "method_id",
                                                    "oracle_region", "regret_cny_per_month"])
                       for path in paths), ignore_index=True)
    if (len(frame) != 450000 or frame.replication_id.nunique() != 10000
            or frame.duplicated(["replication_id", "method_id", "cell_id"]).any()):
        raise ValueError("Revised NB--Weibull decision keys are incomplete")
    design = pd.read_csv(run / "summary/oracle_tariff_design.csv")
    if len(design) != 9 or set(frame.cell_id) != set(design.cell_id):
        raise ValueError("NB--Weibull tariff design does not match decisions")
    design["alpha"] = design.policy_id.map({"BASE_ALPHA105": 1.05, "CONTRACT_ALPHA120": 1.20})
    if design.alpha.isna().any():
        raise ValueError("Unrecognized NB--Weibull tariff policy")
    grouped = frame.groupby(["cell_id", "method_id"], sort=True).regret_cny_per_month
    summary = grouped.agg(n="size", mean="mean", sd="std", p10=lambda x: x.quantile(.10),
                          p25=lambda x: x.quantile(.25), median="median",
                          p75=lambda x: x.quantile(.75), p90=lambda x: x.quantile(.90)).reset_index()
    if len(summary) != 45 or not summary.n.eq(10000).all():
        raise ValueError("Incomplete NB--Weibull cell summary")
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["sd"] / np.sqrt(10000)
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["sd"] / np.sqrt(10000)
    summary = summary.merge(design[["cell_id", "oracle_region", "alpha", "utilization"]],
                            on="cell_id", validate="many_to_one")
    summary.to_csv(run / "summary/FigureS7_cell_distribution.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
                         "axes.titlesize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.25), sharey=True,
                             gridspec_kw={"width_ratios": [2, 5, 2]})
    fig.subplots_adjust(left=.08, right=.985, bottom=.29, top=.84, wspace=.16)
    offsets = np.linspace(-.30, .30, len(METHODS))
    for ax, region, letter in zip(axes, REGIONS, "abc"):
        selected = design[design.oracle_region.eq(region)].sort_values(["alpha", "utilization"])
        for j, cell in enumerate(selected.itertuples(index=False)):
            subset = summary[summary.cell_id.eq(cell.cell_id)].set_index("method_id")
            for method, offset in zip(METHODS, offsets):
                row = subset.loc[method]
                x = j + offset
                ax.vlines(x, row.p10/1000, row.p90/1000, color=COLORS[method], linewidth=.7)
                ax.add_patch(Rectangle((x-.055, row.p25/1000), .11,
                                           max((row.p75-row.p25)/1000, 1e-5),
                                           facecolor=COLORS[method], edgecolor="none", alpha=.75))
                ax.plot([x-.06, x+.06], [row["median"]/1000]*2, color="black", linewidth=.65)
                ax.errorbar(x, row["mean"]/1000,
                            yerr=[[max(0, (row["mean"]-row.ci95_low)/1000)],
                                  [max(0, (row.ci95_high-row["mean"])/1000)]],
                            fmt="o", markersize=2, markerfacecolor="white",
                            markeredgecolor=COLORS[method], ecolor="black", elinewidth=.55,
                            capsize=1.2)
        ax.set_xticks(range(len(selected)),
                      [f"{r.alpha:.2f}/{r.utilization:.3f}" for r in selected.itertuples(index=False)],
                      rotation=60, ha="right")
        ax.set_title(f"({letter})  {region.title()} region", loc="left")
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(axis="y", color="#dddddd", linewidth=.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Regret ($10^3$ CNY/month)")
    handles = [Line2D([], [], color=COLORS[method], marker="s", linestyle="none",
                          markersize=5, label=method) for method in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .985),
               ncol=5, frameon=False, columnspacing=1.1)
    target = args.supplement / "figures/FigureS7.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    fig.savefig(target.with_suffix(".svg"))
    fig.savefig(target.with_suffix(".png"), dpi=500)
    plt.close(fig)
    report = {"status": "PASS", "rows": len(frame), "histories": 10000,
              "run_manifest_sha256": sha256(manifest_path),
              "cell_summary_sha256": sha256(run / "summary/FigureS7_cell_distribution.csv"),
              "figure_pdf_sha256": sha256(target)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
