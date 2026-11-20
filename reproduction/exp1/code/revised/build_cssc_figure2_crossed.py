"""Rebuild Figure 2 and Figure S8 from four-fold CV-CRPS results.

The design follows the effective V1.1 Figure 2 architecture. It updates the
primary panels to the final 10,000-history data and restores the four crossed-
grid panels from the final 1,000-history, 76-cell experiment. Figure S8 gives
the full five-method grid profiles that would be too dense for the main panel.
No models are refit and no Monte Carlo histories are generated here.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import t

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = WORKSPACE / "CSSC_REVIEW_REVISION_20261112" / "source_data"
MAIN_RAW = ROOT / "data/source/primary_decisions_enriched_10000.csv.gz"
CONTRAST_SOURCE = ROOT / "data/source/primary/paired_regret_contrasts_10000.csv"
CROSS = WORKSPACE / "CSSC_REVIEW_REVISION_20261112" / "crossed_grid_cv_crps_20261115"
CROSS_RAW = CROSS / "raw/crossed_grid_no_upper_decisions.csv.gz"
DESIGN_PATH = CROSS / "design/crossed_tariff_design.csv"
CELL_SUMMARY = CROSS / "raw/crossed_grid_no_upper_cell_summary.csv"
DERIVED = ROOT / "data/derived_plotting"
AUDIT = ROOT / "audit"

METHODS = ["TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"]
MODES = ["actual_maximum_demand", "contracted_maximum_demand", "capacity"]
MODE_LABELS = {
    "actual_maximum_demand": "Actual",
    "contracted_maximum_demand": "Contract",
    "capacity": "Capacity",
}
MODE_COLORS = {'actual_maximum_demand':'#527F96','contracted_maximum_demand':'#B48745','capacity':'#6D6D6D'}
MODE_MARKERS = {'actual_maximum_demand':'o','contracted_maximum_demand':'^','capacity':'s'}
COLORS = {'TAIL':'#176B87','EVENT-EMP':'#985D88','GEV':'#D17828','KDE':'#39836C','EMP':'#666666'}

MARKERS = {"TAIL": "o", "EVENT-EMP": "s", "GEV": "^", "KDE": "D", "EMP": "v"}
LINESTYLES = {"TAIL": "-", "EVENT-EMP": "--", "GEV": "-.", "KDE": ":", "EMP": (0, (5, 2))}

CONTRASTS = [
    ("TAIL", "EVENT-EMP", "#9B7A42", "--"),
    ("TAIL", "KDE", "#176B87", "-"),
    ("GEV", "KDE", "#98628A", "-.")
]
REGION_ORDER = [
    "actual_maximum_demand",
    "contracted_maximum_demand",
    "capacity",
]
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8.1,
    "axes.labelsize": 8.0,
    "axes.titlesize": 9.1,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.0,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "eps"):
        fig.savefig(stem.with_suffix(f".{ext}"), format=ext)
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    with Image.open(stem.with_suffix(".png")) as image:
        buffer = BytesIO()
        image.convert("RGB").save(
            buffer, format="TIFF", compression="tiff_lzw", dpi=(600, 600)
        )
    stem.with_suffix(".tiff").write_bytes(buffer.getvalue())


def light_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color="#dddddd", linewidth=0.45)
    ax.set_axisbelow(True)


def add_panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"({letter})  {title}", loc="left", pad=4.0)


def formal_t_interval(
    contrast_table: pd.DataFrame, method_a: str, method_b: str, mode: str
) -> tuple[float, float, float]:
    """Return formal mean and t95 endpoints in CNY, preserving contrast sign."""
    rows = contrast_table[
        contrast_table.scope.eq("oracle_region") & contrast_table.oracle_mode.eq(mode)
    ]
    direct = rows[rows.method_a.eq(method_a) & rows.method_b.eq(method_b)]
    if not direct.empty:
        row = direct.iloc[0]
        return (
            float(row.paired_regret_difference_a_minus_b),
            float(row.t95_low),
            float(row.t95_high),
        )
    reverse = rows[rows.method_a.eq(method_b) & rows.method_b.eq(method_a)]
    if reverse.empty:
        raise AssertionError(f"No formal paired interval: {method_a} - {method_b}, {mode}")
    row = reverse.iloc[0]
    return (
        -float(row.paired_regret_difference_a_minus_b),
        -float(row.t95_high),
        -float(row.t95_low),
    )


def build_primary_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    primary = pd.read_csv(
        MAIN_RAW,
        usecols=[
            "replication_id", "method_id", "cell_id", "support_policy",
            "oracle_mode", "q99_relative_error", "regret_cny_per_month",
        ],
    )
    assert len(primary) == 450_000
    assert primary.replication_id.nunique() == 10_000
    assert set(primary.method_id) == set(METHODS)
    assert primary.cell_id.nunique() == 9
    assert primary.support_policy.eq("no_upper").all()
    assert not primary.duplicated(["replication_id", "method_id", "cell_id"]).any()

    q99 = primary[["replication_id", "method_id", "q99_relative_error"]].drop_duplicates()
    assert len(q99) == 50_000
    repeat_spread = primary.groupby(["replication_id", "method_id"]).q99_relative_error.agg(
        lambda values: values.max() - values.min()
    )
    assert repeat_spread.max() == 0
    e_event = q99[q99.method_id.eq("EVENT-EMP")].sort_values("replication_id").q99_relative_error
    e_month = q99[q99.method_id.eq("EMP")].sort_values("replication_id").q99_relative_error
    assert np.array_equal(e_event.to_numpy(), e_month.to_numpy())

    regional = primary.groupby(
        ["replication_id", "oracle_mode", "method_id"], observed=True
    ).regret_cny_per_month.mean().unstack("method_id")
    contrast_table = pd.read_csv(CONTRAST_SOURCE)
    summary_rows = []
    ecdf_rows = []
    max_check_error = 0.0
    for mode in REGION_ORDER:
        region = regional.xs(mode, level="oracle_mode")
        for method_a, method_b, color, linestyle in CONTRASTS:
            values = (region[method_a] - region[method_b]).sort_values(kind="mergesort")
            assert len(values) == 10_000
            mean = float(values.mean())
            half_width = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            low, high = mean - half_width, mean + half_width
            formal_mean, formal_low, formal_high = formal_t_interval(
                contrast_table, method_a, method_b, mode
            )
            max_check_error = max(
                max_check_error,
                abs(mean - formal_mean), abs(low - formal_low), abs(high - formal_high),
            )
            summary_rows.append({
                "oracle_mode": mode,
                "contrast": f"{method_a} - {method_b}",
                "histories": len(values),
                "mean_cny": mean,
                "t95_low_cny": low,
                "t95_high_cny": high,
                "formal_source_mean_cny": formal_mean,
                "formal_source_t95_low_cny": formal_low,
                "formal_source_t95_high_cny": formal_high,
            })
            sorted_values = values.to_numpy()
            ecdf_rows.extend({
                "oracle_mode": mode,
                "contrast": f"{method_a} - {method_b}",
                "replication_rank": i,
                "paired_difference_cny": value,
                "paired_difference_1000_cny": value / 1000,
                "empirical_probability": i / len(sorted_values),
            } for i, value in enumerate(sorted_values, start=1))
    assert max_check_error < 1e-7, max_check_error
    return q99, pd.DataFrame(ecdf_rows), pd.DataFrame(summary_rows), {
        "primary_rows": len(primary),
        "primary_histories": int(primary.replication_id.nunique()),
        "q99_history_method_rows": len(q99),
        "event_emp_q99_exact_ties": 10_000,
        "max_regional_contrast_source_difference_cny": max_check_error,
        "regions": REGION_ORDER,
        "support_policy": "no_upper",
    }


def draw_figure2() -> dict:
    primary_checks = json.loads((ROOT / "data/source/primary/primary_analysis_checks.json").read_text(encoding="utf-8"))
    if (primary_checks.get("decision_rows") != 450000
            or primary_checks.get("unique_histories") != 10000
            or not primary_checks.get("all_outputs_created")):
        raise ValueError("The revised primary analysis is incomplete")
    manifest = json.loads((CROSS / "raw/run_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "PASS" or manifest.get("replications") != 1000
            or manifest.get("rows") != 380000 or manifest.get("cells") != 76
            or manifest.get("decisions_sha256") != sha256(CROSS_RAW)):
        raise ValueError("The revised crossed-grid run is incomplete or hash-mismatched")
    q99, paired_ecdf, paired_summary, qc = build_primary_inputs()
    cross = pd.read_csv(
        CROSS_RAW,
        usecols=[
            "replication_id", "method_id", "support_policy", "alpha", "cell_id",
            "utilization", "oracle_mode", "action_correct", "regret_cny_per_month",
            "regret_over_oracle_value",
        ],
    )
    design = pd.read_csv(DESIGN_PATH)
    formal_cells = pd.read_csv(CELL_SUMMARY)
    assert len(cross) == 380_000
    assert cross.replication_id.nunique() == 1_000
    assert cross.cell_id.nunique() == 76
    assert set(cross.method_id) == set(METHODS)
    assert cross.support_policy.eq("nonnegative_no_artificial_upper_truncation").all()
    assert not cross.duplicated(["replication_id", "method_id", "cell_id"]).any()
    assert len(design) == 76 and design.duplicated(["alpha", "utilization"]).sum() == 0
    assert len(np.unique(design.alpha)) == 4
    assert design.groupby("alpha").utilization.nunique().eq(19).all()
    assert cross.groupby(["cell_id", "method_id"]).replication_id.nunique().eq(1_000).all()

    computed = cross.groupby(["cell_id", "method_id"], observed=True).agg(
        mean_regret_cny=("regret_cny_per_month", "mean"),
        mode_accuracy=("action_correct", "mean"),
        mean_normalized_regret=("regret_over_oracle_value", "mean"),
    ).reset_index()
    verified = computed.merge(formal_cells, on=["cell_id", "method_id"], validate="one_to_one")
    checks = {
        "mean_regret_matches_formal_cell_summary": float(
            np.max(np.abs(verified.mean_regret_cny_x - verified.mean_regret_cny_y))
        ),
        "mode_accuracy_matches_formal_cell_summary": float(
            np.max(np.abs(verified.mode_accuracy - verified.action_accuracy))
        ),
        "normalized_mean_matches_formal_cell_summary": float(
            np.max(np.abs(verified.mean_normalized_regret - verified.mean_regret_over_oracle))
        ),
    }
    assert checks["mean_regret_matches_formal_cell_summary"] < 1e-7
    assert checks["mode_accuracy_matches_formal_cell_summary"] < 1e-12
    assert checks["normalized_mean_matches_formal_cell_summary"] < 1e-12

    cell_design = design[["cell_id", "alpha", "utilization", "oracle_mode"]]
    cell_effects = computed.merge(cell_design, on="cell_id", validate="many_to_one")
    mean_matrix = cell_effects.pivot(index="cell_id", columns="method_id", values="mean_regret_cny")
    diff_records = []
    for cell in cell_design.itertuples(index=False):
        for method_a, method_b, _, _ in CONTRASTS[1:]:
            diff_records.append({
                "cell_id": cell.cell_id,
                "alpha": cell.alpha,
                "utilization": cell.utilization,
                "oracle_mode": cell.oracle_mode,
                "contrast": f"{method_a} - {method_b}",
                "mean_difference_cny": float(mean_matrix.loc[cell.cell_id, method_a] - mean_matrix.loc[cell.cell_id, method_b]),
                "histories_per_method_cell": 1_000,
                "inference": "descriptive cell mean; no interval",
            })
    grid_effects = pd.DataFrame(diff_records).sort_values(["alpha", "utilization", "contrast"])
    assert len(grid_effects) == 152

    DERIVED.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/derived_plotting/Figure2_q99_ecdf.csv").write_text(
        q99.assign(absolute_q99_error_percent=q99.q99_relative_error.abs() * 100)
        .sort_values(["method_id", "absolute_q99_error_percent"])
        .assign(empirical_probability=lambda frame: frame.groupby("method_id").cumcount().add(1)
                / frame.groupby("method_id").replication_id.transform("count"))
        .to_csv(index=False)
    )
    paired_ecdf.to_csv(DERIVED / "Figure2_regional_paired_ecdf.csv", index=False)
    paired_summary.to_csv(DERIVED / "Figure2_regional_t95_summary.csv", index=False)
    grid_effects.to_csv(DERIVED / "Figure2_crossed_grid_selected_contrasts.csv", index=False)

    fig = plt.figure(figsize=(178 * MM, 210 * MM))
    outer = fig.add_gridspec(
        5, 2,
        left=0.105, right=0.985, top=0.972, bottom=0.042,
        height_ratios=[1.0, 1.0, 0.25, 0.98, 0.22],
        hspace=0.62, wspace=0.30,
    )

    # (a) tail-fidelity ECDFs: one observation per history and method.
    ax = fig.add_subplot(outer[0, 0])
    qplot = q99.assign(error=q99.q99_relative_error.abs() * 100)
    display = qplot.groupby("method_id").error.agg(histories="size", maximum_error_percent="max", beyond_200_percent=lambda v: int((v>200).sum())).reset_index()
    display["display_max_percent"]=200
    display.to_csv(DERIVED/"Figure2_q99_display_limits.csv",index=False)
    qc["q99_display_limit_audit"]=display.to_dict("records")
    for method in ["TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"]:
        values = np.sort(qplot.loc[qplot.method_id.eq(method), "error"].to_numpy())
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                color=COLORS[method], linestyle=LINESTYLES[method], linewidth=1.05,
                label=method)
    ax.set_xscale("symlog", linthresh=20, linscale=1)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"Absolute $q_{0.99}$ error (%)"+"\n(signed-log scale)", fontsize=7.5)
    ax.set_ylabel("Empirical probability")
    add_panel_title(ax, "a", "Upper-tail comparator")
    light_grid(ax)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=2.4,
              borderaxespad=0.45, labelspacing=0.26)
    ax.text(0.97, 0.095, "EVENT-EMP = EMP\n10,000 / 10,000 endpoint ties",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8)

    # (b--d) history-paired regional differences plus their formal t95 strips.
    region_axes = []
    global_abs = float(paired_ecdf.paired_difference_1000_cny.abs().max())
    scale = 10 ** np.floor(np.log10(max(global_abs, 1.0)))
    xlim = next(
        factor * scale for factor in (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10)
        if factor * scale >= global_abs
    )
    xticks = sorted(set([-xlim, -10, 0, 10, xlim]))
    for i, (mode, letter) in enumerate(zip(REGION_ORDER, ["b", "c", "d"])):
        # Explicit positions preserve the V1.1 two-by-two story architecture.
        slot = outer[0, 1] if i == 0 else (outer[1, 0] if i == 1 else outer[1, 1])
        local = slot.subgridspec(2, 1, height_ratios=[3.25, 0.72], hspace=0.03)
        ax_ecdf = fig.add_subplot(local[0])
        ax_ci = fig.add_subplot(local[1], sharex=ax_ecdf)
        for method_a, method_b, color, linestyle in CONTRASTS:
            label = f"{method_a} − {method_b}"
            subdata = paired_ecdf[
                paired_ecdf.oracle_mode.eq(mode) & paired_ecdf.contrast.eq(f"{method_a} - {method_b}")
            ]
            values = subdata.paired_difference_1000_cny.to_numpy()
            x, counts = np.unique(values, return_counts=True)
            cdf = np.cumsum(counts) / counts.sum()
            if len(x) > 1_000:
                keep = np.unique(np.linspace(0, len(x) - 1, 1_000, dtype=int))
                x, cdf = x[keep], cdf[keep]
            ax_ecdf.step(x, cdf, where="post", color=color, linestyle=linestyle,
                         linewidth=0.72, label=label)
            summary = paired_summary[
                paired_summary.oracle_mode.eq(mode) & paired_summary.contrast.eq(f"{method_a} - {method_b}")
            ].iloc[0]
            y = 2 - CONTRASTS.index((method_a, method_b, color, linestyle))
            lo, mean, hi = summary.t95_low_cny / 1000, summary.mean_cny / 1000, summary.t95_high_cny / 1000
            ax_ci.plot([lo, hi], [y, y], color=color, linewidth=1.4, solid_capstyle="round")
            ax_ci.plot(mean, y, marker="o", color=color, markersize=3.1, zorder=4)
        for target in (ax_ecdf, ax_ci):
            target.set_xscale("symlog", linthresh=2, linscale=0.8)
            target.set_xlim(-xlim, xlim)
            target.set_xticks(xticks)
            target.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            target.axvline(0, color="#777777", linestyle=":", linewidth=0.7, zorder=0)
        ax_ecdf.set_ylim(0, 1.02)
        ax_ecdf.set_yticks([0, 0.5, 1.0])
        ax_ecdf.tick_params(axis="x", labelbottom=False)
        ax_ci.set_ylim(-0.6, 2.6)
        ax_ci.set_yticks([])
        ax_ci.spines["left"].set_visible(False)
        ax_ecdf.set_ylabel("Empirical probability" if i == 1 else "")
        ax_ci.set_xlabel("Paired regret difference\n"+r"($10^3$ CNY/month; signed-log scale)", fontsize=7.5)
        add_panel_title(ax_ecdf, letter, f"{MODE_LABELS[mode]} region")
        light_grid(ax_ecdf)
        region_axes.append(ax_ecdf)

    # Contrast legend occupies its own slim row; it does not obscure data.
    legend_ax = fig.add_subplot(outer[2, :])
    legend_ax.axis("off")
    contrast_handles = [
        Line2D([], [], color=color, linestyle=linestyle, linewidth=1.2,
               label=f"{method_a} − {method_b}\n" + ({("TAIL", "EVENT-EMP"): "same event-level information", ("TAIL", "KDE"): "different information; full procedure", ("GEV", "KDE"): "same monthly-maxima information"}[(method_a,method_b)]))
        for method_a, method_b, color, linestyle in CONTRASTS
    ]
    legend_ax.legend(handles=contrast_handles, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                     ncol=3, frameon=False, handlelength=2.0, columnspacing=1.2, fontsize=6.7)

    # (e--h) the final, no-artificial-upper-bound crossed grid.
    bottom = outer[3, :].subgridspec(1, 4, wspace=0.20)
    bottom_axes = []
    grid_ymax = max(5.0, float(np.ceil(grid_effects.mean_difference_cny.abs().max() / 5000) * 5))
    for j, alpha in enumerate(sorted(design.alpha.unique())):
        ax = fig.add_subplot(bottom[0, j])
        z = grid_effects[np.isclose(grid_effects.alpha, alpha)]
        for method_a, method_b, color, linestyle in CONTRASTS[1:]:
            series = z[z.contrast.eq(f"{method_a} - {method_b}")].sort_values("utilization")
            ax.plot(series.utilization, series.mean_difference_cny / 1000,
                    color=color, linestyle=linestyle, linewidth=1.0)
        ax.axhline(0, color="#666666", linestyle=":", linewidth=0.7)
        ax.set_xlim(0.39, 0.77)
        ax.set_ylim(-grid_ymax, grid_ymax)
        ax.set_xticks([0.40, 0.50, 0.60, 0.70, 0.76])
        ax.set_xlabel(r"Utilization $u$")
        if j == 0:
            ax.set_ylabel("Mean difference\n($10^3$ CNY/month)")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(f"({chr(ord('e') + j)})  $\\alpha={alpha:.2f}$", loc="left", pad=4)
        for mode in MODES:
            rugs = design[(np.isclose(design.alpha, alpha)) & design.oracle_mode.eq(mode)]
            ax.plot(rugs.utilization, np.full(len(rugs), 0.035), linestyle="none",
                    marker=MODE_MARKERS[mode], markersize=3.1, color=MODE_COLORS[mode],
                    transform=ax.get_xaxis_transform(), clip_on=False)
        light_grid(ax)
        bottom_axes.append(ax)
    mode_legend_ax = fig.add_subplot(outer[4, :])
    mode_legend_ax.axis("off")
    mode_handles = [
        Line2D([], [], color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], linestyle="none",
               markersize=4.0, label=MODE_LABELS[mode])
        for mode in MODES
    ]
    mode_legend_ax.legend(handles=mode_handles, loc="center", ncol=3,
                          frameon=False, title="Oracle mode at each evaluated cell (rug)",
                          title_fontsize=6.8, columnspacing=2.0, handletextpad=0.5)

    stem = WORKSPACE / "CSSC_SUBMISSION_20261112/Manuscript/figures/Figure_2"
    export_figure(fig, stem)
    plt.close(fig)
    qc.update({
        "crossed_grid_rows": len(cross),
        "crossed_grid_histories": int(cross.replication_id.nunique()),
        "crossed_grid_cells": int(cross.cell_id.nunique()),
        "crossed_grid_methods": sorted(cross.method_id.unique().tolist()),
        "crossed_grid_policy": sorted(cross.support_policy.unique().tolist()),
        "crossed_grid_rows_per_method_cell": 1000,
        "crossed_grid_alpha_values": sorted(float(v) for v in design.alpha.unique()),
        "crossed_grid_utilization_values_per_alpha": 19,
        "crossed_grid_cell_summary_checks": checks,
        "crossed_grid_selected_contrast_cell_values": len(grid_effects),
        "crossed_grid_contrast_range_1000_cny": [
            float(grid_effects.mean_difference_cny.min() / 1000),
            float(grid_effects.mean_difference_cny.max() / 1000),
        ],
        "crossed_grid_cis": "none; descriptive cell means",
        "regional_ecdf_display_vertices": "exact unique jumps when <=1000; otherwise 1000 rank-spaced vertices; all 10000 paired histories determine the empirical probabilities",
        "figure2_formats": ["pdf", "svg", "eps", "png", "tiff"],
        "raw_crossed_decision_sha256": sha256(CROSS_RAW),
        "crossed_run_manifest_sha256": sha256(CROSS / "raw/run_manifest.json"),
        "figure2_pdf_sha256": sha256(stem.with_suffix(".pdf")),
    })
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "figure2_primary_crossed_qc.json").write_text(json.dumps(qc, indent=2))
    return {"crossed": cross, "design": design, "computed": computed, "grid_effects": grid_effects, "qc": qc}


def draw_figure_s8(payload: dict) -> dict:
    cross = payload["crossed"]
    design = payload["design"]
    cell_summary = cross.groupby(["cell_id", "method_id", "alpha", "utilization"], observed=True).agg(
        mean_normalized_regret=("regret_over_oracle_value", "mean"),
        mode_accuracy=("action_correct", "mean"),
        histories=("replication_id", "size"),
    ).reset_index()
    assert len(cell_summary) == 76 * 5
    assert cell_summary.histories.eq(1000).all()
    cell_summary = cell_summary.merge(
        design[["cell_id", "oracle_mode"]], on="cell_id", validate="many_to_one"
    )
    cell_summary["mean_normalized_regret_percent"] = 100 * cell_summary.mean_normalized_regret
    cell_summary["mode_accuracy_percent"] = 100 * cell_summary.mode_accuracy
    cell_summary.sort_values(["alpha", "utilization", "method_id"]).to_csv(
        ROOT / "data/derived_plotting/FigureS8_crossed_full_method_profiles.csv", index=False
    )

    fig, axes = plt.subplots(2, 4, figsize=(178 * MM, 112 * MM), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.17, wspace=0.18, hspace=0.28)
    for col, alpha in enumerate(sorted(design.alpha.unique())):
        for row, (metric, ylabel, panel_word) in enumerate([
            ("mean_normalized_regret_percent", r"Mean normalized regret (%)", "Mean full-action loss"),
            ("mode_accuracy_percent", "Oracle-mode accuracy (%)", "Mode accuracy"),
        ]):
            ax = axes[row, col]
            z = cell_summary[np.isclose(cell_summary.alpha, alpha)]
            for method in METHODS:
                series = z[z.method_id.eq(method)].sort_values("utilization")
                ax.plot(
                    series.utilization, series[metric],
                    color=COLORS[method], linestyle=LINESTYLES[method], linewidth=0.9,
                    marker=MARKERS[method], markersize=2.0, markeredgewidth=0.35,
                    label=method,
                )
            ax.set_xlim(0.39, 0.77)
            ax.set_xticks([0.40, 0.50, 0.60, 0.70, 0.76])
            if row == 0:
                ax.set_title(r"$\alpha=" + f"{alpha:.2f}" + r"$", pad=4)
                ax.set_ylim(0, 1.75)
            else:
                ax.set_ylim(0, 100)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel(r"Utilization $u$")
            ax.text(0.02, 0.94, "abcdefgh"[row * 4 + col], transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.3, fontweight="bold")
            light_grid(ax, axis="y")
    handles = [
        Line2D([], [], color=COLORS[m], linestyle=LINESTYLES[m], marker=MARKERS[m],
               markersize=3.0, linewidth=1.0, label=m)
        for m in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.99),
               ncol=5, frameon=False, columnspacing=1.35, handlelength=2.1)
    fig.text(0.54, 0.035,
             "Each method–cell profile uses 1,000 shared histories; lines connect the 19 evaluated utilization cells.",
             ha="center", fontsize=7.0)
    stem = WORKSPACE / "CSSC_SUBMISSION_20261112/Supporting_Information/figures/FigureS8"
    export_figure(fig, stem)
    plt.close(fig)
    return {
        "rows": len(cross),
        "cell_method_profiles": len(cell_summary),
        "cells": 76,
        "methods": 5,
        "histories_per_method_cell": 1000,
        "metrics": ["mean normalized full-action regret", "oracle-mode accuracy"],
        "uncertainty": "descriptive profiles; no intervals",
        "formats": ["pdf", "svg", "eps", "png", "tiff"],
        "figure_s8_pdf_sha256": sha256(stem.with_suffix(".pdf")),
    }


def main() -> None:
    payload = draw_figure2()
    s8 = draw_figure_s8(payload)
    qc = payload["qc"]
    qc["figure_s8"] = s8
    (AUDIT / "figure2_primary_crossed_qc.json").write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
