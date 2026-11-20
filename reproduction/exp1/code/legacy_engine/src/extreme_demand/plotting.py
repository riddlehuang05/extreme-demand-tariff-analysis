"""Reproducible, data-driven paper figures for Track A."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


def configure_style(config: Mapping[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": str(config["font_family"]),
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.dpi": int(config["dpi"]),
            "savefig.dpi": int(config["dpi"]),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def save_figure(
    figure: Figure,
    output_stem: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in formats:
        path = output_stem.with_suffix(f".{extension}")
        figure.savefig(path, bbox_inches="tight", dpi=dpi)
        paths.append(path)
    plt.close(figure)
    return paths


def _finish_axes(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.7)


def add_smoke_mark(figure: Figure, text: str | None) -> None:
    if text:
        figure.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=7, color="#777777")


def framework_data() -> pd.DataFrame:
    rows = [
        (0, "Policy", "Tariff modes\nand rates", 0, 0, 1),
        (1, "Process", "EAF heat-level\nevents", 0, 1, 2),
        (2, "Aggregation", "Monthly 15-min\nmaximum", 0, 2, 3),
        (3, "Information", "History and\ntail estimator", 1, 2, 4),
        (4, "Decision", "Mode and\ncontract demand", 1, 1, 5),
        (5, "Evaluation", "Fee, regret and\nmode error", 1, 0, -1),
    ]
    return pd.DataFrame(rows, columns=["node_id", "short_label", "detail", "grid_row", "grid_col", "next_node_id"])


def plot_framework(data: pd.DataFrame, palette: Mapping[str, str], smoke: str | None) -> Figure:
    require_columns(data, ["node_id", "short_label", "detail", "grid_row", "grid_col", "next_node_id"], "framework data")
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    colors = [palette["blue"], palette["orange"], palette["sky_blue"], palette["green"], palette["purple"], palette["vermillion"]]
    positions: dict[int, tuple[float, float]] = {}
    for position, row in data.sort_values("node_id").reset_index(drop=True).iterrows():
        x = float(row["grid_col"]) * 2.6
        y = 1.25 - float(row["grid_row"]) * 1.35
        positions[int(row["node_id"])] = (x, y)
        axis.text(
            x,
            y,
            f"{row['short_label']}\n{row['detail']}",
            ha="center",
            va="center",
            color="white" if position != 2 else "black",
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.55,rounding_size=0.08", "fc": colors[position], "ec": "black", "lw": 0.7},
        )
    for _, row in data.iterrows():
        source = positions[int(row["node_id"])]
        next_id = int(row["next_node_id"])
        if next_id < 0:
            continue
        target = positions[next_id]
        if np.isclose(source[0], target[0]):
            start = (source[0], source[1] - 0.58)
            end = (target[0], target[1] + 0.58)
            shrink = 0
        else:
            start, end, shrink = source, target, 48
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "black", "shrinkA": shrink, "shrinkB": shrink})
    axis.set_xlim(-1.1, 6.3)
    axis.set_ylim(-0.85, 2.0)
    axis.axis("off")
    add_smoke_mark(figure, smoke)
    return figure


def prepare_mechanism(grid: pd.DataFrame) -> pd.DataFrame:
    required = ["mean_target", "upper_probability", "upper_amplitude", "contract_cost_change", "spread_D_over_S", "value_of_tail_information", "mode_switch"]
    require_columns(grid, required, "mechanism grid")
    switch_counts = grid.groupby("mean_target", observed=True)["mode_switch"].sum()
    target = float(switch_counts.sort_values(ascending=False, kind="stable").index[0])
    selected = grid.loc[np.isclose(grid["mean_target"], target), required].copy()
    selected["tail_severity"] = selected["upper_probability"] * selected["upper_amplitude"] ** 2
    return selected.sort_values(["upper_probability", "upper_amplitude"]).reset_index(drop=True)


def plot_mechanism(data: pd.DataFrame, palette: Mapping[str, str], smoke: str | None) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for probability, group in data.groupby("upper_probability", sort=True):
        label = f"upper probability={probability:g}"
        axes[0].plot(group["upper_amplitude"], group["contract_cost_change"], marker="o", ms=3, lw=1.0, label=label)
        axes[1].plot(group["upper_amplitude"], group["value_of_tail_information"], marker="o", ms=3, lw=1.0, label=label)
    switches = data[data["mode_switch"].astype(bool)]
    axes[0].scatter(switches["upper_amplitude"], switches["contract_cost_change"], marker="s", s=28, facecolors="none", edgecolors=palette["vermillion"], label="mode switch")
    axes[0].set(xlabel="Upper-state amplitude", ylabel="Contract-cost change", title="(a) Equal-mean tail cost")
    axes[1].set(xlabel="Upper-state amplitude", ylabel="Value of tail information", title="(b) Decision value")
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[0].legend(frameon=False, loc="best")
    for axis in axes:
        _finish_axes(axis)
    add_smoke_mark(figure, smoke)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def prepare_generator_trace(trace: pd.DataFrame, max_minutes: int = 1440) -> pd.DataFrame:
    required = ["month_id", "minute", "eaf_mw", "background_mw", "poc_mw", "fixed15_mw", "sliding15_mw"]
    require_columns(trace, required, "generator trace")
    month_id = trace["month_id"].min()
    selected = trace[(trace["month_id"] == month_id) & (trace["minute"] < max_minutes)].copy()
    return selected[required].sort_values("minute").reset_index(drop=True)


def plot_generator_trace(data: pd.DataFrame, palette: Mapping[str, str], smoke: str | None) -> Figure:
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True, height_ratios=[1.35, 1.0])
    axes[0].plot(data["minute"] / 60.0, data["eaf_mw"], color=palette["blue"], lw=0.8, label="EAF")
    axes[0].plot(data["minute"] / 60.0, data["background_mw"], color=palette["orange"], lw=0.8, label="Background")
    axes[0].plot(data["minute"] / 60.0, data["poc_mw"], color="black", lw=0.9, label="Point of connection")
    axes[1].plot(data["minute"] / 60.0, data["fixed15_mw"], color=palette["green"], lw=0.9, label="Fixed 15-min")
    axes[1].plot(data["minute"] / 60.0, data["sliding15_mw"], color=palette["purple"], lw=0.9, ls="--", label="Sliding 15-min")
    axes[0].set(ylabel="Power (MW)", title="(a) One-minute process-calibrated load")
    axes[1].set(xlabel="Hour of month trace", ylabel="Demand (MW)", title="(b) Billing-window aggregation")
    for axis in axes:
        axis.legend(frameon=False, ncol=3, loc="upper right")
        _finish_axes(axis)
    add_smoke_mark(figure, smoke)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return figure


def prepare_oracle_surface(paired: pd.DataFrame) -> pd.DataFrame:
    required = ["scenario_id", "physical_scenario_id", "utilization_target", "oracle_mode", "oracle_margin", "capacity_fee_monthly"]
    require_columns(paired, required, "paired results")
    data = paired[required].drop_duplicates().copy()
    if data.duplicated(["scenario_id"]).any():
        spread = data.groupby("scenario_id").agg(mode_count=("oracle_mode", "nunique"), margin_count=("oracle_margin", "nunique"))
        if (spread[["mode_count", "margin_count"]] > 1).any().any():
            raise ValueError("oracle quantities vary within scenario")
        data = data.drop_duplicates("scenario_id")
    data["normalized_margin"] = data["oracle_margin"] / (3.0 * data["capacity_fee_monthly"])
    return data.sort_values(["physical_scenario_id", "utilization_target"]).reset_index(drop=True)


def plot_oracle_surface(data: pd.DataFrame, tail_order: list[str], palette: Mapping[str, str], smoke: str | None) -> Figure:
    require_columns(data, ["physical_scenario_id", "utilization_target", "oracle_mode", "normalized_margin"], "oracle surface")
    modes = sorted(data["oracle_mode"].astype(str).unique())
    mode_colors = {mode: [palette["orange"], palette["blue"], palette["green"]][index % 3] for index, mode in enumerate(modes)}
    figure, axis = plt.subplots(figsize=(7.2, 3.3))
    for _, row in data.iterrows():
        y = tail_order.index(str(row["physical_scenario_id"]))
        size = 25.0 + 700.0 * min(abs(float(row["normalized_margin"])), 0.05)
        axis.scatter(float(row["utilization_target"]), y, s=size, c=mode_colors[str(row["oracle_mode"])], edgecolors="black", linewidths=0.35)
    for mode, color in mode_colors.items():
        axis.scatter([], [], s=35, c=color, edgecolors="black", linewidths=0.35, label=mode)
    axis.set_yticks(range(len(tail_order)), labels=tail_order)
    axis.set(xlabel="Target utilization", ylabel="Tail scenario", title="Oracle tariff mode; marker size is normalized decision margin")
    figure.legend(frameon=False, ncol=max(1, len(modes)), loc="upper center", bbox_to_anchor=(0.5, 0.91))
    _finish_axes(axis)
    add_smoke_mark(figure, smoke)
    figure.tight_layout(rect=(0, 0.03, 1, 0.88))
    return figure


def summarize_regret(paired: pd.DataFrame) -> pd.DataFrame:
    required = ["method_id", "training_months", "regret", "oracle_margin", "capacity_fee_monthly"]
    require_columns(paired, required, "paired results")
    normalized = paired.copy()
    normalized["normalized_regret"] = normalized["regret"] / (3.0 * normalized["capacity_fee_monthly"])
    normalized["normalized_margin"] = normalized["oracle_margin"].abs() / (3.0 * normalized["capacity_fee_monthly"])
    frames: list[pd.DataFrame] = []
    for region, subset in [("All cells", normalized), ("Boundary <=1%", normalized[normalized["normalized_margin"] <= 0.01])]:
        grouped = subset.groupby(["method_id", "training_months"], observed=True)["normalized_regret"]
        summary = grouped.agg(mean_regret="mean", sd_regret="std", n="size").reset_index()
        summary["se_regret"] = summary["sd_regret"] / np.sqrt(summary["n"])
        summary["ci95_low"] = summary["mean_regret"] - 1.96 * summary["se_regret"]
        summary["ci95_high"] = summary["mean_regret"] + 1.96 * summary["se_regret"]
        summary["region"] = region
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def plot_regret(data: pd.DataFrame, method_order: list[str], palette: Mapping[str, str], smoke: str | None) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    colors = [palette["black"], palette["orange"], palette["sky_blue"], palette["green"], palette["blue"], palette["vermillion"], palette["purple"], palette["gray"]]
    for axis, region, panel in zip(axes, ["All cells", "Boundary <=1%"], ["(a) All registered cells", "(b) Oracle boundary band"], strict=True):
        region_data = data[data["region"] == region]
        for index, method in enumerate(method_order):
            group = region_data[region_data["method_id"] == method].sort_values("training_months")
            if group.empty:
                continue
            axis.plot(group["training_months"], group["mean_regret"], marker="o", ms=3, lw=1.0, color=colors[index % len(colors)], label=method)
            if group["ci95_low"].notna().all():
                axis.fill_between(group["training_months"], group["ci95_low"], group["ci95_high"], color=colors[index % len(colors)], alpha=0.10, linewidth=0)
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set(xlabel="Training history (months)", title=panel)
        _finish_axes(axis)
    axes[0].set_ylabel("Normalized regret")
    axes[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.22))
    add_smoke_mark(figure, smoke)
    figure.tight_layout(rect=(0, 0.18, 1, 1))
    return figure


def prepare_vti(paired: pd.DataFrame, baselines: Iterable[str] = ("EMP-JOINT", "KDE-MC-JOINT")) -> pd.DataFrame:
    required = ["replication_id", "scenario_id", "physical_scenario_id", "training_months", "method_id", "regret", "oracle_margin", "capacity_fee_monthly"]
    require_columns(paired, required, "paired results")
    keys = ["replication_id", "scenario_id", "physical_scenario_id", "training_months"]
    pivot = paired.pivot_table(index=keys, columns="method_id", values="regret", aggfunc="first")
    metadata = paired.groupby(keys, observed=True).agg(oracle_margin=("oracle_margin", "first"), capacity_fee_monthly=("capacity_fee_monthly", "first"))
    rows: list[pd.DataFrame] = []
    for baseline in baselines:
        if baseline not in pivot or "TAIL-JOINT" not in pivot:
            continue
        part = metadata.copy()
        part["baseline_id"] = baseline
        part["regret_gain"] = pivot[baseline] - pivot["TAIL-JOINT"]
        part["normalized_oracle_margin"] = part["oracle_margin"] / (3.0 * part["capacity_fee_monthly"])
        rows.append(part.reset_index())
    if not rows:
        raise ValueError("paired results do not contain TAIL-JOINT and requested baselines")
    return pd.concat(rows, ignore_index=True)


def plot_vti(data: pd.DataFrame, palette: Mapping[str, str], smoke: str | None) -> Figure:
    figure, axis = plt.subplots(figsize=(7.2, 3.25))
    styles = [(palette["blue"], "o"), (palette["orange"], "s")]
    for (baseline, group), (color, marker) in zip(data.groupby("baseline_id", sort=True), styles, strict=False):
        axis.scatter(group["normalized_oracle_margin"], group["regret_gain"], s=14, alpha=0.45, color=color, marker=marker, label=baseline)
        ordered = group.sort_values("normalized_oracle_margin")
        rolling = ordered["regret_gain"].rolling(max(3, len(ordered) // 12), min_periods=1, center=True).mean()
        axis.plot(ordered["normalized_oracle_margin"], rolling, color=color, lw=1.2)
    axis.axhline(0.0, color="black", lw=0.7)
    axis.set(xlabel="Absolute oracle margin / quarterly capacity fee", ylabel="Regret reduction from TAIL-JOINT (CNY)", title="Value of process-tail information is localized near decision boundaries")
    axis.legend(frameon=False)
    _finish_axes(axis)
    add_smoke_mark(figure, smoke)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return figure


def summarize_ablation(paired: pd.DataFrame) -> pd.DataFrame:
    required = ["physical_scenario_id", "method_id", "regret", "capacity_fee_monthly"]
    require_columns(paired, required, "paired results")
    normalized = paired.assign(normalized_regret=paired["regret"] / (3.0 * paired["capacity_fee_monthly"]))
    result = normalized.groupby(["physical_scenario_id", "method_id"], observed=True)["normalized_regret"].agg(mean_regret="mean", n="size").reset_index()
    return result


def plot_ablation(data: pd.DataFrame, tail_order: list[str], palette: Mapping[str, str], smoke: str | None) -> Figure:
    methods = [value for value in ["EMP-JOINT", "KDE-MC-JOINT", "GEV-JOINT", "TAIL-NODECLUSTER", "TAIL-JOINT"] if value in set(data["method_id"])]
    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    colors = [palette["gray"], palette["sky_blue"], palette["orange"], palette["purple"], palette["blue"]]
    x = np.arange(len(tail_order), dtype=float)
    for index, method in enumerate(methods):
        values = data[data["method_id"] == method].set_index("physical_scenario_id")["mean_regret"].reindex(tail_order)
        axis.plot(x, values, marker="o", ms=3, lw=1.0, color=colors[index], label=method)
    axis.set_xticks(x, labels=tail_order, rotation=25, ha="right")
    axis.set(xlabel="Truth family / tail severity", ylabel="Mean regret / quarterly capacity fee", title="Estimator ablation across registered truths")
    axis.legend(frameon=False, ncol=3)
    _finish_axes(axis)
    add_smoke_mark(figure, smoke)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return figure


def prepare_window_smoke(monthly: pd.DataFrame) -> pd.DataFrame:
    require_columns(monthly, ["M_fixed15_kw", "M_sliding15_kw"], "monthly calibration")
    difference = monthly["M_sliding15_kw"] - monthly["M_fixed15_kw"]
    n = int(difference.notna().sum())
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return pd.DataFrame([{"contrast_id": "sliding_minus_fixed_peak", "group": "Billing window", "estimate": mean, "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se, "unit": "kW", "n": n}])


def plot_robustness_forest(data: pd.DataFrame, palette: Mapping[str, str], smoke: str | None) -> Figure:
    require_columns(data, ["contrast_id", "estimate", "ci95_low", "ci95_high", "unit"], "robustness results")
    ordered = data.reset_index(drop=True)
    y = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(7.2, max(2.4, 0.36 * len(ordered) + 1.1)))
    lower = ordered["estimate"] - ordered["ci95_low"]
    upper = ordered["ci95_high"] - ordered["estimate"]
    axis.errorbar(ordered["estimate"], y, xerr=np.vstack([lower, upper]), fmt="o", color=palette["blue"], ecolor="black", capsize=2.5, lw=0.8)
    axis.axvline(0.0, color="black", lw=0.7, ls="--")
    axis.set_yticks(y, labels=ordered["contrast_id"])
    axis.set(xlabel=f"Effect estimate ({ordered['unit'].iloc[0]})", ylabel="", title="Policy and engineering robustness")
    axis.invert_yaxis()
    _finish_axes(axis)
    add_smoke_mark(figure, smoke)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return figure


def summarize_external(scores: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    require_columns(scores, ["analysis_variant", "method_id", "crps", "negative_log_score", "stop_loss_score"], "external scores")
    require_columns(decisions, ["analysis_variant", "method_id", "opportunity_loss"], "external decisions")
    score_summary = scores.groupby(["analysis_variant", "method_id"], observed=True).agg(crps=("crps", "mean"), negative_log_score=("negative_log_score", "mean"), stop_loss_score=("stop_loss_score", "mean"), origins=("origin_id", "nunique")).reset_index()
    loss_summary = decisions.groupby(["analysis_variant", "method_id"], observed=True)["opportunity_loss"].mean().rename("mean_opportunity_loss").reset_index()
    return score_summary.merge(loss_summary, on=["analysis_variant", "method_id"], how="left", validate="one_to_one")


def plot_external(data: pd.DataFrame, palette: Mapping[str, str], smoke: str | None) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    methods = sorted(data["method_id"].unique())
    variants = sorted(data["analysis_variant"].unique())
    x = np.arange(len(methods), dtype=float)
    width = 0.8 / max(1, len(variants))
    colors = [palette["blue"], palette["orange"], palette["green"], palette["purple"]]
    for index, variant in enumerate(variants):
        group = data[data["analysis_variant"] == variant].set_index("method_id").reindex(methods)
        offset = (index - (len(variants) - 1) / 2.0) * width
        axes[0].bar(x + offset, group["crps"], width=width, color=colors[index % len(colors)], label=variant)
        axes[1].bar(x + offset, group["mean_opportunity_loss"], width=width, color=colors[index % len(colors)], label=variant)
    for axis, title, ylabel in [(axes[0], "(a) Distribution score", "Mean CRPS"), (axes[1], "(b) Decision interface", "Mean opportunity loss")]:
        axis.set_xticks(x, labels=methods, rotation=30, ha="right")
        axis.set(title=title, ylabel=ylabel)
        _finish_axes(axis)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(1.05, -0.25), ncol=2)
    add_smoke_mark(figure, smoke)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    return figure
