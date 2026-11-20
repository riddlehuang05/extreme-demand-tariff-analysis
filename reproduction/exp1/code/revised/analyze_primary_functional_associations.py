#!/usr/bin/env python3
"""Teacher/Sonnet add-on summaries from the enriched primary decision table.

This script is deliberately post-processing only: it does not fit estimators,
modify the primary runner, or alter the base engine.  Its correlation family
matches the ten pointwise contrasts in the original analysis (five methods x
two absolute-functional-error versus absolute-q99-error Spearman differences).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t


METHODS = ("TAIL", "EVENT-EMP", "GEV", "KDE", "EMP")
MODE_ACTUAL = "actual_maximum_demand"
MODE_CONTRACT = "contracted_maximum_demand"
MODE_CAPACITY = "capacity"
PRIMARY_PAIRS = {frozenset(("GEV", "KDE")), frozenset(("TAIL", "EVENT-EMP"))}
ERRORS = {
    "q99": "q99_relative_error",
    "mean": "mean_relative_error",
    "stoploss": "stoploss_relative_error",
}


def mean_ci(values: np.ndarray) -> tuple[float, float, float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    se = float(sd / np.sqrt(n)) if n > 1 else np.nan
    half = float(student_t.ppf(.975, n - 1) * se) if n > 1 else np.nan
    return mean, sd, se, half, n


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlations row-wise for equally sized 2D arrays."""
    rx = rankdata(np.asarray(x, dtype=float), axis=1, method="average")
    ry = rankdata(np.asarray(y, dtype=float), axis=1, method="average")
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", rx, ry, optimize=True)
    denominator = np.sqrt(
        np.einsum("ij,ij->i", rx, rx, optimize=True)
        * np.einsum("ij,ij->i", ry, ry, optimize=True)
    )
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def clustered_group_summary(frame: pd.DataFrame, group_cols: list[str], value_col: str,
                            history_col: str = "replication_id") -> pd.DataFrame:
    """Means/intervals with histories, not tariff rows, as independent units."""
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        history_means = group.groupby(history_col, sort=False)[value_col].mean().to_numpy(float)
        mean, sd, se, half, n = mean_ci(history_means)
        row = dict(zip(group_cols, keys))
        row.update({
            "decision_rows": int(len(group)), "independent_histories": n,
            "mean": mean, "sd_across_history_means": sd,
            "mcse": se, "t95_low": mean - half if n > 1 else np.nan,
            "t95_high": mean + half if n > 1 else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_fig3d(df: pd.DataFrame, out: Path) -> None:
    subset = df.loc[
        df["oracle_mode"].eq(MODE_CONTRACT) & df["action_correct"].astype(bool)
    ].copy()
    if subset.empty:
        raise ValueError("No oracle-contract, mode-correct rows were found")
    if not np.isfinite(subset["d_mv"].to_numpy(dtype=float)).all():
        raise ValueError("d_MV must be finite on oracle-contract, mode-correct rows")
    metrics = ["regret_cny_per_month", "mode_selection_regret_cny", "within_contract_regret_cny"]
    if not set(metrics).issubset(subset.columns):
        raise ValueError("Enriched table is missing regret decomposition fields")
    # At a correctly selected contract mode, mode-selection regret is exactly
    # zero and full regret equals the within-contract component.  We retain both
    # fields so the rowwise decomposition remains auditable.
    def summarize_certificate(certificate_type: str, certified_mask: pd.Series) -> None:
        tagged = subset.copy()
        tagged["certificate_type"] = certificate_type
        tagged["certificate_status"] = np.where(certified_mask, "certified", "uncertified")
        cell_keys = ["method_id", "cell_id", "certificate_type", "certificate_status"]
        pooled_keys = ["method_id", "certificate_type", "certificate_status"]
        cell_frames = []
        pooled_frames = []
        for metric in metrics:
            a = clustered_group_summary(tagged, cell_keys, metric).rename(columns={"mean": f"{metric}_mean"})
            cell_frames.append(a)
            b = clustered_group_summary(tagged, pooled_keys, metric).rename(columns={"mean": f"{metric}_mean"})
            pooled_frames.append(b)
        cell_summary = cell_frames[0]
        for part in cell_frames[1:]:
            cell_summary = cell_summary.merge(part, on=cell_keys, validate="one_to_one", suffixes=("", "_duplicate"))
            cell_summary = cell_summary.loc[:, ~cell_summary.columns.str.endswith("_duplicate")]
        pooled_summary = pooled_frames[0]
        for part in pooled_frames[1:]:
            pooled_summary = pooled_summary.merge(part, on=pooled_keys, validate="one_to_one", suffixes=("", "_duplicate"))
            pooled_summary = pooled_summary.loc[:, ~pooled_summary.columns.str.endswith("_duplicate")]
        stem = "teacher_fig3d_dmv" if certificate_type == "d_mv_lt_1" else "teacher_fig3d_pairwise"
        cell_summary.to_csv(out / f"{stem}_certificate_by_cell.csv", index=False)
        pooled_summary.to_csv(out / f"{stem}_certificate_pooled.csv", index=False)

    # Match original Figure 3(d) first; pairwise is an additional parallel split.
    summarize_certificate("d_mv_lt_1", subset["d_mv"] < 1.0)
    summarize_certificate("pairwise_gap", subset["pairwise_gap_certificate"].astype(bool))
    checks = {
        "subset_definition": "oracle_mode == contracted_maximum_demand AND action_correct",
        "primary_certificate_split": "d_MV < 1 certified vs d_MV >= 1 uncertified (original Figure 3d definition)",
        "additional_certificate_split": "pairwise_gap_certificate true vs false",
        "decision_rows": int(len(subset)),
        "histories": int(subset.replication_id.nunique()),
        "maximum_abs_mode_selection_regret_cny": float(subset.mode_selection_regret_cny.abs().max()),
        "maximum_abs_decomposition_error_cny": float(
            (subset.regret_cny_per_month - subset.mode_selection_regret_cny
             - subset.within_contract_regret_cny).abs().max()
        ),
        "interpretation": "Conditional on oracle contract and correct mode, total regret is within-contract loss; uncertainty uses history-clustered means.",
    }
    (out / "teacher_fig3d_contract_correct_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )


def summarize_paired_differences(joined: pd.DataFrame, value: str,
                                 group_cols: list[str], out_rows: list[dict]) -> None:
    for keys, group in joined.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        per_history = group.groupby("replication_id", sort=False)[value].mean().to_numpy(float)
        mean, sd, se, half, n = mean_ci(per_history)
        out_rows.append({
            **dict(zip(group_cols, keys)), "independent_histories": n,
            "paired_difference_mean": mean, "sd": sd, "mcse": se,
            "t95_low": mean - half if n > 1 else np.nan,
            "t95_high": mean + half if n > 1 else np.nan,
        })


def build_contrasts(df: pd.DataFrame, out: Path) -> None:
    pair_rows = []
    for method_a, method_b in itertools.combinations(METHODS, 2):
        a = df.loc[df.method_id.eq(method_a), ["replication_id", "cell_id", "alpha", "oracle_mode", "regret_cny_per_month"]]
        b = df.loc[df.method_id.eq(method_b), ["replication_id", "cell_id", "alpha", "oracle_mode", "regret_cny_per_month"]]
        joined = a.merge(
            b, on=["replication_id", "cell_id", "alpha", "oracle_mode"],
            suffixes=("_a", "_b"), validate="one_to_one",
        )
        joined["difference"] = joined.regret_cny_per_month_a - joined.regret_cny_per_month_b
        is_primary = frozenset((method_a, method_b)) in PRIMARY_PAIRS
        # Same-information pair contrasts are prespecified primary estimands;
        # cross-information comparisons remain descriptive pipeline contrasts.
        for region, g in joined.groupby("oracle_mode", sort=True):
            alpha = g.loc[np.isclose(g.alpha, 1.20)]
            if alpha.empty:
                continue
            per_history = alpha.groupby("replication_id", sort=False)["difference"].mean().to_numpy(float)
            mean, sd, se, half, n = mean_ci(per_history)
            pair_rows.append({
                "scope": "alpha_1.20_oracle_region", "oracle_mode": region,
                "cell_ids": "|".join(sorted(alpha.cell_id.unique())),
                "alpha": 1.20, "method_a": method_a, "method_b": method_b,
                "primary_same_information_contrast": is_primary,
                "information_comparison": "same_information" if is_primary else "cross_information_or_other",
                "independent_histories": n, "paired_difference_mean": mean,
                "sd": sd, "mcse": se,
                "t95_low": mean - half if n > 1 else np.nan,
                "t95_high": mean + half if n > 1 else np.nan,
            })
        # Global equal-cell-weighted paired contrast after deleting each cell.
        all_cells = sorted(joined.cell_id.unique())
        for omitted in all_cells:
            keep = joined.loc[~joined.cell_id.eq(omitted)]
            per_history = keep.groupby("replication_id", sort=False)["difference"].mean().to_numpy(float)
            mean, sd, se, half, n = mean_ci(per_history)
            pair_rows.append({
                "scope": "leave_one_cell_out_all_nine", "oracle_mode": "all_regions",
                "cell_ids": "|".join(c for c in all_cells if c != omitted),
                "omitted_cell": omitted, "alpha": "mixed",
                "method_a": method_a, "method_b": method_b,
                "primary_same_information_contrast": is_primary,
                "information_comparison": "same_information" if is_primary else "cross_information_or_other",
                "independent_histories": n, "paired_difference_mean": mean,
                "sd": sd, "mcse": se,
                "t95_low": mean - half if n > 1 else np.nan,
                "t95_high": mean + half if n > 1 else np.nan,
            })
    pd.DataFrame(pair_rows).to_csv(out / "teacher_alpha120_and_leave_one_cell_out_contrasts.csv", index=False)


def build_mechanism_correlations(df: pd.DataFrame, out: Path) -> None:
    # Correlations are diagnostic, not causal.  Endpoints are aligned to the
    # tariff mechanism and stratified before estimation, avoiding pooled-region
    # correlations that mix different regret-generating channels.
    maps = {
        MODE_ACTUAL: {
            "mechanism": "actual_mean_demand",
            "errors": ["delta_mean_kw", "delta_actual_cny"],
            "outcomes": ["mode_selection_regret_cny", "regret_cny_per_month"],
        },
        MODE_CONTRACT: {
            "mechanism": "contract_stoploss_and_optimizer",
            "errors": ["delta_stoploss_kw", "delta_q_p_star_kw", "delta_es_p_star_kw",
                       "delta_contract_cny", "contract_D_error_kw"],
            "outcomes": ["within_contract_regret_cny", "mode_selection_regret_cny",
                         "regret_cny_per_month"],
        },
        MODE_CAPACITY: {
            "mechanism": "capacity_competing_modes",
            "errors": ["delta_actual_cny", "delta_contract_cny"],
            "outcomes": ["mode_selection_regret_cny", "regret_cny_per_month"],
        },
    }
    rows = []
    for (method, cell, mode), group in df.groupby(["method_id", "cell_id", "oracle_mode"], sort=True):
        spec = maps.get(mode)
        if spec is None:
            continue
        errors = [col for col in spec["errors"] if col in group.columns]
        outcomes = [col for col in spec["outcomes"] if col in group.columns]
        for error in errors:
            for outcome in outcomes:
                for transform, x in [("signed", group[error]), ("absolute", group[error].abs())]:
                    pair = pd.DataFrame({"x": x, "y": group[outcome]}).dropna()
                    rho = float(pd.Series(pair.x).corr(pd.Series(pair.y), method="spearman")) if len(pair) > 1 else np.nan
                    rows.append({
                        "method_id": method, "cell_id": cell, "oracle_mode": mode,
                        "tariff_mechanism": spec["mechanism"], "error_field": error,
                        "error_transform": transform, "outcome_field": outcome,
                        "spearman_rho": rho, "decision_rows": len(pair),
                        "independent_histories": int(group.replication_id.nunique()),
                        "inference": "descriptive; no interval; mechanism-stratified",
                    })
    pd.DataFrame(rows).to_csv(out / "mechanism_aligned_error_correlations.csv", index=False)


def build_original_correlation_family(df: pd.DataFrame, out: Path,
                                      bootstrap_replicates: int, seed: int,
                                      batch_size: int) -> None:
    required = ["replication_id", "method_id", "cell_id", "regret_cny_per_month",
                *ERRORS.values()]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Correlation family requires missing columns: {missing}")
    frame = df.copy()
    # Match the original estimand: first take absolute relative errors per
    # history/cell, then average across all nine equally represented cells.
    for col in ERRORS.values():
        frame[col] = frame[col].abs()
    units = frame.groupby(["replication_id", "method_id"], as_index=False).agg(
        regret=("regret_cny_per_month", "mean"),
        abs_q99=(ERRORS["q99"], "mean"),
        abs_mean=(ERRORS["mean"], "mean"),
        abs_stoploss=(ERRORS["stoploss"], "mean"),
    )
    histories = np.sort(units.replication_id.unique())
    methods = [m for m in METHODS if m in set(units.method_id)]
    if len(methods) != 5 or len(histories) < 3:
        raise ValueError("Expected all five methods and at least three complete histories")
    wide = units.pivot(index="replication_id", columns="method_id", values=["regret", "abs_q99", "abs_mean", "abs_stoploss"]).reindex(histories)
    if wide.isna().any().any():
        raise ValueError("Correlation family needs a complete paired history × method table")
    points = []
    for method in methods:
        y = wide[("regret", method)].to_numpy(float)
        r_q = float(spearman_rows(wide[("abs_q99", method)].to_numpy(float)[None, :], y[None, :])[0])
        r_m = float(spearman_rows(wide[("abs_mean", method)].to_numpy(float)[None, :], y[None, :])[0])
        r_s = float(spearman_rows(wide[("abs_stoploss", method)].to_numpy(float)[None, :], y[None, :])[0])
        points.extend([
            {"method_id": method, "contrast": "abs_mean_error_minus_abs_q99_error", "rho_error": r_m, "rho_q99": r_q, "estimate": r_m-r_q},
            {"method_id": method, "contrast": "abs_stoploss_error_minus_abs_q99_error", "rho_error": r_s, "rho_q99": r_q, "estimate": r_s-r_q},
        ])
    point_df = pd.DataFrame(points)
    n = len(histories)
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap_replicates, len(point_df)), dtype=float)
    # Resample history IDs jointly across methods and all diagnostics.  The
    # maxT distribution then preserves both shared outcomes and paired methods.
    cursor = 0
    for start in range(0, bootstrap_replicates, batch_size):
        b = min(batch_size, bootstrap_replicates - start)
        indexes = rng.integers(0, n, size=(b, n))
        for method in methods:
            y = wide[("regret", method)].to_numpy(float)[indexes]
            rho_q = spearman_rows(wide[("abs_q99", method)].to_numpy(float)[indexes], y)
            rho_m = spearman_rows(wide[("abs_mean", method)].to_numpy(float)[indexes], y)
            rho_s = spearman_rows(wide[("abs_stoploss", method)].to_numpy(float)[indexes], y)
            boot[start:start+b, cursor] = rho_m - rho_q
            boot[start:start+b, cursor+1] = rho_s - rho_q
            cursor += 2
        cursor = 0
    if not np.isfinite(boot).all():
        raise ValueError("Nonfinite bootstrap correlation contrast; inspect ties/constant diagnostics")
    estimates = point_df.estimate.to_numpy(float)
    se = boot.std(axis=0, ddof=1)
    if np.any(se <= 0):
        raise ValueError("Degenerate bootstrap SE in the ten-contrast family")
    centered_t = (boot - estimates[None, :]) / se[None, :]
    max_abs_t = np.max(np.abs(centered_t), axis=1)
    critical = float(np.quantile(max_abs_t, .95, method="higher"))
    observed_t = np.abs(estimates / se)
    adjusted_p = np.array([
        (1 + np.count_nonzero(max_abs_t >= value)) / (bootstrap_replicates + 1)
        for value in observed_t
    ])
    point_df["bootstrap_se"] = se
    point_df["simultaneous_maxT95_low"] = estimates - critical * se
    point_df["simultaneous_maxT95_high"] = estimates + critical * se
    point_df["single_step_maxT_adjusted_p"] = adjusted_p
    point_df["histories"] = n
    point_df["bootstrap_replicates"] = bootstrap_replicates
    point_df["bootstrap_seed"] = seed
    point_df["family_size"] = len(point_df)
    point_df["interval_method"] = "paired history bootstrap; centered studentized single-step max-|T|"
    point_df.to_csv(out / "original_10_correlation_family_maxT.csv", index=False)
    metadata = {
        "family_definition": "For each of five methods, Spearman(abs mean relative error, history-mean regret) minus Spearman(abs q99 relative error, regret), and the analogous abs stop-loss contrast: 10 contrasts total.",
        "aggregation": "Absolute cell-level relative errors and monthly regret are averaged across nine tariff cells within history and method before ranking.",
        "resampling_unit": "replication_id; same resampled history indices shared by all methods and contrasts.",
        "method": "Centered studentized paired-history bootstrap max-|T| single-step familywise 95% intervals and adjusted p-values.",
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": seed,
        "maxT_critical_95": critical, "histories": n,
        "interpretation_limit": "Approximate simultaneous inference under the history bootstrap; contrasts are associations, not causal effects.",
    }
    (out / "original_10_correlation_family_maxT.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Enriched primary_decisions_enriched_10000.csv.gz produced by analyze_primary_10000.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261006)
    parser.add_argument("--bootstrap-batch-size", type=int, default=50)
    parser.add_argument("--expected-n", "--expected-replications", dest="expected_n", type=int, default=None,
                        help="Optional expected history count; any positive N is accepted for pilot/chunk inputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        raise FileNotFoundError(f"Enriched decisions not found: {args.data}")
    if args.bootstrap_replicates < 100 or args.bootstrap_batch_size < 1:
        raise ValueError("Use at least 100 bootstrap replicates and a positive batch size")
    df = pd.read_csv(args.data)
    required = {
        "replication_id", "method_id", "cell_id", "alpha", "oracle_mode", "action_correct",
        "regret_cny_per_month", "mode_selection_regret_cny", "within_contract_regret_cny",
        "pairwise_gap_certificate", "delta_mean_kw", "delta_stoploss_kw", "delta_q99_kw",
        "delta_q_p_star_kw", "delta_es_p_star_kw", "delta_actual_cny", "delta_contract_cny",
        "estimated_contract_D_kw", "oracle_contract_D_kw", "d_mv",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Enriched decision table lacks required fields: {missing}")
    key = ["replication_id", "method_id", "cell_id"]
    if df.duplicated(key).any() or df[key].isna().any().any():
        raise ValueError("Decision key must be complete and unique")
    if set(df.method_id.unique()) != set(METHODS):
        raise ValueError(f"Expected methods {METHODS}; found {sorted(df.method_id.unique())}")
    if (df.groupby(["replication_id", "method_id"]).cell_id.nunique() != 9).any():
        raise ValueError("Each history × method must have all nine tariff cells")
    observed_n = int(df.replication_id.nunique())
    if args.expected_n is not None and (args.expected_n < 1 or observed_n != args.expected_n):
        raise ValueError(f"Expected {args.expected_n} histories, found {observed_n}")
    df["contract_D_error_kw"] = df["estimated_contract_D_kw"] - df["oracle_contract_D_kw"]
    build_fig3d(df, args.out)
    build_contrasts(df, args.out)
    build_mechanism_correlations(df, args.out)
    build_original_correlation_family(df, args.out, args.bootstrap_replicates,
                                      args.bootstrap_seed, args.bootstrap_batch_size)
    summary = {
        "input": str(args.data.resolve()), "decision_rows": int(len(df)),
        "histories": observed_n, "expected_n_checked": args.expected_n,
        "methods": list(METHODS),
        "cells": int(df.cell_id.nunique()),
        "alpha_1_20_cells": sorted(df.loc[np.isclose(df.alpha, 1.20), "cell_id"].unique().tolist()),
        "primary_same_information_pairs": ["GEV vs KDE", "TAIL vs EVENT-EMP"],
        "outputs": [
            "teacher_fig3d_dmv_certificate_by_cell.csv",
            "teacher_fig3d_dmv_certificate_pooled.csv",
            "teacher_fig3d_pairwise_certificate_by_cell.csv",
            "teacher_fig3d_pairwise_certificate_pooled.csv",
            "teacher_fig3d_contract_correct_checks.json",
            "teacher_alpha120_and_leave_one_cell_out_contrasts.csv",
            "mechanism_aligned_error_correlations.csv",
            "original_10_correlation_family_maxT.csv",
            "original_10_correlation_family_maxT.json",
        ],
        "bootstrap_replicates": args.bootstrap_replicates,
        "note": "Post-processing only; no refitting or resimulation.",
    }
    (args.out / "teacher_sonnet_additions_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
