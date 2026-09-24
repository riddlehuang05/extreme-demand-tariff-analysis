from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_revision_experiment as experiment  # noqa: E402


FULL = ROOT / "outputs" / "full" / "revision_results.csv"
FREQUENCY = ROOT / "outputs" / "frequency" / "revision_results.csv"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
R1 = ROOT / "reference_inputs" / "r1"
R1_FILES = (
    "01_mechanism_cell_design.csv",
    "02_mechanism_results_full.csv",
    "05_exact_dgp_recovery_full.csv",
)
SEED = 2026091501


DIAGNOSTICS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "abs_q99_relative_error": lambda frame: frame["q99_relative_error"].abs(),
    "abs_mean_relative_error": lambda frame: frame["mean_relative_error"].abs(),
    "abs_stoploss_relative_error": lambda frame: frame["stoploss_relative_error"].abs(),
    "d_mv": lambda frame: frame["d_mv"],
}


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=float)
    outcomes = np.asarray(labels, dtype=bool)
    positive = int(outcomes.sum())
    negative = int((~outcomes).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = stats.rankdata(values, method="average")
    return float((ranks[outcomes].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def percentile_tail_p(bootstrap: np.ndarray) -> float:
    values = np.asarray(bootstrap, dtype=float)
    return float(
        min(
            1.0,
            2.0
            * (min(int((values <= 0.0).sum()), int((values >= 0.0).sum())) + 1)
            / (values.size + 1),
        )
    )


def paired_inference(values: np.ndarray, rng: np.random.Generator, replicates: int) -> dict[str, float]:
    differences = np.asarray(values, dtype=float)
    indexes = rng.integers(0, differences.size, size=(replicates, differences.size))
    bootstrap = differences[indexes].mean(axis=1)
    bootstrap_median = np.median(differences[indexes], axis=1)
    estimate = float(differences.mean())
    median_estimate = float(np.median(differences))
    sem = float(stats.sem(differences))
    tcrit = float(stats.t.ppf(0.975, differences.size - 1))
    nonzero = differences[differences != 0.0]
    sign_p = float(stats.binomtest(int((nonzero > 0.0).sum()), nonzero.size, 0.5).pvalue) if nonzero.size else 1.0
    return {
        "estimate": estimate,
        "median_paired_difference": median_estimate,
        "median_percentile_ci95_low": float(np.quantile(bootstrap_median, 0.025)),
        "median_percentile_ci95_high": float(np.quantile(bootstrap_median, 0.975)),
        "fraction_negative": float(np.mean(differences < 0.0)),
        "fraction_zero": float(np.mean(differences == 0.0)),
        "fraction_positive": float(np.mean(differences > 0.0)),
        "percentile_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "percentile_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "percentile_tail_area_p": percentile_tail_p(bootstrap),
        "paired_t_ci95_low": estimate - tcrit * sem,
        "paired_t_ci95_high": estimate + tcrit * sem,
        "paired_t_p": float(stats.ttest_1samp(differences, 0.0).pvalue),
        "paired_sign_p": sign_p,
        "replications": int(differences.size),
    }


def method_and_cell_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_level = frame.drop_duplicates(["replication_id", "method_id"])
    methods = (
        fit_level.groupby(["method_id", "information_set", "model_class"])
        .agg(
            replications=("replication_id", "nunique"),
            median_abs_q99_error=("q99_relative_error", lambda values: values.abs().median()),
            mean_abs_q99_error=("q99_relative_error", lambda values: values.abs().mean()),
            median_abs_mean_error=("mean_relative_error", lambda values: values.abs().median()),
            mean_abs_mean_error=("mean_relative_error", lambda values: values.abs().mean()),
        )
        .reset_index()
    )
    cells = (
        frame.groupby(["policy_id", "cell_id", "alpha", "utilization", "oracle_mode", "method_id"])
        .agg(
            replications=("replication_id", "nunique"),
            action_accuracy=("action_correct", "mean"),
            mean_regret_cny=("regret_cny_per_month", "mean"),
            se_regret_cny=("regret_cny_per_month", lambda values: values.std(ddof=1) / math.sqrt(values.size)),
            median_regret_cny=("regret_cny_per_month", "median"),
            p90_regret_cny=("regret_cny_per_month", lambda values: values.quantile(0.90)),
            p95_regret_cny=("regret_cny_per_month", lambda values: values.quantile(0.95)),
            mean_regret_over_oracle=("regret_over_oracle_value", "mean"),
            median_regret_over_oracle=("regret_over_oracle_value", "median"),
            p95_regret_over_oracle=("regret_over_oracle_value", lambda values: values.quantile(0.95)),
        )
        .reset_index()
    )
    return methods, cells


def regional_contrasts(frame: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for metric in ("regret_cny_per_month", "regret_over_oracle_value"):
        paired = (
            frame.groupby(["replication_id", "oracle_mode", "method_id"])[metric]
            .mean()
            .unstack("method_id")
        )
        comparisons = [
            ("TAIL", "EVENT-EMP"),
            ("TAIL", "GEV"),
            ("TAIL", "KDE"),
            ("TAIL", "EMP"),
            ("GEV", "KDE"),
            ("GEV", "EMP"),
        ]
        for region in sorted(frame["oracle_mode"].unique()):
            block = paired.xs(region, level="oracle_mode")
            for method_a, method_b in comparisons:
                result = paired_inference(
                    (block[method_a] - block[method_b]).to_numpy(dtype=float), rng, replicates
                )
                rows.append(
                    {
                        "metric": metric,
                        "oracle_mode": region,
                        "contrast": f"{method_a}-minus-{method_b}",
                        "same_information_set": bool(
                            (method_a in {"TAIL", "EVENT-EMP"} and method_b in {"TAIL", "EVENT-EMP"})
                            or (method_a in {"GEV", "KDE", "EMP"} and method_b in {"GEV", "KDE", "EMP"})
                        ),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def diagnostic_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for cell_id, group in frame.groupby("cell_id"):
        labels = ~group["action_correct"].to_numpy(dtype=bool)
        gap = float(group["oracle_margin_cny"].iloc[0])
        best = float(group["oracle_best_value_cny"].iloc[0])
        for diagnostic, function in DIAGNOSTICS.items():
            rows.append(
                {
                    "cell_id": cell_id,
                    "oracle_mode": group["oracle_mode"].iloc[0],
                    "relative_gap": gap / best,
                    "diagnostic": diagnostic,
                    "auc_within_cell": auc(function(group).to_numpy(dtype=float), labels),
                    "mode_error_prevalence": float(labels.mean()),
                }
            )
    by_cell = pd.DataFrame(rows)
    macro = (
        by_cell.groupby("diagnostic")
        .agg(
            macro_auc=("auc_within_cell", "mean"),
            min_cell_auc=("auc_within_cell", "min"),
            max_cell_auc=("auc_within_cell", "max"),
            cells_with_both_classes=("auc_within_cell", "count"),
        )
        .reset_index()
    )

    method_rows = []
    for (cell_id, method_id), group in frame.groupby(["cell_id", "method_id"]):
        labels = ~group["action_correct"].to_numpy(dtype=bool)
        for diagnostic, function in DIAGNOSTICS.items():
            method_rows.append(
                {
                    "cell_id": cell_id,
                    "method_id": method_id,
                    "diagnostic": diagnostic,
                    "auc_within_cell_method": auc(function(group).to_numpy(dtype=float), labels),
                    "mode_errors": int(labels.sum()),
                    "correct_modes": int((~labels).sum()),
                }
            )
    by_cell_method = pd.DataFrame(method_rows)

    association_rows = []
    fit_units = (
        frame.assign(
            abs_q99_relative_error=frame["q99_relative_error"].abs(),
            abs_mean_relative_error=frame["mean_relative_error"].abs(),
            abs_stoploss_relative_error=frame["stoploss_relative_error"].abs(),
        )
        .groupby(["replication_id", "method_id"])
        .agg(
            regret=("regret_cny_per_month", "mean"),
            abs_q99_relative_error=("abs_q99_relative_error", "mean"),
            abs_mean_relative_error=("abs_mean_relative_error", "mean"),
            abs_stoploss_relative_error=("abs_stoploss_relative_error", "mean"),
            d_mv=("d_mv", "mean"),
        )
        .reset_index()
    )
    for method_id, group in fit_units.groupby("method_id"):
        for diagnostic in DIAGNOSTICS:
            association_rows.append(
                {
                    "method_id": method_id,
                    "diagnostic": diagnostic,
                    "spearman_rho": float(stats.spearmanr(group[diagnostic], group["regret"]).statistic),
                    "replications": int(len(group)),
                }
            )
    ranked = fit_units.copy()
    for column in [*DIAGNOSTICS, "regret"]:
        ranked[column] = ranked.groupby("method_id")[column].rank(pct=True, method="average")
    for diagnostic in DIAGNOSTICS:
        association_rows.append(
            {
                "method_id": "WITHIN_METHOD_RANK_POOLED",
                "diagnostic": diagnostic,
                "spearman_rho": float(stats.spearmanr(ranked[diagnostic], ranked["regret"]).statistic),
                "replications": int(len(ranked)),
            }
        )
    associations = pd.DataFrame(association_rows)
    return by_cell, macro, by_cell_method, associations


def certificate_table(frame: pd.DataFrame) -> pd.DataFrame:
    error = ~frame["action_correct"].astype(bool)
    unsafe = frame["d_mv"] >= 1.0
    return pd.DataFrame(
        [
            {
                "rows": int(len(frame)),
                "mode_errors": int(error.sum()),
                "safe_rows_d_mv_below_one": int((~unsafe).sum()),
                "safe_fraction": float((~unsafe).mean()),
                "safe_zone_mode_errors": int((error & ~unsafe).sum()),
                "sensitivity_of_d_mv_ge_one_for_mode_error": float(unsafe[error].mean()),
                "ppv_of_d_mv_ge_one_for_mode_error": float(error[unsafe].mean()),
                "specificity_of_d_mv_below_one": float((~unsafe)[~error].mean()),
            }
        ]
    )


def fit_diagnostic_frequencies(frame: pd.DataFrame) -> pd.DataFrame:
    fits = frame.drop_duplicates(["replication_id", "method_id"])
    tail = fits[fits["method_id"] == "TAIL"].copy()
    return (
        tail.groupby(
            ["fit_status", "selected_threshold_quantile", "count_model", "finite_variance_flag"],
            dropna=False,
        )
        .size()
        .rename("replications")
        .reset_index()
    )


def frequency_table(frame: pd.DataFrame) -> pd.DataFrame:
    fit_level = frame.drop_duplicates(["tail_rate_per_month", "replication_id", "method_id"])
    return (
        fit_level.groupby(["tail_rate_per_month", "tail_event_probability_per_month", "method_id"])
        .agg(
            replications=("replication_id", "nunique"),
            median_abs_q99_error=("q99_relative_error", lambda values: values.abs().median()),
            mean_abs_q99_error=("q99_relative_error", lambda values: values.abs().mean()),
            fallback_fraction=("fit_status", lambda values: values.str.startswith("FALLBACK").mean()),
        )
        .reset_index()
        .merge(
            frame.groupby(["tail_rate_per_month", "method_id"])
            .agg(
                action_accuracy=("action_correct", "mean"),
                mean_regret_over_oracle=("regret_over_oracle_value", "mean"),
                p95_regret_over_oracle=("regret_over_oracle_value", lambda values: values.quantile(0.95)),
            )
            .reset_index(),
            on=["tail_rate_per_month", "method_id"],
        )
    )


def exact_oracle_audit(frame: pd.DataFrame) -> pd.DataFrame:
    old_design = pd.read_csv(R1 / "01_mechanism_cell_design.csv")
    settings = experiment.load_settings()
    oracle = experiment.OracleDistribution(experiment.dgp_from_settings(settings))
    cells = experiment.policies_and_cells(settings, oracle, False)
    new_rows = []
    for cell in cells:
        decision = cell["oracle"]
        new_rows.append(
            {
                "cell_id": cell["cell_id"],
                "analytic_S_kva": cell["S_kva"],
                "analytic_oracle_mode": decision["selected_mode"],
                "analytic_oracle_best_value_cny": decision["best_value_cny"],
                "analytic_oracle_margin_cny": decision["margin_cny"],
                "analytic_oracle_contract_D_kw": decision["contract_D_kw"],
            }
        )
    audit = old_design.merge(pd.DataFrame(new_rows), on="cell_id")
    audit["fixed_mc_mean_kw"] = float(frame["true_mean_kw"].iloc[0]) + 106.0  # overwritten below
    old_results = pd.read_csv(R1 / "02_mechanism_results_full.csv", nrows=1)
    audit["fixed_mc_mean_kw"] = float(old_results["true_mean_kw"].iloc[0])
    audit["analytic_mean_kw"] = oracle.mean()
    audit["mean_bias_fixed_mc_kw"] = audit["fixed_mc_mean_kw"] - audit["analytic_mean_kw"]
    audit["fixed_mc_q99_kw"] = float(old_results["true_q99_kw"].iloc[0])
    audit["analytic_q99_kw"] = oracle.ppf(0.99)
    audit["q99_bias_fixed_mc_kw"] = audit["fixed_mc_q99_kw"] - audit["analytic_q99_kw"]
    audit["margin_change_cny"] = audit["analytic_oracle_margin_cny"] - audit["oracle_margin_cny"]
    audit["mode_changed"] = audit["analytic_oracle_mode"] != audit["oracle_mode"]
    return audit


def conditional_coverage_table() -> pd.DataFrame:
    source = pd.read_csv(R1 / "05_exact_dgp_recovery_full.csv")
    rows = []
    for months, group in source.groupby("training_months"):
        for parameter, column in (("xi", "xi_covered"), ("sigma", "sigma_covered")):
            values = group[column].astype(bool)
            successes, total = int(values.sum()), int(values.size)
            low = 0.0 if successes == 0 else float(stats.beta.ppf(0.025, successes, total - successes + 1))
            high = 1.0 if successes == total else float(stats.beta.ppf(0.975, successes + 1, total - successes))
            rows.append(
                {
                    "training_months": int(months),
                    "parameter": parameter,
                    "covered": successes,
                    "replications": total,
                    "conditional_coverage": successes / total,
                    "clopper_pearson_mc_ci95_low": low,
                    "clopper_pearson_mc_ci95_high": high,
                    "coverage_scope": "conditional_on_selected_threshold",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(FULL)
    frequency = pd.read_csv(FREQUENCY)
    settings = experiment.load_settings()
    bootstrap_replicates = int(settings["inference_bootstrap_replicates"])

    methods, cells = method_and_cell_tables(frame)
    contrasts = regional_contrasts(frame, bootstrap_replicates)
    auc_cell, auc_macro, auc_cell_method, associations = diagnostic_tables(frame)
    certificate = certificate_table(frame)
    tail_frequencies = fit_diagnostic_frequencies(frame)
    frequency_summary = frequency_table(frequency)
    has_r1_inputs = all((R1 / name).is_file() for name in R1_FILES)

    outputs = {
        "method_fidelity_summary.csv": methods,
        "cell_decision_summary.csv": cells,
        "regional_paired_contrasts.csv": contrasts,
        "diagnostic_auc_by_cell.csv": auc_cell,
        "diagnostic_auc_macro.csv": auc_macro,
        "diagnostic_auc_by_cell_method.csv": auc_cell_method,
        "diagnostic_spearman_by_method.csv": associations,
        "d_mv_certificate.csv": certificate,
        "tail_selection_frequencies.csv": tail_frequencies,
        "frequency_sensitivity_summary.csv": frequency_summary,
    }
    if has_r1_inputs:
        oracle_audit = exact_oracle_audit(frame)
        outputs["exact_oracle_bias_audit.csv"] = oracle_audit
        outputs["conditional_coverage_mc_intervals.csv"] = conditional_coverage_table()
    for name, table in outputs.items():
        table.to_csv(TABLES / name, index=False)

    tail_status = (
        frame.drop_duplicates(["replication_id", "method_id"])
        .query("method_id == 'TAIL'")["fit_status"]
        .value_counts()
        .to_dict()
    )
    report = {
        "main_replications": int(frame["replication_id"].nunique()),
        "main_rows": int(len(frame)),
        "methods": sorted(frame["method_id"].unique()),
        "analytic_oracle": {
            "mean_kw": float(frame["true_mean_kw"].iloc[0]),
            "q99_kw": float(frame["true_q99_kw"].iloc[0]),
            "q999_kw": experiment.OracleDistribution(
                experiment.dgp_from_settings(settings)
            ).ppf(0.999),
        },
        "legacy_r1_comparison_included": has_r1_inputs,
        "d_mv_certificate": certificate.iloc[0].to_dict(),
        "tail_fit_status_counts": tail_status,
        "bootstrap_replicates": bootstrap_replicates,
    }
    if has_r1_inputs:
        report["analytic_oracle"].update({
            "old_fixed_mc_mean_bias_kw": float(oracle_audit["mean_bias_fixed_mc_kw"].iloc[0]),
            "old_fixed_mc_q99_bias_kw": float(oracle_audit["q99_bias_fixed_mc_kw"].iloc[0]),
            "oracle_mode_changes": int(oracle_audit["mode_changed"].sum()),
        })
    (REPORTS / "analysis_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
