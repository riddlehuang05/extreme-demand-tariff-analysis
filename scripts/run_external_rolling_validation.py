from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor_track_a"))

from track_a.estimators import FitResult, fit_empirical, fit_gev, fit_kde, fit_point  # noqa: E402
from track_a.external.scoring import empirical_crps, kde_negative_log_score  # noqa: E402
from track_a.external.tail import (  # noqa: E402
    RunsCandidate,
    _monthly_reconstruction,
    _parametric_bootstrap,
    runs_cluster_peaks,
)


CONFIG = ROOT / "configs" / "external_rolling_validation.yaml"
INPUT = ROOT / "inputs" / "korea_gate6_data_full"
TABLES = ROOT / "tables"

VARIANT_OUTPUTS = {
    "natural_tail": "external_rolling_cv",
    "all_observations": "external_rolling_cv_all_observations",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["external_rolling_validation"]


def _rename(fit: FitResult, method_id: str) -> FitResult:
    return FitResult(method_id, fit.draws_kw, fit.fit_status, fit.threshold, fit.xi, fit.sigma, fit.cluster_rate, fit.diagnostics)


def _rng(seed_root: int, factory_index: int, origin_index: int, method_index: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed_root, 61, factory_index, origin_index, method_index, stream]))


def _candidate_fold(
    absolute_series: pd.Series,
    absolute_maxima: pd.Series,
    validation_month: str,
    threshold_quantile: float,
    run_length: int,
    draws: int,
    rng: np.random.Generator,
    endpoint_upper: float,
) -> dict[str, Any] | None:
    fit_months = [str(value) for value in absolute_maxima.index if str(value) != validation_month]
    fit_maxima = absolute_maxima.loc[fit_months]
    if fit_maxima.empty:
        return None
    normalization = float(fit_maxima.mean())
    periods = absolute_series.index.to_period("M").astype(str)
    fit_series = (absolute_series[periods.isin(fit_months)] / normalization).dropna()
    validation_maximum = float(absolute_maxima.loc[validation_month] / normalization)
    threshold = float(fit_series.quantile(threshold_quantile))
    peaks = runs_cluster_peaks(fit_series, threshold, run_length)
    excesses = peaks.to_numpy(dtype=float) - threshold
    if excesses.size < 3:
        return None
    xi, _, sigma = (float(value) for value in stats.genpareto.fit(excesses, floc=0.0))
    if not np.isfinite([xi, sigma]).all() or sigma <= 0.0:
        return None
    endpoint = threshold - sigma / xi if xi < 0.0 else None
    months = max(1, len(fit_months))
    candidate = RunsCandidate(
        threshold_quantile, threshold, run_length, xi, sigma, len(peaks), len(peaks) / months,
        float("nan"), None, None, endpoint, True, (),
    )
    predictive, _ = _monthly_reconstruction(fit_series, candidate, draws, rng, endpoint_upper)
    return {
        "validation_month": validation_month,
        "fit_month_count": len(fit_months),
        "fold_normalization_kw": normalization,
        "threshold": threshold,
        "xi": xi,
        "sigma": sigma,
        "cluster_count": len(peaks),
        "validation_log_score": -kde_negative_log_score(predictive, validation_maximum),
        "finite_endpoint_conflict": bool(endpoint is not None and (endpoint < float(peaks.max()) or endpoint > endpoint_upper)),
    }


def _fallback_cv_scores(
    absolute_series: pd.Series,
    absolute_maxima: pd.Series,
    threshold_quantile: float,
    run_length: int,
    draws: int,
    rng: np.random.Generator,
    endpoint_upper: float,
) -> tuple[float, float]:
    empirical_scores = []
    exponential_scores = []
    months_all = [str(value) for value in absolute_maxima.index]
    for validation_month in months_all:
        fit_months = [month for month in months_all if month != validation_month]
        fit_maxima = absolute_maxima.loc[fit_months]
        normalization = float(fit_maxima.mean())
        validation = float(absolute_maxima.loc[validation_month] / normalization)
        empirical = rng.choice((fit_maxima / normalization).to_numpy(dtype=float), size=draws, replace=True)
        empirical_scores.append(-kde_negative_log_score(empirical, validation))
        periods = absolute_series.index.to_period("M").astype(str)
        fit_series = (absolute_series[periods.isin(fit_months)] / normalization).dropna()
        threshold = float(fit_series.quantile(threshold_quantile))
        peaks = runs_cluster_peaks(fit_series, threshold, run_length)
        excesses = peaks.to_numpy(dtype=float) - threshold
        if excesses.size and float(np.mean(excesses)) > 0.0:
            candidate = RunsCandidate(
                threshold_quantile, threshold, run_length, 0.0, float(np.mean(excesses)), len(peaks),
                len(peaks) / max(1, len(fit_months)), float("nan"), None, None, None, False, (),
            )
            predictive, _ = _monthly_reconstruction(fit_series, candidate, draws, rng, endpoint_upper)
            exponential_scores.append(-kde_negative_log_score(predictive, validation))
    empirical_mean = float(np.mean(empirical_scores))
    exponential_mean = float(np.mean(exponential_scores)) if len(exponential_scores) == len(months_all) else float("-inf")
    return empirical_mean, exponential_mean


def fit_rolling_tail(
    absolute_series: pd.Series,
    absolute_maxima: pd.Series,
    n_draws: int,
    selection_draws: int,
    rng: np.random.Generator,
    bootstrap_rng: np.random.Generator,
    cfg: dict[str, Any],
    gated: bool,
) -> tuple[FitResult, list[dict[str, Any]]]:
    method_id = "K-GTAIL" if gated else "K-UGPD"
    months = [str(value) for value in absolute_maxima.sort_index().index]
    normalization = float(absolute_maxima.mean())
    normalized_series = absolute_series / normalization
    normalized_maxima = absolute_maxima / normalization
    candidate_rows: list[dict[str, Any]] = []
    aggregates = []
    for threshold_quantile in [float(value) for value in cfg["threshold_quantiles"]]:
        for run_length in [int(value) for value in cfg["run_lengths_minutes"]]:
            folds = []
            for validation_month in months:
                try:
                    row = _candidate_fold(
                        absolute_series, absolute_maxima, validation_month, threshold_quantile, run_length,
                        selection_draws, rng, float(cfg["endpoint_upper_normalized"]),
                    )
                except Exception:
                    row = None
                candidate_rows.append(
                    {
                        "method_id": method_id,
                        "threshold_quantile": threshold_quantile,
                        "run_length_minutes": run_length,
                        "validation_month": validation_month,
                        "fold_status": "PASS" if row is not None else "FAIL",
                        **({} if row is None else row),
                    }
                )
                if row is not None:
                    folds.append(row)
            if len(folds) == len(months):
                aggregates.append(
                    {
                        "threshold_quantile": threshold_quantile,
                        "run_length_minutes": run_length,
                        "mean_validation_log_score": float(np.mean([row["validation_log_score"] for row in folds])),
                    }
                )
    if not aggregates:
        draws = rng.choice(normalized_maxima.to_numpy(dtype=float), size=n_draws, replace=True)
        return FitResult(method_id, draws, "FALLBACK_EMPIRICAL", diagnostics={"reason": "no_complete_rolling_candidate"}), candidate_rows
    chosen = max(aggregates, key=lambda row: (row["mean_validation_log_score"], -row["threshold_quantile"], -row["run_length_minutes"]))
    q = float(chosen["threshold_quantile"])
    run_length = int(chosen["run_length_minutes"])
    final_series = normalized_series.dropna()
    threshold = float(final_series.quantile(q))
    peaks = runs_cluster_peaks(final_series, threshold, run_length)
    excesses = peaks.to_numpy(dtype=float) - threshold
    if excesses.size < 3:
        draws = rng.choice(normalized_maxima.to_numpy(dtype=float), size=n_draws, replace=True)
        return FitResult(method_id, draws, "FALLBACK_EMPIRICAL", diagnostics={"reason": "final_insufficient_clusters"}), candidate_rows
    xi, _, sigma = (float(value) for value in stats.genpareto.fit(excesses, floc=0.0))
    width = None
    pvalue = None
    successful = None
    if gated:
        width, pvalue, successful = _parametric_bootstrap(
            excesses, xi, sigma, int(cfg["formal_bootstrap_replicates"]), bootstrap_rng
        )
    endpoint = threshold - sigma / xi if xi < 0.0 else None
    reasons = []
    if gated and len(peaks) < int(cfg["minimum_clusters_gated"]):
        reasons.append("insufficient_clusters")
    if gated and (width is None or width > float(cfg["shape_ci_width_gate"])):
        reasons.append("shape_ci_width")
    if gated and (pvalue is None or not np.isfinite(pvalue) or pvalue < float(cfg["parametric_ks_pvalue_gate"])):
        reasons.append("parametric_bootstrap_ks")
    if endpoint is not None and (endpoint < float(peaks.max()) or endpoint > float(cfg["endpoint_upper_normalized"])):
        reasons.append("finite_endpoint_conflict")

    fallback_diagnostics: dict[str, Any] = {}
    status = "PASS"
    if gated and reasons:
        empirical_score, exponential_score = _fallback_cv_scores(
            absolute_series, absolute_maxima, q, run_length, selection_draws, rng,
            float(cfg["endpoint_upper_normalized"]),
        )
        fallback_diagnostics = {
            "fallback_empirical_rolling_log_score": empirical_score,
            "fallback_exponential_rolling_log_score": exponential_score,
        }
        if exponential_score >= empirical_score and len(peaks) >= max(10, int(cfg["minimum_clusters_gated"]) // 2):
            xi = 0.0
            sigma = float(np.mean(excesses))
            status = "FALLBACK_EXPONENTIAL"
        else:
            draws = rng.choice(normalized_maxima.to_numpy(dtype=float), size=n_draws, replace=True)
            return FitResult(
                method_id, draws, "FALLBACK_EMPIRICAL",
                diagnostics={"reason": "final_gate_failure", "reasons": reasons, "selection_scheme": "leave_one_training_month_out", **fallback_diagnostics},
            ), candidate_rows
    candidate = RunsCandidate(
        q, threshold, run_length, xi, sigma, len(peaks), len(peaks) / max(1, len(months)),
        float(chosen["mean_validation_log_score"]), width, pvalue, endpoint, not reasons, tuple(reasons),
    )
    draws, truncation = _monthly_reconstruction(
        final_series, candidate, n_draws, rng, float(cfg["endpoint_upper_normalized"])
    )
    fit = FitResult(
        method_id, draws, status, threshold, xi, sigma, candidate.cluster_rate,
        {
            "selection_scheme": "leave_one_training_month_out",
            "validation_months": months,
            "candidate_count": len(aggregates),
            "selected_threshold_quantile": q,
            "run_length_minutes": run_length,
            "mean_validation_log_score": chosen["mean_validation_log_score"],
            "cluster_count": len(peaks),
            "shape_ci_width": width,
            "bootstrap_ks_pvalue": pvalue,
            "bootstrap_successful": successful,
            "final_rejection_reasons": reasons,
            "truncation_fraction": truncation,
            **fallback_diagnostics,
        },
    )
    return fit, candidate_rows


def fit_block_bootstrap(
    series: pd.Series,
    target_month: str,
    n_draws: int,
    block_length: int,
    rng: np.random.Generator,
    variant: str,
) -> FitResult:
    block_maxima = []
    for _, group in series.groupby(series.index.to_period("M")):
        rolling = group.rolling(block_length, min_periods=block_length)
        maxima = rolling.max()
        adjacent = group.index.to_series().diff().eq(pd.Timedelta(minutes=15))
        consecutive = adjacent.rolling(block_length - 1).sum().eq(block_length - 1)
        maxima = maxima[(rolling.count() == block_length) & consecutive].dropna()
        block_maxima.extend(maxima.to_numpy(dtype=float))
    block_maxima_array = np.asarray(block_maxima, dtype=float)
    if block_maxima_array.size == 0:
        raise RuntimeError("no complete moving blocks")
    days = pd.Period(target_month, freq="M").days_in_month
    target_windows = int(days * 24 * 4)
    blocks_per_draw = int(math.ceil(target_windows / block_length))
    indexes = rng.integers(0, block_maxima_array.size, size=(n_draws, blocks_per_draw))
    draws = block_maxima_array[indexes].max(axis=1)
    return FitResult(
        "K-BLOCK", draws, "PASS",
        diagnostics={
            "information_set": f"fixed_nonoverlapping_15min_{variant}",
            "block_length_windows": block_length,
            "moving_blocks_available": int(block_maxima_array.size),
            "target_windows": target_windows,
            "blocks_per_draw": blocks_per_draw,
        },
    )


def pinball(observation: float, quantile: float, probability: float) -> float:
    residual = observation - quantile
    return float(probability * residual if residual >= 0.0 else (1.0 - probability) * (-residual))


def score_fit(fit: FitResult, observation: float, quantile_grid: list[float]) -> dict[str, Any]:
    draws = fit.draws_kw
    quantiles = np.quantile(draws, quantile_grid)
    q95 = float(np.quantile(draws, 0.95))
    q99 = float(np.quantile(draws, 0.99))
    return {
        "method_id": fit.method_id,
        "fit_status": fit.fit_status,
        "crps": empirical_crps(draws, observation),
        "qwcrps_090": float(
            2.0
            * np.trapezoid(
                [pinball(observation, float(q), tau) for q, tau in zip(quantiles, quantile_grid)],
                quantile_grid,
            )
        ),
        "draw_mean": float(np.mean(draws)),
        "draw_q95": q95,
        "draw_q99": q99,
        "q95_pinball": pinball(observation, q95, 0.95),
        "q99_pinball": pinball(observation, q99, 0.99),
        "q95_exceedance_brier": float(((observation > q95) - 0.05) ** 2),
        "q99_exceedance_brier": float(((observation > q99) - 0.01) ** 2),
    }


def process_unit(task: dict[str, Any]) -> dict[str, Any]:
    cfg = task["settings"]
    fixed = pd.read_parquet(INPUT / "fixed15.parquet")
    source = fixed[
        (fixed["factory"] == task["factory"])
        & (fixed["variant"] == task["variant"])
        & (fixed["measurement"] == "fixed_nonoverlapping_15min")
    ]
    absolute = source.set_index("window_timestamp")["load_kw"].sort_index()
    periods = absolute.index.to_period("M").astype(str)
    train_absolute = absolute[periods.isin(task["train_months"])]
    test_absolute = absolute[periods == task["test_month"]].dropna()
    monthly_absolute = train_absolute.groupby(train_absolute.index.to_period("M").astype(str)).max()
    normalization = float(monthly_absolute.mean())
    normalized = train_absolute / normalization
    normalized_maxima = monthly_absolute / normalization
    observation = float(test_absolute.max() / normalization)
    seed_root = int(cfg["seed_root"])
    fi = int(task["factory_index"])
    oi = int(task["origin_index"])
    n_draws = int(cfg["prediction_draws"])
    endpoint = float(cfg["endpoint_upper_normalized"])
    point = _rename(fit_point(normalized_maxima.to_numpy(dtype=float), n_draws), "K-POINT")
    empirical = _rename(fit_empirical(normalized_maxima.to_numpy(dtype=float), n_draws, _rng(seed_root, fi, oi, 1, 1)), "K-EMP")
    kde = _rename(fit_kde(normalized_maxima.to_numpy(dtype=float), n_draws, _rng(seed_root, fi, oi, 2, 1)), "K-KDE")
    gev_raw = fit_gev(normalized_maxima.to_numpy(dtype=float), n_draws, _rng(seed_root, fi, oi, 3, 1), endpoint, 0.01)
    gev = _rename(gev_raw, "K-GEV")
    ugpd, ugpd_candidates = fit_rolling_tail(
        train_absolute, monthly_absolute, n_draws, int(cfg["selection_draws"]),
        _rng(seed_root, fi, oi, 4, 1), _rng(seed_root, fi, oi, 4, 2), cfg, False,
    )
    gtail, gtail_candidates = fit_rolling_tail(
        train_absolute, monthly_absolute, n_draws, int(cfg["selection_draws"]),
        _rng(seed_root, fi, oi, 5, 1), _rng(seed_root, fi, oi, 5, 2), cfg, True,
    )
    block = fit_block_bootstrap(
        normalized,
        task["test_month"],
        n_draws,
        int(cfg["block_length_windows"]),
        _rng(seed_root, fi, oi, 6, 1),
        str(task["variant"]),
    )
    common = {
        "variant": task["variant"],
        "measurement": "fixed_nonoverlapping_15min",
        "factory": task["factory"],
        "origin_id": task["origin_id"],
        "train_months": json.dumps(task["train_months"]),
        "test_month": task["test_month"],
        "train_month_count": len(task["train_months"]),
        "normalization_kw": normalization,
        "observed_normalized_max": observation,
    }
    fits = [point, empirical, kde, gev, ugpd, gtail, block]
    scores = [{**common, **score_fit(fit, observation, [float(value) for value in cfg["quantile_score_grid"]])} for fit in fits]
    fit_rows = [{**common, **fit.as_row()} for fit in fits]
    candidate_rows = [
        {**common, **row}
        for row in [*ugpd_candidates, *gtail_candidates]
    ]
    return {"scores": scores, "fits": fit_rows, "candidates": candidate_rows}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_OUTPUTS),
        default="natural_tail",
        help="External reconstruction variant. Outputs are kept in separate folders.",
    )
    args = parser.parse_args()
    variant = str(args.variant)
    output = ROOT / "outputs" / VARIANT_OUTPUTS[variant]
    cfg = settings()
    eligibility = pd.read_csv(INPUT / "monthly_eligibility.csv", dtype={"month": str})
    fixed = pd.read_parquet(INPUT / "fixed15.parquet", columns=["factory", "variant", "measurement"])
    factories = sorted(
        fixed.loc[
            (fixed["variant"] == variant)
            & (fixed["measurement"] == "fixed_nonoverlapping_15min"),
            "factory",
        ].unique()
    )
    tasks = []
    excluded = []
    for factory_index, factory in enumerate(factories):
        eligible = eligibility[
            (eligibility["factory"] == factory)
            & (eligibility["variant"] == variant)
            & (eligibility["measurement"] == "fixed_nonoverlapping_15min")
        ].set_index("month")
        for origin_index, origin in enumerate(cfg["rolling_origins"]):
            required = [*origin["train_months"], origin["test_month"]]
            passed = all(month in eligible.index and bool(eligible.loc[month, "eligible_95"]) for month in required)
            row = {
                "settings": cfg,
                "factory": factory,
                "variant": variant,
                "factory_index": factory_index,
                "origin_index": origin_index,
                **origin,
            }
            (tasks if passed else excluded).append(row)
    started = time.perf_counter()
    outputs = []
    with ProcessPoolExecutor(max_workers=int(cfg["workers"])) as executor:
        futures = {executor.submit(process_unit, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            outputs.append(result)
            print(
                f"[external-rolling:{variant}] {task['factory']} {task['origin_id']} PASS",
                flush=True,
            )
    scores = pd.DataFrame([row for result in outputs for row in result["scores"]])
    fits = pd.DataFrame([row for result in outputs for row in result["fits"]])
    candidates = pd.DataFrame([row for result in outputs for row in result["candidates"]])
    scores = scores.sort_values(["factory", "origin_id", "method_id"]).reset_index(drop=True)
    fits = fits.sort_values(["factory", "origin_id", "method_id"]).reset_index(drop=True)
    candidates = candidates.sort_values(["factory", "origin_id", "method_id", "threshold_quantile", "run_length_minutes", "validation_month"]).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    scores_path = output / "prediction_scores.csv"
    fits_path = output / "fit_diagnostics.csv"
    candidates_path = output / "candidate_rolling_cv.csv"
    atomic_csv(scores, scores_path)
    atomic_csv(fits, fits_path)
    atomic_csv(candidates, candidates_path)
    table_suffix = "" if variant == "natural_tail" else "_all_observations"
    atomic_csv(scores, TABLES / f"external_rolling_tail_scores{table_suffix}.csv")

    metrics = ["crps", "qwcrps_090", "q95_pinball", "q99_pinball", "q95_exceedance_brier", "q99_exceedance_brier"]
    factory = scores.groupby(["factory", "method_id"], as_index=False)[metrics].mean()
    rng = np.random.default_rng(int(cfg["seed_root"]) + 9000)
    contrast_rows = []
    for metric in metrics:
        pivot = factory.pivot(index="factory", columns="method_id", values=metric)
        for baseline in sorted(value for value in scores["method_id"].unique() if value != "K-GTAIL"):
            difference = (pivot[baseline] - pivot["K-GTAIL"]).dropna().to_numpy(dtype=float)
            indexes = rng.integers(0, difference.size, size=(int(cfg["factory_bootstrap_replicates"]), difference.size))
            boot = difference[indexes].mean(axis=1)
            contrast_rows.append(
                {
                    "metric": metric,
                    "contrast": f"{baseline}-minus-K-GTAIL",
                    "positive_favors_K_GTAIL": True,
                    "factory_mean_gain": float(np.mean(difference)),
                    "factory_cluster_ci95_low": float(np.quantile(boot, 0.025)),
                    "factory_cluster_ci95_high": float(np.quantile(boot, 0.975)),
                    "factories": int(difference.size),
                    "bootstrap_replicates": int(cfg["factory_bootstrap_replicates"]),
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrast_path = output / "factory_cluster_contrasts.csv"
    atomic_csv(contrasts, contrast_path)
    atomic_csv(
        contrasts,
        TABLES / f"external_rolling_factory_contrasts{table_suffix}.csv",
    )

    validation_months = set(candidates["validation_month"].dropna().astype(str))
    test_months = set(scores["test_month"].astype(str))
    per_row_leakage = candidates.apply(lambda row: str(row["validation_month"]) == str(row["test_month"]), axis=1)
    expected_score_rows = len(tasks) * 7
    checks = {
        "eligible_units_present": len(tasks) > 0,
        "expected_score_rows": len(scores) == expected_score_rows,
        "unique_score_keys": not scores.duplicated(["factory", "origin_id", "method_id"]).any(),
        "all_scores_finite": bool(np.isfinite(scores[metrics].to_numpy(dtype=float)).all()),
        "no_test_month_used_for_candidate_validation": not bool(per_row_leakage.any()),
        "rolling_validation_has_all_training_months": bool((candidates.groupby(["factory", "origin_id", "method_id", "threshold_quantile", "run_length_minutes"])["validation_month"].nunique() == candidates.groupby(["factory", "origin_id", "method_id", "threshold_quantile", "run_length_minutes"])["train_month_count"].first()).all()),
        "same_information_non_evt_comparator_present": "K-BLOCK" in set(scores["method_id"]),
        "tail_weighted_score_present": "qwcrps_090" in scores.columns,
    }
    signature_paths = [Path(__file__), CONFIG, INPUT / "fixed15.parquet", INPUT / "monthly_eligibility.csv"]
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "task_id": f"EXTERNAL_ROLLING_{variant.upper()}_V1",
        "variant": variant,
        "measurement": "fixed_nonoverlapping_15min",
        "signature": hashlib.sha256(
            (variant + "".join(sha256(path) for path in signature_paths)).encode("ascii")
        ).hexdigest(),
        "eligible_units": len(tasks),
        "excluded_units": len(excluded),
        "score_rows": int(len(scores)),
        "fit_rows": int(len(fits)),
        "candidate_fold_rows": int(len(candidates)),
        "methods": sorted(scores["method_id"].unique()),
        "checks": checks,
        "duration_seconds": time.perf_counter() - started,
        "scores_sha256": sha256(scores_path),
        "fits_sha256": sha256(fits_path),
        "candidates_sha256": sha256(candidates_path),
        "contrasts_sha256": sha256(contrast_path),
        "validation_month_values": sorted(validation_months),
        "test_month_values": sorted(test_months),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(f"external rolling validation failed: {checks}")


if __name__ == "__main__":
    main()
