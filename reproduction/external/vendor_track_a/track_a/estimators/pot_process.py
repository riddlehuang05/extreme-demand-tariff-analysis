"""Process-aware POT estimator with validation gates and explicit fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .base import FitResult


@dataclass(frozen=True)
class _Candidate:
    quantile: float
    threshold: float
    xi: float
    sigma: float
    exceedances: int
    shape_ci_width: float
    endpoint_kw: float | None
    ks_pvalue: float
    count_model: str
    count_mean: float
    count_variance: float
    nb_size: float | None
    nb_probability: float | None
    validation_log_score: float
    valid_main: bool
    rejection_reasons: tuple[str, ...]


def _process_events(clusters: pd.DataFrame, decluster: bool) -> pd.DataFrame:
    required = {"month_id", "heat_id", "cluster_max_kw"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"cluster table missing required columns: {sorted(missing)}")
    frame = clusters.copy()
    frame["month_id"] = frame["month_id"].astype(int)
    if not decluster or "rare_cluster_id" not in frame or "rare_event_flag" not in frame:
        return frame[["month_id", "heat_id", "cluster_max_kw"]].rename(columns={"heat_id": "event_id"})
    rare_id = frame["rare_cluster_id"]
    is_rare = frame["rare_event_flag"].astype(bool) & rare_id.notna()
    frame["event_id"] = np.where(
        is_rare,
        "R" + frame["month_id"].astype(str) + ":" + rare_id.astype(str),
        "H" + frame["month_id"].astype(str) + ":" + frame["heat_id"].astype(str),
    )
    return (
        frame.groupby(["month_id", "event_id"], as_index=False)["cluster_max_kw"]
        .max()
        .sort_values(["month_id", "event_id"])
    )


def _count_parameters(counts: np.ndarray) -> tuple[str, float, float, float | None, float | None]:
    mean = float(np.mean(counts))
    variance = float(np.var(counts, ddof=1)) if counts.size > 1 else mean
    if mean <= 0.0 or variance / mean <= 1.2:
        return "poisson", mean, variance, None, None
    size = mean * mean / max(variance - mean, np.finfo(float).eps)
    probability = size / (size + mean)
    return "negative_binomial", mean, variance, float(size), float(probability)


def _draw_counts(candidate: _Candidate, n: int, rng: np.random.Generator) -> np.ndarray:
    if candidate.count_model == "poisson":
        return rng.poisson(candidate.count_mean, size=n)
    if candidate.nb_size is None or candidate.nb_probability is None:
        raise RuntimeError("negative-binomial candidate is missing parameters")
    return rng.negative_binomial(candidate.nb_size, candidate.nb_probability, size=n)


def _subthreshold_maxima(events: pd.DataFrame, months: np.ndarray, threshold: float) -> np.ndarray:
    below = events[events["cluster_max_kw"] <= threshold]
    by_month = below.groupby("month_id")["cluster_max_kw"].max()
    values = by_month.reindex(months).fillna(0.0).to_numpy(dtype=float)
    return values if values.size else np.array([0.0])


def _reconstruct(
    candidate: _Candidate,
    subthreshold: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    physical_bound_kw: float,
) -> tuple[np.ndarray, float]:
    baseline = rng.choice(subthreshold, size=n_draws, replace=True)
    counts = _draw_counts(candidate, n_draws, rng)
    tail_max = np.zeros(n_draws, dtype=float)
    positive = np.flatnonzero(counts > 0)
    for index in positive:
        severities = stats.genpareto.rvs(
            candidate.xi,
            loc=0.0,
            scale=candidate.sigma,
            size=int(counts[index]),
            random_state=rng,
        )
        tail_max[index] = candidate.threshold + float(np.max(severities))
    raw = np.maximum(baseline, tail_max)
    truncation = float(np.mean(raw > physical_bound_kw))
    return np.clip(raw, 0.0, physical_bound_kw), truncation


def _shape_ci_width(excesses: np.ndarray, replicates: int, rng: np.random.Generator) -> float:
    estimates = []
    for _ in range(replicates):
        sample = rng.choice(excesses, size=excesses.size, replace=True)
        try:
            xi, _, sigma = stats.genpareto.fit(sample, floc=0.0)
            if np.isfinite(xi) and np.isfinite(sigma) and sigma > 0.0:
                estimates.append(float(xi))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    if len(estimates) < max(10, int(0.8 * replicates)):
        return float("inf")
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(high - low)


def _log_score(draws: np.ndarray, validation: np.ndarray) -> float:
    if validation.size == 0 or np.isclose(draws.std(), 0.0):
        return float("nan")
    kde = stats.gaussian_kde(draws)
    density = np.maximum(kde.evaluate(validation), np.finfo(float).tiny)
    return float(np.mean(np.log(density)))


def _fit_candidate(
    quantile: float,
    events: pd.DataFrame,
    fit_months: np.ndarray,
    validation_maxima: np.ndarray,
    bootstrap_replicates: int,
    validation_draws: int,
    minimum_exceedances_main: int,
    shape_ci_width_gate: float,
    physical_bound_kw: float,
    rng: np.random.Generator,
    bootstrap_rng: np.random.Generator,
) -> tuple[_Candidate, np.ndarray]:
    values = events["cluster_max_kw"].to_numpy(dtype=float)
    threshold = float(np.quantile(values, quantile))
    excesses = values[values > threshold] - threshold
    if excesses.size == 0:
        raise ValueError("threshold has no exceedances")
    xi, _, sigma = (float(value) for value in stats.genpareto.fit(excesses, floc=0.0))
    transformed = stats.genpareto.cdf(excesses, xi, loc=0.0, scale=sigma)
    ks_pvalue = float(stats.kstest(transformed, "uniform").pvalue)
    ci_width = _shape_ci_width(excesses, bootstrap_replicates, bootstrap_rng)
    endpoint = threshold - sigma / xi if xi < 0.0 else None
    counts = (
        events.assign(exceed=events["cluster_max_kw"] > threshold)
        .groupby("month_id")["exceed"]
        .sum()
        .reindex(fit_months, fill_value=0)
        .to_numpy(dtype=int)
    )
    count_model, count_mean, count_variance, nb_size, nb_probability = _count_parameters(counts)
    reasons = []
    if excesses.size < minimum_exceedances_main:
        reasons.append(f"fewer_than_{minimum_exceedances_main}_exceedances")
    if ci_width > shape_ci_width_gate:
        reasons.append("shape_ci_width")
    if endpoint is not None and (endpoint < values.max() - 1e-9 or endpoint > physical_bound_kw):
        reasons.append("finite_endpoint_physical_conflict")
    if ks_pvalue < 0.01:
        reasons.append("gpd_uniform_transform_ks")
    provisional = _Candidate(
        quantile, threshold, xi, sigma, int(excesses.size), ci_width, endpoint, ks_pvalue,
        count_model, count_mean, count_variance, nb_size, nb_probability,
        float("nan"), not reasons, tuple(reasons),
    )
    subthreshold = _subthreshold_maxima(events, fit_months, threshold)
    validation_sample, _ = _reconstruct(
        provisional, subthreshold, validation_draws, rng, physical_bound_kw
    )
    score = _log_score(validation_sample, validation_maxima)
    candidate = _Candidate(**{**provisional.__dict__, "validation_log_score": score})
    return candidate, subthreshold


def fit_process_tail(
    monthly_maxima: pd.DataFrame,
    heat_clusters: pd.DataFrame,
    n_draws: int,
    rng: np.random.Generator,
    bootstrap_rng: np.random.Generator,
    threshold_quantiles: list[float],
    minimum_exceedances_main: int,
    minimum_exceedances_fallback: int,
    bootstrap_replicates: int,
    holdout_fraction: float,
    shape_ci_width_gate: float,
    physical_bound_kw: float,
    truncation_fraction_gate: float,
    validation_draws: int,
    stability_penalty_weight: float,
    decluster: bool = True,
) -> FitResult:
    method_id = "TAIL-JOINT" if decluster else "TAIL-NODECLUSTER"
    month_frame = monthly_maxima[["month_id", "M_fixed15_kw"]].drop_duplicates("month_id").sort_values("month_id")
    if month_frame.empty:
        raise ValueError("monthly_maxima cannot be empty")
    all_months = month_frame["month_id"].to_numpy(dtype=int)
    split = max(1, min(all_months.size - 1, int(np.floor(all_months.size * (1.0 - holdout_fraction)))))
    if all_months.size < 2:
        draws = np.full(n_draws, float(month_frame["M_fixed15_kw"].iloc[0]))
        return FitResult(method_id, draws, "FALLBACK_EMPIRICAL", diagnostics={"reason": "fewer_than_two_months"})
    fit_months, validation_months = all_months[:split], all_months[split:]
    process = _process_events(heat_clusters, decluster)
    fit_events = process[process["month_id"].isin(fit_months)]
    validation_maxima = month_frame[month_frame["month_id"].isin(validation_months)]["M_fixed15_kw"].to_numpy(dtype=float)
    if fit_events.empty:
        draws = rng.choice(month_frame["M_fixed15_kw"].to_numpy(dtype=float), size=n_draws, replace=True)
        return FitResult(method_id, draws, "FALLBACK_EMPIRICAL", diagnostics={"reason": "no_fit_events"})

    candidates: list[_Candidate] = []
    subthreshold_by_quantile: dict[float, np.ndarray] = {}
    for quantile in threshold_quantiles:
        try:
            candidate, subthreshold = _fit_candidate(
                quantile, fit_events, fit_months, validation_maxima,
                bootstrap_replicates, validation_draws, minimum_exceedances_main, shape_ci_width_gate,
                physical_bound_kw, rng, bootstrap_rng,
            )
            candidates.append(candidate)
            subthreshold_by_quantile[quantile] = subthreshold
        except Exception:
            continue

    main = [candidate for candidate in candidates if candidate.valid_main and np.isfinite(candidate.validation_log_score)]
    fit_status = "PASS"
    fallback_reason = None
    if main:
        median_xi = float(np.median([candidate.xi for candidate in main]))
        chosen = max(
            main,
            key=lambda candidate: (
                candidate.validation_log_score
                - stability_penalty_weight * abs(candidate.xi - median_xi),
                -candidate.quantile,
            ),
        )
    else:
        fallback_candidates = [
            candidate
            for candidate in candidates
            if candidate.exceedances >= minimum_exceedances_fallback
        ]
        if fallback_candidates:
            base = max(fallback_candidates, key=lambda candidate: (candidate.exceedances, -candidate.quantile))
            excesses = fit_events.loc[fit_events["cluster_max_kw"] > base.threshold, "cluster_max_kw"].to_numpy(dtype=float) - base.threshold
            chosen = _Candidate(
                **{
                    **base.__dict__,
                    "xi": 0.0,
                    "sigma": float(excesses.mean()),
                    "endpoint_kw": None,
                    "valid_main": False,
                }
            )
            fit_status = "FALLBACK_EXPONENTIAL"
            fallback_reason = "no_threshold_passed_main_diagnostics"
        else:
            source = month_frame[month_frame["month_id"].isin(fit_months)]["M_fixed15_kw"].to_numpy(dtype=float)
            draws = rng.choice(source, size=n_draws, replace=True)
            return FitResult(
                method_id,
                draws,
                "FALLBACK_EMPIRICAL",
                diagnostics={
                    "reason": f"fewer_than_{minimum_exceedances_fallback}_exceedances_or_fit_failure",
                    "process_event_count": int(len(fit_events)),
                    "candidate_count": len(candidates),
                },
            )
    subthreshold = _subthreshold_maxima(fit_events, fit_months, chosen.threshold)
    draws, truncation = _reconstruct(chosen, subthreshold, n_draws, rng, physical_bound_kw)
    if truncation > truncation_fraction_gate:
        fit_status = "FAIL_TRUNCATION_GATE"
    diagnostics: dict[str, Any] = {
        "decluster": decluster,
        "fit_months": int(fit_months.size),
        "validation_months": int(validation_months.size),
        "process_event_count": int(len(fit_events)),
        "count_model": chosen.count_model,
        "count_variance": chosen.count_variance,
        "shape_ci_width": chosen.shape_ci_width,
        "ks_pvalue": chosen.ks_pvalue,
        "endpoint_kw": chosen.endpoint_kw,
        "validation_log_score": chosen.validation_log_score,
        "truncation_fraction": truncation,
        "fallback_reason": fallback_reason,
        "candidate_diagnostics": [candidate.__dict__ for candidate in candidates],
    }
    return FitResult(
        method_id,
        draws,
        fit_status,
        threshold=chosen.threshold,
        xi=chosen.xi,
        sigma=chosen.sigma,
        cluster_rate=chosen.count_mean,
        diagnostics=diagnostics,
    )
