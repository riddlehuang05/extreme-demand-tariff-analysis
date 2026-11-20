"""Label-free runs-POT estimators for the Korean external module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from track_a.estimators.base import FitResult
from track_a.external.scoring import kde_negative_log_score


@dataclass(frozen=True)
class RunsCandidate:
    threshold_quantile: float
    threshold: float
    run_length_minutes: int
    xi: float
    sigma: float
    cluster_count: int
    cluster_rate: float
    validation_log_score: float
    shape_ci_width: float | None
    bootstrap_ks_pvalue: float | None
    endpoint: float | None
    valid_gate: bool
    rejection_reasons: tuple[str, ...]


def runs_cluster_peaks(series: pd.Series, threshold: float, run_length_minutes: int) -> pd.Series:
    exceed = series.dropna()
    exceed = exceed[exceed > threshold].sort_index()
    if exceed.empty:
        return pd.Series(dtype=float, name="peak_value")
    gaps = exceed.index.to_series().diff()
    groups = (gaps.isna() | (gaps > pd.Timedelta(minutes=run_length_minutes))).cumsum()
    peaks = exceed.groupby(groups.to_numpy()).max()
    peak_times = exceed.groupby(groups.to_numpy()).apply(lambda values: values.idxmax())
    return pd.Series(peaks.to_numpy(dtype=float), index=pd.DatetimeIndex(peak_times), name="peak_value").sort_index()


def _parametric_bootstrap(
    excesses: np.ndarray,
    xi: float,
    sigma: float,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    observed = float(stats.kstest(excesses, "genpareto", args=(xi, 0.0, sigma)).statistic)
    shapes = []
    statistics = []
    for _ in range(replicates):
        sample = stats.genpareto.rvs(xi, loc=0.0, scale=sigma, size=excesses.size, random_state=rng)
        try:
            boot_xi, _, boot_sigma = stats.genpareto.fit(sample, floc=0.0)
            if not np.isfinite([boot_xi, boot_sigma]).all() or boot_sigma <= 0.0:
                continue
            statistic = stats.kstest(sample, "genpareto", args=(boot_xi, 0.0, boot_sigma)).statistic
            shapes.append(float(boot_xi))
            statistics.append(float(statistic))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    if len(shapes) < max(10, int(0.8 * replicates)):
        return float("inf"), float("nan"), len(shapes)
    low, high = np.quantile(shapes, [0.025, 0.975])
    pvalue = (1.0 + sum(value >= observed for value in statistics)) / (1.0 + len(statistics))
    return float(high - low), float(pvalue), len(shapes)


def _monthly_reconstruction(
    series: pd.Series,
    candidate: RunsCandidate,
    n_draws: int,
    rng: np.random.Generator,
    endpoint_upper: float,
) -> tuple[np.ndarray, float]:
    months = series.index.to_period("M")
    below = series.where(series <= candidate.threshold).groupby(months).max().dropna().to_numpy(dtype=float)
    if below.size == 0:
        below = np.array([0.0])
    baseline = rng.choice(below, size=n_draws, replace=True)
    counts = rng.poisson(candidate.cluster_rate, size=n_draws)
    tail = np.zeros(n_draws)
    for index in np.flatnonzero(counts):
        severity = stats.genpareto.rvs(
            candidate.xi,
            loc=0.0,
            scale=candidate.sigma,
            size=int(counts[index]),
            random_state=rng,
        )
        tail[index] = candidate.threshold + float(np.max(severity))
    raw = np.maximum(baseline, tail)
    truncation = float(np.mean(raw > endpoint_upper))
    return np.clip(raw, 0.0, endpoint_upper), truncation


def _candidate(
    fit_series: pd.Series,
    validation_maximum: float,
    threshold_quantile: float,
    run_length_minutes: int,
    screening_bootstrap: int,
    minimum_clusters: int,
    shape_width_gate: float,
    ks_pvalue_gate: float,
    endpoint_upper: float,
    rng: np.random.Generator,
    screening_bootstrap_rng: np.random.Generator,
) -> RunsCandidate | None:
    threshold = float(fit_series.quantile(threshold_quantile))
    peaks = runs_cluster_peaks(fit_series, threshold, run_length_minutes)
    excesses = peaks.to_numpy(dtype=float) - threshold
    if excesses.size < 3:
        return None
    xi, _, sigma = (float(value) for value in stats.genpareto.fit(excesses, floc=0.0))
    endpoint = threshold - sigma / xi if xi < 0.0 else None
    width, pvalue, _ = _parametric_bootstrap(
        excesses, xi, sigma, screening_bootstrap, screening_bootstrap_rng
    )
    months = max(1, fit_series.index.to_period("M").nunique())
    provisional = RunsCandidate(
        threshold_quantile, threshold, run_length_minutes, xi, sigma, len(peaks),
        len(peaks) / months, float("nan"), width, pvalue, endpoint, False, (),
    )
    draws, _ = _monthly_reconstruction(fit_series, provisional, 2000, rng, endpoint_upper)
    score = -kde_negative_log_score(draws, validation_maximum)
    reasons = []
    if len(peaks) < minimum_clusters:
        reasons.append("insufficient_clusters")
    if width > shape_width_gate:
        reasons.append("shape_ci_width")
    if not np.isfinite(pvalue) or pvalue < ks_pvalue_gate:
        reasons.append("parametric_bootstrap_ks")
    if endpoint is not None and (endpoint < peaks.max() or endpoint > endpoint_upper):
        reasons.append("finite_endpoint_conflict")
    return RunsCandidate(
        threshold_quantile, threshold, run_length_minutes, xi, sigma, len(peaks),
        len(peaks) / months, score, width, pvalue, endpoint, not reasons, tuple(reasons),
    )


def _gate_failure_fallback_choice(
    fit_series: pd.Series,
    fit_monthly_maxima: pd.Series,
    validation_maximum: float,
    chosen: RunsCandidate,
    endpoint_upper: float,
    rng: np.random.Generator,
) -> tuple[str, dict[str, float | None]]:
    """Choose a failed-gate fallback using the training validation month only."""
    empirical_draws = rng.choice(
        fit_monthly_maxima.to_numpy(dtype=float), size=2000, replace=True
    )
    empirical_score = -kde_negative_log_score(empirical_draws, validation_maximum)
    peaks = runs_cluster_peaks(fit_series, chosen.threshold, chosen.run_length_minutes)
    excesses = peaks.to_numpy(dtype=float) - chosen.threshold
    exponential_score: float | None = None
    if excesses.size and np.isfinite(excesses).all() and float(excesses.mean()) > 0.0:
        exponential = RunsCandidate(
            chosen.threshold_quantile,
            chosen.threshold,
            chosen.run_length_minutes,
            0.0,
            float(excesses.mean()),
            len(peaks),
            chosen.cluster_rate,
            float("nan"),
            None,
            None,
            None,
            False,
            ("validation_only_fallback",),
        )
        exponential_draws, _ = _monthly_reconstruction(
            fit_series, exponential, 2000, rng, endpoint_upper
        )
        exponential_score = -kde_negative_log_score(exponential_draws, validation_maximum)
    selected = (
        "FALLBACK_EXPONENTIAL"
        if exponential_score is not None and exponential_score >= empirical_score
        else "FALLBACK_EMPIRICAL"
    )
    return selected, {
        "fallback_empirical_validation_log_score": float(empirical_score),
        "fallback_exponential_validation_log_score": (
            None if exponential_score is None else float(exponential_score)
        ),
    }


def fit_runs_tail(
    training_series: pd.Series,
    training_monthly_maxima: pd.Series,
    n_draws: int,
    rng: np.random.Generator,
    bootstrap_rng: np.random.Generator,
    threshold_quantiles: list[float],
    run_lengths_minutes: list[int],
    screening_bootstrap: int,
    formal_bootstrap: int,
    minimum_clusters: int,
    shape_width_gate: float,
    ks_pvalue_gate: float,
    endpoint_upper: float,
    gated: bool,
) -> FitResult:
    method_id = "K-GTAIL" if gated else "K-UGPD"
    maxima = training_monthly_maxima.sort_index()
    if len(maxima) < 2:
        draws = rng.choice(maxima.to_numpy(dtype=float), size=n_draws, replace=True)
        return FitResult(method_id, draws, "FALLBACK_EMPIRICAL", diagnostics={"reason": "fewer_than_two_training_months"})
    validation_month = maxima.index[-1]
    fit_months = maxima.index[:-1]
    fit_series = training_series[training_series.index.to_period("M").astype(str).isin(fit_months)].dropna()
    validation_maximum = float(maxima.iloc[-1])
    candidates = []
    for threshold_quantile in threshold_quantiles:
        for run_length in run_lengths_minutes:
            try:
                candidate = _candidate(
                    fit_series, validation_maximum, threshold_quantile, run_length,
                    screening_bootstrap if gated else 0, minimum_clusters, shape_width_gate,
                    ks_pvalue_gate, endpoint_upper, rng, bootstrap_rng,
                )
                if candidate is not None:
                    candidates.append(candidate)
            except Exception:
                continue
    eligible = [candidate for candidate in candidates if candidate.valid_gate] if gated else candidates
    if not eligible:
        draws = rng.choice(maxima.to_numpy(dtype=float), size=n_draws, replace=True)
        return FitResult(
            method_id,
            draws,
            "FALLBACK_EMPIRICAL",
            diagnostics={"reason": "no_candidate_passed", "candidate_diagnostics": [candidate.__dict__ for candidate in candidates]},
        )
    chosen = max(eligible, key=lambda candidate: (candidate.validation_log_score, -candidate.threshold_quantile, -candidate.run_length_minutes))
    final_threshold = float(training_series.dropna().quantile(chosen.threshold_quantile))
    final_peaks = runs_cluster_peaks(training_series, final_threshold, chosen.run_length_minutes)
    excesses = final_peaks.to_numpy(dtype=float) - final_threshold
    xi, _, sigma = (float(value) for value in stats.genpareto.fit(excesses, floc=0.0))
    width, pvalue, successful = _parametric_bootstrap(
        excesses,
        xi,
        sigma,
        formal_bootstrap if gated else 0,
        bootstrap_rng,
    )
    endpoint = final_threshold - sigma / xi if xi < 0.0 else None
    final_reasons = []
    if gated and len(final_peaks) < minimum_clusters:
        final_reasons.append("insufficient_clusters")
    if gated and width > shape_width_gate:
        final_reasons.append("shape_ci_width")
    if gated and (not np.isfinite(pvalue) or pvalue < ks_pvalue_gate):
        final_reasons.append("parametric_bootstrap_ks")
    if endpoint is not None and (endpoint < final_peaks.max() or endpoint > endpoint_upper):
        final_reasons.append("finite_endpoint_conflict")
    months = max(1, training_series.index.to_period("M").nunique())
    final = RunsCandidate(
        chosen.threshold_quantile, final_threshold, chosen.run_length_minutes, xi, sigma,
        len(final_peaks), len(final_peaks) / months, chosen.validation_log_score,
        width, pvalue, endpoint, not final_reasons, tuple(final_reasons),
    )
    if gated and final_reasons:
        fallback_status, fallback_diagnostics = _gate_failure_fallback_choice(
            fit_series,
            maxima.iloc[:-1],
            validation_maximum,
            chosen,
            endpoint_upper,
            rng,
        )
        if len(final_peaks) >= max(10, minimum_clusters // 2):
            if fallback_status == "FALLBACK_EXPONENTIAL":
                final = RunsCandidate(**{**final.__dict__, "xi": 0.0, "sigma": float(excesses.mean())})
                status = fallback_status
            else:
                draws = rng.choice(maxima.to_numpy(dtype=float), size=n_draws, replace=True)
                return FitResult(
                    method_id,
                    draws,
                    fallback_status,
                    diagnostics={
                        "reason": "final_gate_failure",
                        "reasons": final_reasons,
                        **fallback_diagnostics,
                    },
                )
        else:
            draws = rng.choice(maxima.to_numpy(dtype=float), size=n_draws, replace=True)
            return FitResult(
                method_id,
                draws,
                "FALLBACK_EMPIRICAL",
                diagnostics={
                    "reason": "final_gate_failure_insufficient_clusters",
                    "reasons": final_reasons,
                    **fallback_diagnostics,
                },
            )
    else:
        status = "PASS"
    draws, truncation = _monthly_reconstruction(training_series, final, n_draws, rng, endpoint_upper)
    return FitResult(
        method_id,
        draws,
        status,
        threshold=final.threshold,
        xi=final.xi,
        sigma=final.sigma,
        cluster_rate=final.cluster_rate,
        diagnostics={
            "run_length_minutes": final.run_length_minutes,
            "cluster_count": final.cluster_count,
            "shape_ci_width": width,
            "bootstrap_ks_pvalue": pvalue,
            "bootstrap_successful": successful,
            "endpoint": endpoint,
            "truncation_fraction": truncation,
            "final_rejection_reasons": final_reasons,
            **(fallback_diagnostics if gated and final_reasons else {}),
            "candidate_diagnostics": [candidate.__dict__ for candidate in candidates],
        },
    )
