"""Four-fold, month-level CRPS threshold selection for the primary TAIL law.

The score is deterministic conditional on each fitted fold.  This module does
not use test-month outcomes for fitting and does not use a nominal KS p-value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate, stats


@dataclass(frozen=True)
class FoldFit:
    quantile: float
    fold: int
    fit_months: int
    validation_months: int
    threshold: float
    xi: float
    sigma: float
    excesses: int
    count_model: str
    count_mean: float
    crps: float
    status: str


def _count_parameters(counts: np.ndarray) -> tuple[str, float, float | None, float | None]:
    mean = float(np.mean(counts))
    variance = float(np.var(counts, ddof=1)) if len(counts) > 1 else mean
    if mean <= 0 or variance / mean <= 1.2:
        return "poisson", mean, None, None
    size = mean * mean / (variance - mean)
    return "negative_binomial", mean, float(size), float(size / (size + mean))


def _tail_cdf(x: float, threshold: float, xi: float, sigma: float,
              count_model: str, rate: float, nb_size: float | None,
              nb_probability: float | None) -> float:
    if x < threshold:
        survival = 1.0
    else:
        z = (x - threshold) / sigma
        support = 1.0 + xi * z
        survival = (0.0 if support <= 0 else
                    math.exp(-z) if abs(xi) < 1e-10 else
                    support ** (-1.0 / xi))
    if count_model == "poisson":
        return math.exp(-rate * survival)
    assert nb_size is not None and nb_probability is not None
    odds = (1.0 - nb_probability) / nb_probability
    return math.exp(-nb_size * math.log1p(odds * survival))


def _crps_one(y: float, threshold: float, xi: float, sigma: float,
              baseline: np.ndarray, count_model: str, rate: float,
              nb_size: float | None, nb_probability: float | None) -> float:
    """Integrate the empirical subthreshold CDF exactly and the tail adaptively."""
    if y < 0:
        raise ValueError("monthly maximum must be nonnegative")
    p_zero = _tail_cdf(0, threshold, xi, sigma, count_model, rate,
                       nb_size, nb_probability)
    edges = np.unique(np.r_[0.0, baseline, threshold, y])
    edges = edges[(edges >= 0) & (edges <= threshold)]
    below = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mid = (left + right) / 2.0
        cdf = p_zero * float(np.mean(baseline <= mid))
        below += (right - left) * (cdf * cdf if mid < y else (1 - cdf) ** 2)
    if y < threshold and (edges.size == 0 or edges[-1] < threshold):
        below += (threshold - max(y, edges[-1] if edges.size else 0)) * (1 - p_zero) ** 2

    def integrand(x: float, left_side: bool) -> float:
        cdf = _tail_cdf(x, threshold, xi, sigma, count_model, rate,
                        nb_size, nb_probability)
        return cdf * cdf if left_side else (1 - cdf) ** 2

    above = 0.0
    if y > threshold:
        above += integrate.quad(lambda x: integrand(x, True), threshold, y,
                                epsabs=1e-5, epsrel=1e-7, limit=100)[0]
    lower = max(threshold, y)
    endpoint = threshold - sigma / xi if xi < 0 else np.inf
    if lower < endpoint:
        surv_lower = (1.0 if lower == threshold else
                      max(0.0, 1.0 + xi * (lower - threshold) / sigma) ** (-1.0 / xi)
                      if abs(xi) > 1e-10 else
                      math.exp(-(lower - threshold) / sigma))
        def transformed_tail(survival: float) -> float:
            if survival == 0.0:
                return 0.0
            if count_model == "poisson":
                complement = -math.expm1(-rate * survival)
            else:
                assert nb_size is not None and nb_probability is not None
                odds = (1.0 - nb_probability) / nb_probability
                complement = -math.expm1(-nb_size * math.log1p(odds * survival))
            dx_ds = (sigma / survival if abs(xi) < 1e-10 else
                     sigma * survival ** (-xi - 1.0))
            return complement * complement * dx_ds
        above += integrate.quad(transformed_tail, 0.0, surv_lower,
                                epsabs=1e-5, epsrel=2e-7, limit=200)[0]
    return float(below + above)


def select_threshold(monthly: pd.DataFrame, events: pd.DataFrame,
                     quantiles: tuple[float, ...] = (0.9, 0.925, 0.95, 0.975),
                     minimum_exceedances: int = 100) -> tuple[float, list[FoldFit]]:
    """Choose the lowest-CRPS quantile from four disjoint six-month folds."""
    months = monthly[["month_id", "M_fixed15_kw"]].drop_duplicates("month_id").sort_values("month_id")
    if len(months) != 24 or months.month_id.nunique() != 24:
        raise ValueError("primary selection requires 24 distinct training months")
    month_ids = months.month_id.to_numpy(dtype=int)
    outcomes = months.set_index("month_id").M_fixed15_kw
    if "rare_cluster_id" in events and "rare_event_flag" in events:
        frame = events.copy()
        rare = frame.rare_event_flag.astype(bool) & frame.rare_cluster_id.notna()
        frame["event_id"] = np.where(
            rare,
            "R" + frame.month_id.astype(str) + ":" + frame.rare_cluster_id.astype(str),
            "H" + frame.month_id.astype(str) + ":" + frame.heat_id.astype(str),
        )
        events = frame.groupby(["month_id", "event_id"], as_index=False).cluster_max_kw.max()
    if set(events.month_id.unique()) - set(month_ids):
        raise ValueError("event month outside the training history")
    folds = np.array_split(month_ids, 4)
    records: list[FoldFit] = []
    for fold_index, held_out in enumerate(folds):
        fit_months = month_ids[~np.isin(month_ids, held_out)]
        fit_events = events.loc[events.month_id.isin(fit_months)]
        values = fit_events.cluster_max_kw.to_numpy(dtype=float)
        held_out_maxima = outcomes.loc[held_out].to_numpy(dtype=float)
        for q in quantiles:
            threshold = float(np.quantile(values, q))
            excesses = values[values > threshold] - threshold
            status = "PASS"
            xi = sigma = crps = float("nan")
            model = ""
            rate = float("nan")
            if len(excesses) < minimum_exceedances:
                status = "INSUFFICIENT_EXCESSES"
            else:
                try:
                    xi, _, sigma = map(float, stats.genpareto.fit(excesses, floc=0.0))
                    if not (math.isfinite(xi) and math.isfinite(sigma) and sigma > 0 and xi < 1):
                        status = "INVALID_FINITE_MEAN_FIT"
                    else:
                        counts = (fit_events.assign(exceed=fit_events.cluster_max_kw > threshold)
                                  .groupby("month_id").exceed.sum().reindex(fit_months, fill_value=0)
                                  .to_numpy(dtype=int))
                        model, rate, nb_size, nb_probability = _count_parameters(counts)
                        sub = (fit_events.loc[fit_events.cluster_max_kw <= threshold]
                               .groupby("month_id").cluster_max_kw.max()
                               .reindex(fit_months).fillna(0.0).to_numpy(dtype=float))
                        crps = float(np.mean([_crps_one(float(y), threshold, xi, sigma,
                                                        sub, model, rate, nb_size, nb_probability)
                                              for y in held_out_maxima]))
                        if not math.isfinite(crps):
                            status = "NONFINITE_CRPS"
                except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                    status = type(exc).__name__
            records.append(FoldFit(float(q), fold_index, len(fit_months), len(held_out),
                                   threshold, xi, sigma, len(excesses), model, rate, crps, status))
    scores = {}
    for q in quantiles:
        rows = [row for row in records if row.quantile == q]
        if len(rows) == 4 and all(row.status == "PASS" for row in rows):
            scores[q] = float(np.mean([row.crps for row in rows]))
    if not scores:
        raise ValueError("no complete finite-mean candidate across all four folds")
    return min(scores, key=lambda q: (scores[q], q)), records
