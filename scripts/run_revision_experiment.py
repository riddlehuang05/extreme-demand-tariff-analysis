from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Protocol
import warnings

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from track_a.estimators.pot_process import fit_process_tail  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "revision.yaml"
DESIGN_PATH = ROOT / "DESIGN.md"
CAPACITY = "capacity"
ACTUAL = "actual_maximum_demand"
CONTRACT = "contracted_maximum_demand"


class Distribution(Protocol):
    def cdf(self, x: float) -> float: ...
    def ppf(self, probability: float) -> float: ...
    def mean(self) -> float: ...
    def stop_loss(self, threshold: float) -> float: ...


@dataclass(frozen=True)
class DGP:
    threshold_kw: float
    xi: float
    sigma_kw: float
    tail_rate: float
    baseline_events: int
    baseline_lower_kw: float
    beta_a: float
    beta_b: float


@dataclass(frozen=True)
class Policy:
    policy_id: str
    demand_rate: float
    capacity_rate: float
    alpha: float
    kappa: float
    delta: float
    contract_upper_ratio: float

    @property
    def contract_probability(self) -> float:
        return 1.0 - 1.0 / (self.kappa * self.alpha)


def load_settings() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["revision"]


def dgp_from_settings(settings: dict[str, Any], tail_rate: float | None = None) -> DGP:
    row = settings["dgp"]
    return DGP(
        threshold_kw=float(row["threshold_kw"]),
        xi=float(row["gpd_shape_xi"]),
        sigma_kw=float(row["gpd_scale_kw"]),
        tail_rate=float(row["poisson_tail_rate_per_month"] if tail_rate is None else tail_rate),
        baseline_events=int(row["baseline_events_per_month"]),
        baseline_lower_kw=float(row["baseline_lower_kw"]),
        beta_a=float(row["baseline_beta_a"]),
        beta_b=float(row["baseline_beta_b"]),
    )


def seed_rng(seed_root: int, *namespace: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed_root, *namespace]))


def true_monthly_quantile(probability: float, dgp: DGP) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0,1)")
    no_tail = math.exp(-dgp.tail_rate)
    if probability < no_tail:
        beta_probability = (probability / no_tail) ** (1.0 / dgp.baseline_events)
        unit = stats.beta.ppf(beta_probability, dgp.beta_a, dgp.beta_b)
        return float(
            dgp.baseline_lower_kw
            + (dgp.threshold_kw - dgp.baseline_lower_kw) * unit
        )
    survival = -math.log(probability) / dgp.tail_rate
    excess = stats.genpareto.isf(survival, dgp.xi, loc=0.0, scale=dgp.sigma_kw)
    return float(dgp.threshold_kw + excess)


def generate_event_history(
    n_months: int, dgp: DGP, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, float | int]] = []
    event_rows: list[dict[str, float | int]] = []
    for month in range(n_months):
        baseline = dgp.baseline_lower_kw + (dgp.threshold_kw - dgp.baseline_lower_kw) * rng.beta(
            dgp.beta_a, dgp.beta_b, size=dgp.baseline_events
        )
        count = int(rng.poisson(dgp.tail_rate))
        tail = dgp.threshold_kw + stats.genpareto.rvs(
            dgp.xi, loc=0.0, scale=dgp.sigma_kw, size=count, random_state=rng
        )
        values = np.concatenate([baseline, np.asarray(tail, dtype=float)])
        monthly_rows.append({"month_id": month, "M_fixed15_kw": float(values.max())})
        event_rows.extend(
            {"month_id": month, "heat_id": index, "cluster_max_kw": float(value)}
            for index, value in enumerate(values)
        )
    return pd.DataFrame(monthly_rows), pd.DataFrame(event_rows)


def _poisson_tail_integral(s_max: float, xi: float, sigma: float, rate: float) -> float:
    if s_max <= 0.0 or rate <= 0.0:
        return 0.0

    def integrand(survival: float) -> float:
        if survival < 1e-10:
            return rate * survival ** (-xi)
        power = -1.0 if abs(xi) < 1e-10 else -xi - 1.0
        return -math.expm1(-rate * survival) * survival**power

    value = integrate.quad(
        integrand, 0.0, float(s_max), epsabs=1e-7, epsrel=2e-10, limit=250
    )[0]
    return float(sigma * value)


class OracleDistribution:
    def __init__(self, dgp: DGP):
        if dgp.xi >= 1.0:
            raise ValueError("oracle mean is infinite when xi >= 1")
        self.dgp = dgp
        self._tail_mean = _poisson_tail_integral(1.0, dgp.xi, dgp.sigma_kw, dgp.tail_rate)
        self._mean = self.stop_loss(0.0)

    def cdf(self, x: float) -> float:
        d = self.dgp
        if x < d.baseline_lower_kw:
            return 0.0
        if x < d.threshold_kw:
            z = (x - d.baseline_lower_kw) / (d.threshold_kw - d.baseline_lower_kw)
            return float(math.exp(-d.tail_rate) * stats.beta.cdf(z, d.beta_a, d.beta_b) ** d.baseline_events)
        survival = stats.genpareto.sf(x - d.threshold_kw, d.xi, loc=0.0, scale=d.sigma_kw)
        return float(math.exp(-d.tail_rate * survival))

    def ppf(self, probability: float) -> float:
        return true_monthly_quantile(probability, self.dgp)

    def mean(self) -> float:
        return self._mean

    def stop_loss(self, threshold: float) -> float:
        d = self.dgp
        threshold = max(0.0, float(threshold))
        if threshold >= d.threshold_kw:
            survival = stats.genpareto.sf(
                threshold - d.threshold_kw, d.xi, loc=0.0, scale=d.sigma_kw
            )
            return _poisson_tail_integral(survival, d.xi, d.sigma_kw, d.tail_rate)
        points = [d.baseline_lower_kw] if threshold < d.baseline_lower_kw < d.threshold_kw else None
        below = integrate.quad(
            lambda x: 1.0 - self.cdf(x),
            threshold,
            d.threshold_kw,
            points=points,
            epsabs=1e-6,
            epsrel=2e-11,
            limit=150,
        )[0]
        return float(below + self._tail_mean)


class DiscreteDistribution:
    def __init__(self, support: np.ndarray, probabilities: np.ndarray | None = None):
        values = np.asarray(support, dtype=float)
        if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError("support must be a finite nonempty vector")
        if probabilities is None:
            unique, counts = np.unique(values, return_counts=True)
            probs = counts / counts.sum()
        else:
            order = np.argsort(values)
            unique = values[order]
            probs = np.asarray(probabilities, dtype=float)[order]
        if np.any(probs < 0.0) or not np.isclose(probs.sum(), 1.0, atol=1e-10):
            raise ValueError("invalid probabilities")
        keep = probs > 0.0
        self.support = unique[keep]
        self.probabilities = probs[keep] / probs[keep].sum()
        self.cumulative = np.cumsum(self.probabilities)

    def cdf(self, x: float) -> float:
        index = int(np.searchsorted(self.support, x, side="right"))
        return 0.0 if index == 0 else float(self.cumulative[index - 1])

    def ppf(self, probability: float) -> float:
        index = int(np.searchsorted(self.cumulative, probability, side="left"))
        return float(self.support[min(index, self.support.size - 1)])

    def mean(self) -> float:
        return float(np.dot(self.support, self.probabilities))

    def stop_loss(self, threshold: float) -> float:
        return float(np.dot(np.maximum(self.support - threshold, 0.0), self.probabilities))


class NonnegativeGaussianKDE:
    def __init__(self, sample: np.ndarray):
        values = np.asarray(sample, dtype=float)
        if values.size < 2 or np.isclose(values.std(), 0.0):
            raise ValueError("degenerate sample")
        self.sample = values
        self.bandwidth = float(1.06 * values.std(ddof=1) * values.size ** (-1.0 / 5.0))
        self.bandwidth = max(self.bandwidth, np.finfo(float).eps * max(values.mean(), 1.0))
        self._normalizers = stats.norm.cdf(values / self.bandwidth)
        self._normalizer = float(self._normalizers.sum())

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        upper = stats.norm.cdf((x - self.sample) / self.bandwidth)
        lower = stats.norm.cdf(-self.sample / self.bandwidth)
        return float(np.clip((upper - lower).sum() / self._normalizer, 0.0, 1.0))

    def ppf(self, probability: float) -> float:
        high = float(self.sample.max() + 12.0 * self.bandwidth)
        while self.cdf(high) < probability:
            high *= 2.0
        return float(optimize.brentq(lambda x: self.cdf(x) - probability, 0.0, high))

    def mean(self) -> float:
        z = self.sample / self.bandwidth
        numerator = self.sample * stats.norm.cdf(z) + self.bandwidth * stats.norm.pdf(z)
        return float(numerator.sum() / self._normalizer)

    def stop_loss(self, threshold: float) -> float:
        threshold = max(0.0, float(threshold))
        z = (self.sample - threshold) / self.bandwidth
        numerator = (
            (self.sample - threshold) * stats.norm.cdf(z)
            + self.bandwidth * stats.norm.pdf(z)
        )
        return float(numerator.sum() / self._normalizer)


class TruncatedGEV:
    def __init__(self, shape_c: float, loc: float, scale: float, upper: float):
        self.shape_c = float(shape_c)
        self.loc = float(loc)
        self.scale = float(scale)
        self.upper = float(upper)
        self._f0 = float(stats.genextreme.cdf(0.0, self.shape_c, loc=self.loc, scale=self.scale))
        self._fu = float(stats.genextreme.cdf(self.upper, self.shape_c, loc=self.loc, scale=self.scale))
        self._z = self._fu - self._f0
        if self._z <= 0.0:
            raise ValueError("empty truncated GEV support")

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= self.upper:
            return 1.0
        raw = stats.genextreme.cdf(x, self.shape_c, loc=self.loc, scale=self.scale)
        return float(np.clip((raw - self._f0) / self._z, 0.0, 1.0))

    def ppf(self, probability: float) -> float:
        raw_probability = self._f0 + probability * self._z
        return float(stats.genextreme.ppf(raw_probability, self.shape_c, loc=self.loc, scale=self.scale))

    def mean(self) -> float:
        return self.stop_loss(0.0)

    def stop_loss(self, threshold: float) -> float:
        threshold = min(max(0.0, float(threshold)), self.upper)
        return float(
            integrate.quad(
                lambda x: 1.0 - self.cdf(x),
                threshold,
                self.upper,
                epsabs=1e-7,
                epsrel=2e-8,
                limit=180,
            )[0]
        )


class TailProcessDistribution:
    def __init__(
        self,
        threshold: float,
        xi: float,
        sigma: float,
        baseline_monthly_maxima: np.ndarray,
        count_model: str,
        count_mean: float,
        count_variance: float,
        nb_size: float | None,
        nb_probability: float | None,
    ):
        if xi >= 1.0:
            raise ValueError("TAIL distribution must satisfy xi < 1")
        self.threshold = float(threshold)
        self.xi = float(xi)
        self.sigma = float(sigma)
        self.baseline = np.asarray(baseline_monthly_maxima, dtype=float)
        self.count_model = count_model
        self.count_mean = float(count_mean)
        self.count_variance = float(count_variance)
        self.nb_size = nb_size
        self.nb_probability = nb_probability

    def _pgf(self, g: float) -> float:
        g = float(np.clip(g, 0.0, 1.0))
        if self.count_model == "poisson":
            return float(math.exp(-self.count_mean * (1.0 - g)))
        if self.nb_size is None or self.nb_probability is None:
            raise RuntimeError("negative-binomial parameters missing")
        p = self.nb_probability
        return float(
            math.exp(
                -self.nb_size
                * math.log1p(((1.0 - p) / p) * (1.0 - g))
            )
        )

    def _tail_integral(self, survival_max: float) -> float:
        if survival_max <= 0.0 or self.count_mean <= 0.0:
            return 0.0
        transform_power = 1.0 / (1.0 - self.xi)
        factor = float(survival_max) ** (1.0 - self.xi) * transform_power

        def tail_probability_ratio(survival: float) -> float:
            if survival <= np.finfo(float).tiny:
                return self.count_mean
            if self.count_model == "poisson":
                tail_probability = -math.expm1(-self.count_mean * survival)
            else:
                if self.nb_size is None or self.nb_probability is None:
                    raise RuntimeError("negative-binomial parameters missing")
                odds = (1.0 - self.nb_probability) / self.nb_probability
                log_pgf = -self.nb_size * math.log1p(odds * survival)
                tail_probability = -math.expm1(log_pgf)
            return tail_probability / survival

        def integrand(unit: float) -> float:
            survival = float(survival_max) * unit**transform_power
            return factor * tail_probability_ratio(survival)

        value = integrate.quad(
            integrand, 0.0, 1.0, epsabs=1e-7, epsrel=3e-10, limit=160
        )[0]
        return float(self.sigma * value)

    def cdf(self, x: float) -> float:
        baseline_cdf = float(np.mean(self.baseline <= x))
        if x < self.threshold:
            return baseline_cdf * self._pgf(0.0)
        gpd_cdf = float(stats.genpareto.cdf(x - self.threshold, self.xi, loc=0.0, scale=self.sigma))
        return float(np.clip(baseline_cdf * self._pgf(gpd_cdf), 0.0, 1.0))

    def ppf(self, probability: float) -> float:
        high = max(float(self.baseline.max()), self.threshold + 10.0 * self.sigma)
        if self.xi < 0.0:
            high = max(high, self.threshold - self.sigma / self.xi)
        else:
            while self.cdf(high) < probability:
                high *= 2.0
        return float(optimize.brentq(lambda x: self.cdf(x) - probability, 0.0, high))

    def mean(self) -> float:
        return self.stop_loss(0.0)

    def stop_loss(self, threshold: float) -> float:
        threshold = max(0.0, float(threshold))
        if threshold >= self.threshold:
            survival = float(
                stats.genpareto.sf(
                    threshold - self.threshold, self.xi, loc=0.0, scale=self.sigma
                )
            )
            return self._tail_integral(survival)
        p_zero = self._pgf(0.0)
        cdf_area = float(
            np.mean(np.maximum(self.threshold - np.maximum(self.baseline, threshold), 0.0))
        )
        below = (self.threshold - threshold) - p_zero * cdf_area
        return float(below + self._tail_integral(1.0))


def _count_parameters(counts: np.ndarray) -> tuple[str, float, float, float | None, float | None]:
    mean = float(np.mean(counts))
    variance = float(np.var(counts, ddof=1)) if counts.size > 1 else mean
    if mean <= 0.0 or variance / max(mean, np.finfo(float).eps) <= 1.2:
        return "poisson", mean, variance, None, None
    size = mean * mean / max(variance - mean, np.finfo(float).eps)
    probability = size / (size + mean)
    return "negative_binomial", mean, variance, float(size), float(probability)


def fit_tail_exact(
    monthly: pd.DataFrame,
    events: pd.DataFrame,
    settings: dict[str, Any],
    mode: str,
    rng: np.random.Generator,
    bootstrap_rng: np.random.Generator,
    physical_bound: float,
) -> tuple[Distribution, dict[str, Any]]:
    cfg = settings["tail"]
    original = fit_process_tail(
        monthly,
        events,
        int(settings["tail_selection_draws"][mode]),
        rng,
        bootstrap_rng,
        [float(value) for value in cfg["threshold_quantiles"]],
        int(cfg["minimum_exceedances_main"]),
        int(cfg["minimum_exceedances_fallback"]),
        int(settings["estimator_bootstrap_replicates"][mode]),
        float(cfg["holdout_fraction"]),
        float(cfg["shape_ci_width_gate"]),
        physical_bound,
        0.01,
        int(settings["validation_draws"][mode]),
        float(cfg["stability_penalty_weight"]),
        decluster=True,
    )
    candidates = list((original.diagnostics or {}).get("candidate_diagnostics", []))
    finite_candidates = [
        row
        for row in candidates
        if bool(row.get("valid_main"))
        and np.isfinite(row.get("validation_log_score", np.nan))
        and np.isfinite(row.get("xi", np.nan))
        and float(row["xi"]) < 1.0
    ]
    selection_status = "PASS"
    fallback_reason: str | None = None
    if finite_candidates:
        median_xi = float(np.median([float(row["xi"]) for row in finite_candidates]))
        chosen = max(
            finite_candidates,
            key=lambda row: (
                float(row["validation_log_score"])
                - float(cfg["stability_penalty_weight"]) * abs(float(row["xi"]) - median_xi),
                -float(row["quantile"]),
            ),
        )
        force_exponential = False
    else:
        fallback_candidates = [
            row
            for row in candidates
            if int(row.get("exceedances", 0)) >= int(cfg["minimum_exceedances_fallback"])
        ]
        if not fallback_candidates:
            return DiscreteDistribution(monthly["M_fixed15_kw"].to_numpy(dtype=float)), {
                "fit_status": "FALLBACK_EMPIRICAL",
                "fallback_reason": "no_finite_mean_candidate_or_insufficient_exceedances",
                "selected_threshold_quantile": np.nan,
                "threshold_kw": np.nan,
                "xi_hat": np.nan,
                "count_model": "empirical",
                "refit_used_all_months": True,
                "finite_mean_gate_pass": False,
                "finite_variance_flag": False,
            }
        chosen = max(
            fallback_candidates,
            key=lambda row: (int(row["exceedances"]), -float(row["quantile"])),
        )
        force_exponential = True
        selection_status = "FALLBACK_EXPONENTIAL"
        fallback_reason = "no_candidate_passed_all_selection_and_xi_less_than_one_gates"

    threshold = float(chosen["threshold"])
    values = events["cluster_max_kw"].to_numpy(dtype=float)
    excesses = values[values > threshold] - threshold
    if excesses.size < int(cfg["minimum_exceedances_fallback"]):
        return DiscreteDistribution(monthly["M_fixed15_kw"].to_numpy(dtype=float)), {
            "fit_status": "FALLBACK_EMPIRICAL",
            "fallback_reason": "insufficient_exceedances_after_full_history_refit",
            "selected_threshold_quantile": float(chosen["quantile"]),
            "threshold_kw": threshold,
            "xi_hat": np.nan,
            "count_model": "empirical",
            "refit_used_all_months": True,
            "finite_mean_gate_pass": False,
            "finite_variance_flag": False,
        }
    try:
        xi, _, sigma = (float(value) for value in stats.genpareto.fit(excesses, floc=0.0))
    except Exception:
        xi, sigma = float("nan"), float("nan")
    invalid_refit = (
        force_exponential
        or not np.isfinite(xi)
        or not np.isfinite(sigma)
        or sigma <= 0.0
        or xi >= 1.0
        or (xi < 0.0 and threshold - sigma / xi < values.max() - 1e-8)
    )
    if invalid_refit:
        xi = 0.0
        sigma = float(excesses.mean())
        selection_status = "FALLBACK_EXPONENTIAL"
        fallback_reason = fallback_reason or "full_history_refit_failed_finite_mean_or_support_gate"

    months = np.sort(monthly["month_id"].unique())
    counts = (
        events.assign(exceed=events["cluster_max_kw"] > threshold)
        .groupby("month_id")["exceed"]
        .sum()
        .reindex(months, fill_value=0)
        .to_numpy(dtype=int)
    )
    count_model, count_mean, count_variance, nb_size, nb_probability = _count_parameters(counts)
    below = events[events["cluster_max_kw"] <= threshold]
    baseline = (
        below.groupby("month_id")["cluster_max_kw"]
        .max()
        .reindex(months, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    distribution = TailProcessDistribution(
        threshold,
        xi,
        sigma,
        baseline,
        count_model,
        count_mean,
        count_variance,
        nb_size,
        nb_probability,
    )
    return distribution, {
        "fit_status": selection_status,
        "fallback_reason": fallback_reason,
        "selected_threshold_quantile": float(chosen["quantile"]),
        "threshold_kw": threshold,
        "selection_xi_hat": float(chosen["xi"]),
        "xi_hat": xi,
        "sigma_hat_kw": sigma,
        "count_model": count_model,
        "count_mean": count_mean,
        "count_variance": count_variance,
        "refit_used_all_months": True,
        "finite_mean_gate_pass": bool(xi < 1.0),
        "finite_variance_flag": bool(xi < 0.5),
        "selection_fit_months": int(math.floor(len(months) * (1.0 - float(cfg["holdout_fraction"])))),
        "selection_validation_months": len(months) - int(math.floor(len(months) * (1.0 - float(cfg["holdout_fraction"])))),
    }


def fit_kde_exact(maxima: np.ndarray) -> tuple[Distribution, dict[str, Any]]:
    try:
        distribution = NonnegativeGaussianKDE(maxima)
        return distribution, {
            "fit_status": "PASS",
            "bandwidth_kw": distribution.bandwidth,
        }
    except ValueError:
        return DiscreteDistribution(maxima), {
            "fit_status": "FALLBACK_EMPIRICAL",
            "fallback_reason": "degenerate_monthly_maxima",
        }


def _fit_gev_finite_mean(maxima: np.ndarray) -> tuple[float, float, float]:
    sample = np.sort(np.asarray(maxima, dtype=float))
    n = sample.size
    if n < 6:
        raise ValueError("insufficient_monthly_maxima_for_gev")
    order = np.arange(1, n + 1, dtype=float)
    b0 = float(sample.mean())
    b1 = float(np.mean((order - 1.0) / (n - 1.0) * sample))
    b2 = float(
        np.mean((order - 1.0) * (order - 2.0) / ((n - 1.0) * (n - 2.0)) * sample)
    )
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    if not np.isfinite([l1, l2, l3]).all() or l2 <= 0.0:
        raise ValueError("invalid_sample_l_moments")
    tau3 = l3 / l2

    def theoretical_tau3(xi: float) -> float:
        if abs(xi) < 1e-8:
            return float((2.0 * math.log(3.0) - 3.0 * math.log(2.0)) / math.log(2.0))
        return float((2.0 * 3.0**xi - 3.0 * 2.0**xi + 1.0) / (2.0**xi - 1.0))

    lower_xi, upper_xi = -0.90, 0.90
    lower_tau, upper_tau = theoretical_tau3(lower_xi), theoretical_tau3(upper_xi)
    target = float(np.clip(tau3, min(lower_tau, upper_tau) + 1e-10, max(lower_tau, upper_tau) - 1e-10))
    xi = float(
        optimize.brentq(
            lambda value: theoretical_tau3(value) - target,
            lower_xi,
            upper_xi,
        )
    )
    if abs(xi) < 1e-7:
        scale = l2 / math.log(2.0)
        loc = l1 - scale * float(np.euler_gamma)
    else:
        gamma = float(special.gamma(1.0 - xi))
        scale = l2 * xi / (gamma * (2.0**xi - 1.0))
        loc = l1 - scale * (gamma - 1.0) / xi
    if not np.isfinite([xi, loc, scale]).all() or scale <= 0.0:
        raise ValueError("invalid_finite_mean_gev_l_moment_fit")
    return -xi, float(loc), float(scale)


def fit_gev_exact(maxima: np.ndarray, physical_bound: float) -> tuple[Distribution, dict[str, Any]]:
    try:
        shape_c, loc, scale = _fit_gev_finite_mean(maxima)
        xi = -shape_c
        outside = float(
            stats.genextreme.cdf(0.0, shape_c, loc=loc, scale=scale)
            + stats.genextreme.sf(physical_bound, shape_c, loc=loc, scale=scale)
        )
        if not np.isfinite([shape_c, loc, scale, outside]).all() or scale <= 0.0:
            raise ValueError("invalid_gev_parameters")
        if outside > 0.01:
            raise ValueError("gev_physical_support_gate")
        return TruncatedGEV(shape_c, loc, scale, physical_bound), {
            "fit_status": "PASS_LMOM_FINITE_MEAN",
            "xi_hat": xi,
            "sigma_hat_kw": scale,
            "truncation_probability": outside,
            "finite_mean_gate_pass": True,
            "finite_variance_flag": bool(xi < 0.5),
        }
    except Exception as exc:
        return DiscreteDistribution(maxima), {
            "fit_status": "FALLBACK_EMPIRICAL",
            "fallback_reason": str(exc),
            "finite_mean_gate_pass": False,
        }


def fit_event_empirical(events: pd.DataFrame) -> tuple[Distribution, dict[str, Any]]:
    values = np.sort(events["cluster_max_kw"].to_numpy(dtype=float))
    support, event_counts = np.unique(values, return_counts=True)
    event_cdf = np.cumsum(event_counts) / event_counts.sum()
    monthly_counts = events.groupby("month_id").size().to_numpy(dtype=int)
    maximum_cdf = np.mean(event_cdf[:, None] ** monthly_counts[None, :], axis=1)
    probabilities = np.diff(np.r_[0.0, maximum_cdf])
    probabilities[-1] += 1.0 - probabilities.sum()
    return DiscreteDistribution(support, probabilities), {
        "fit_status": "PASS",
        "event_rows": int(values.size),
        "count_distribution": "empirical_monthly_event_count",
    }


def distribution_builds(
    monthly: pd.DataFrame,
    events: pd.DataFrame,
    settings: dict[str, Any],
    mode: str,
    seed_root: int,
    seed_namespace: tuple[int, ...],
    physical_bound: float,
) -> list[tuple[str, str, str, Distribution, dict[str, Any]]]:
    maxima = monthly["M_fixed15_kw"].to_numpy(dtype=float)
    tail, tail_diag = fit_tail_exact(
        monthly,
        events,
        settings,
        mode,
        seed_rng(seed_root, *seed_namespace, 1),
        seed_rng(seed_root, *seed_namespace, 2),
        physical_bound,
    )
    event_emp, event_diag = fit_event_empirical(events)
    gev, gev_diag = fit_gev_exact(maxima, physical_bound)
    kde, kde_diag = fit_kde_exact(maxima)
    empirical = DiscreteDistribution(maxima)
    return [
        ("TAIL", "event_level", "GPD_count_pipeline", tail, tail_diag),
        ("EVENT-EMP", "event_level", "nonparametric_event_maximum", event_emp, event_diag),
        ("GEV", "monthly_maxima", "GEV", gev, gev_diag),
        ("KDE", "monthly_maxima", "Gaussian_KDE", kde, kde_diag),
        ("EMP", "monthly_maxima", "empirical", empirical, {"fit_status": "PASS"}),
    ]


def optimized_decision(distribution: Distribution, s_kva: float, policy: Policy) -> dict[str, Any]:
    lower = policy.delta * s_kva
    upper = policy.contract_upper_ratio * s_kva
    quantile = distribution.ppf(policy.contract_probability)
    contract_d = float(np.clip(quantile / policy.alpha, lower, upper))
    costs = {
        CAPACITY: float(policy.capacity_rate * s_kva),
        ACTUAL: float(policy.demand_rate * distribution.mean()),
        CONTRACT: float(
            policy.demand_rate
            * (contract_d + policy.kappa * distribution.stop_loss(policy.alpha * contract_d))
        ),
    }
    selected = min(costs, key=lambda name: (costs[name], name))
    ordered = sorted(costs.values())
    return {
        "selected_mode": selected,
        "contract_D_kw": contract_d,
        "mode_costs": costs,
        "margin_cny": float(ordered[1] - ordered[0]),
        "best_value_cny": float(costs[selected]),
    }


def true_cost_of_estimated_action(
    selected_mode: str,
    contract_d: float,
    oracle: OracleDistribution,
    s_kva: float,
    policy: Policy,
) -> float:
    if selected_mode == CAPACITY:
        return float(policy.capacity_rate * s_kva)
    if selected_mode == ACTUAL:
        return float(policy.demand_rate * oracle.mean())
    return float(
        policy.demand_rate
        * (contract_d + policy.kappa * oracle.stop_loss(policy.alpha * contract_d))
    )


def policies_and_cells(settings: dict[str, Any], oracle: OracleDistribution, alpha120_only: bool) -> list[dict[str, Any]]:
    tariff = settings["tariff"]
    rows: list[dict[str, Any]] = []
    for document in tariff["policies"]:
        if alpha120_only and not math.isclose(float(document["alpha"]), 1.20):
            continue
        policy = Policy(
            str(document["id"]),
            float(tariff["demand_rate"]),
            float(tariff["capacity_rate"]),
            float(document["alpha"]),
            float(document["kappa"]),
            float(tariff["delta"]),
            float(tariff["contract_upper_ratio"]),
        )
        for cell in document["cells"]:
            s_kva = oracle.mean() / float(cell["utilization"])
            decision = optimized_decision(oracle, s_kva, policy)
            rows.append(
                {
                    "policy": policy,
                    "cell_id": str(cell["id"]),
                    "utilization": float(cell["utilization"]),
                    "S_kva": float(s_kva),
                    "oracle": decision,
                }
            )
    return rows


def run_replication(
    replication: int,
    tail_rate: float,
    rate_index: int,
    settings: dict[str, Any],
    mode: str,
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seed_root = int(settings["seed_root"])
    mode_code = {"smoke": 1, "full": 2, "frequency": 3}[mode]
    dgp = dgp_from_settings(settings, tail_rate)
    oracle = OracleDistribution(dgp)
    history_rng = seed_rng(seed_root, mode_code, rate_index, replication, 10)
    monthly, events = generate_event_history(int(settings["training_months"]), dgp, history_rng)
    physical_bound = 10.0 * oracle.ppf(0.999)
    builds = distribution_builds(
        monthly,
        events,
        settings,
        mode,
        seed_root,
        (mode_code, rate_index, replication, 20),
        physical_bound,
    )
    true_mean = oracle.mean()
    true_q99 = oracle.ppf(0.99)
    rows: list[dict[str, Any]] = []
    for method_id, information_set, model_class, distribution, diagnostics in builds:
        estimated_mean = distribution.mean()
        estimated_q99 = distribution.ppf(0.99)
        for cell in cells:
            policy: Policy = cell["policy"]
            oracle_decision = cell["oracle"]
            s_kva = float(cell["S_kva"])
            estimated = optimized_decision(distribution, s_kva, policy)
            oracle_costs = oracle_decision["mode_costs"]
            mode_errors = {
                name: abs(float(estimated["mode_costs"][name]) - float(oracle_costs[name]))
                for name in (CAPACITY, ACTUAL, CONTRACT)
            }
            d_mv = 2.0 * max(mode_errors.values()) / float(oracle_decision["margin_cny"])
            selected_true_cost = true_cost_of_estimated_action(
                estimated["selected_mode"],
                estimated["contract_D_kw"],
                oracle,
                s_kva,
                policy,
            )
            regret = max(0.0, selected_true_cost - float(oracle_decision["best_value_cny"]))
            oracle_threshold = policy.alpha * float(oracle_decision["contract_D_kw"])
            true_stoploss = oracle.stop_loss(oracle_threshold)
            estimated_stoploss = distribution.stop_loss(oracle_threshold)
            action_correct = estimated["selected_mode"] == oracle_decision["selected_mode"]
            row = {
                "replication_id": replication,
                "tail_rate_per_month": tail_rate,
                "tail_event_probability_per_month": 1.0 - math.exp(-tail_rate),
                "method_id": method_id,
                "information_set": information_set,
                "model_class": model_class,
                "fit_status": diagnostics.get("fit_status", "PASS"),
                "policy_id": policy.policy_id,
                "alpha": policy.alpha,
                "kappa": policy.kappa,
                "cell_id": cell["cell_id"],
                "utilization": cell["utilization"],
                "S_kva": s_kva,
                "oracle_mode": oracle_decision["selected_mode"],
                "selected_mode": estimated["selected_mode"],
                "action_correct": bool(action_correct),
                "oracle_margin_cny": oracle_decision["margin_cny"],
                "oracle_best_value_cny": oracle_decision["best_value_cny"],
                "estimated_best_value_cny": estimated["best_value_cny"],
                "maximum_mode_value_error_cny": max(mode_errors.values()),
                "d_mv": d_mv,
                "d_mv_below_one": bool(d_mv < 1.0),
                "true_mean_kw": true_mean,
                "estimated_mean_kw": estimated_mean,
                "mean_relative_error": (estimated_mean - true_mean) / true_mean,
                "true_q99_kw": true_q99,
                "estimated_q99_kw": estimated_q99,
                "q99_relative_error": (estimated_q99 - true_q99) / true_q99,
                "oracle_contract_D_kw": oracle_decision["contract_D_kw"],
                "estimated_contract_D_kw": estimated["contract_D_kw"],
                "contract_D_error_over_S": (estimated["contract_D_kw"] - oracle_decision["contract_D_kw"]) / s_kva,
                "stoploss_threshold_kw": oracle_threshold,
                "true_stoploss_kw": true_stoploss,
                "estimated_stoploss_kw": estimated_stoploss,
                "stoploss_relative_error": (estimated_stoploss - true_stoploss) / true_stoploss,
                "regret_cny_per_month": regret,
                "regret_over_oracle_value": regret / float(oracle_decision["best_value_cny"]),
                "regret_over_capacity_fee": regret / (policy.capacity_rate * s_kva),
                "draw_count_for_final_functionals": 0,
                "final_functional_evaluation": "deterministic",
                "selected_threshold_quantile": diagnostics.get("selected_threshold_quantile", np.nan),
                "threshold_kw": diagnostics.get("threshold_kw", np.nan),
                "xi_hat": diagnostics.get("xi_hat", np.nan),
                "sigma_hat_kw": diagnostics.get("sigma_hat_kw", np.nan),
                "count_model": diagnostics.get("count_model", "not_applicable"),
                "fallback_reason": diagnostics.get("fallback_reason"),
                "refit_used_all_months": diagnostics.get("refit_used_all_months", False),
                "finite_mean_gate_pass": diagnostics.get("finite_mean_gate_pass", True),
                "finite_variance_flag": diagnostics.get("finite_variance_flag", True),
            }
            rows.append(row)
    return rows


def signature() -> str:
    digest = hashlib.sha256()
    files = [Path(__file__), CONFIG_PATH, DESIGN_PATH]
    files.extend(sorted((ROOT / "vendor").rglob("*.py")))
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def run_chunk(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(task["output"])
    start = int(task["start"])
    end = int(task["end"])
    rate_index = int(task["rate_index"])
    tail_rate = float(task["tail_rate"])
    label = f"rate{rate_index:02d}_{start:05d}_{end - 1:05d}"
    csv_path = output / "parts" / f"part_{label}.csv"
    meta_path = output / "parts" / f"part_{label}.json"
    if csv_path.exists() and meta_path.exists():
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        if record.get("signature") == task["signature"] and record.get("status") == "PASS":
            return {**record, "resumed": True}
    try:
        settings = task["settings"]
        dgp = dgp_from_settings(settings, tail_rate)
        oracle = OracleDistribution(dgp)
        cells = policies_and_cells(settings, oracle, bool(task["alpha120_only"]))
        rows: list[dict[str, Any]] = []
        for replication in range(start, end):
            rows.extend(
                run_replication(
                    replication,
                    tail_rate,
                    rate_index,
                    settings,
                    str(task["mode"]),
                    cells,
                )
            )
        frame = pd.DataFrame(rows)
        _atomic_csv(frame, csv_path)
        record = {
            "status": "PASS",
            "signature": task["signature"],
            "tail_rate": tail_rate,
            "rate_index": rate_index,
            "start": start,
            "end": end,
            "rows": len(frame),
            "csv": str(csv_path.resolve()),
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "duration_seconds": time.perf_counter() - started,
            "resumed": False,
        }
        temporary = meta_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(temporary, meta_path)
        return record
    except Exception as exc:
        return {
            "status": "FAIL",
            "signature": task["signature"],
            "tail_rate": tail_rate,
            "rate_index": rate_index,
            "start": start,
            "end": end,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "duration_seconds": time.perf_counter() - started,
        }


def run(mode: str) -> dict[str, Any]:
    if mode not in {"smoke", "full", "frequency"}:
        raise ValueError(mode)
    settings = load_settings()
    run_signature = signature()
    output = ROOT / "outputs" / mode
    output.mkdir(parents=True, exist_ok=True)
    if mode in {"full", "frequency"}:
        smoke_summary = ROOT / "outputs" / "smoke" / "summary.json"
        if not smoke_summary.exists():
            raise RuntimeError("smoke run is required before full experiments")
        smoke = json.loads(smoke_summary.read_text(encoding="utf-8"))
        if smoke.get("status") != "PASS" or smoke.get("signature") != run_signature:
            raise RuntimeError("smoke run does not match current code signature")

    if mode == "frequency":
        rates = [float(value) for value in settings["frequency_rates"]]
        total = int(settings["replications"]["frequency_per_rate"])
        chunk = int(settings["chunk_replications"]["frequency"])
        alpha120_only = True
    else:
        rates = [float(settings["dgp"]["poisson_tail_rate_per_month"])]
        total = int(settings["replications"][mode])
        chunk = int(settings["chunk_replications"][mode])
        alpha120_only = False

    tasks = []
    for rate_index, tail_rate in enumerate(rates):
        for start in range(0, total, chunk):
            tasks.append(
                {
                    "output": str(output.resolve()),
                    "settings": settings,
                    "signature": run_signature,
                    "mode": mode,
                    "rate_index": rate_index,
                    "tail_rate": tail_rate,
                    "start": start,
                    "end": min(start + chunk, total),
                    "alpha120_only": alpha120_only,
                }
            )
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(settings["workers"])) as executor:
        futures = [executor.submit(run_chunk, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[{mode}] rate={record['tail_rate']:g} {record['start']}:{record['end']} "
                f"{record['status']} {record['duration_seconds']:.1f}s",
                flush=True,
            )
    failures = [row for row in records if row["status"] != "PASS"]
    if failures:
        summary = {
            "status": "FAIL",
            "mode": mode,
            "signature": run_signature,
            "failures": failures,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise RuntimeError(f"worker failure: {failures[0]}")
    records.sort(key=lambda row: (row["rate_index"], row["start"]))
    frame = pd.concat([pd.read_csv(row["csv"]) for row in records], ignore_index=True)
    result_path = output / "revision_results.csv"
    _atomic_csv(frame, result_path)
    expected_cells = 7 if alpha120_only else 9
    expected_rows = len(rates) * total * 5 * expected_cells
    safe = frame[frame["d_mv_below_one"].astype(bool)]
    checks = {
        "expected_row_count": int(len(frame)) == expected_rows,
        "unique_keys": not frame.duplicated(
            ["tail_rate_per_month", "replication_id", "method_id", "cell_id"]
        ).any(),
        "all_primary_metrics_finite": bool(
            np.isfinite(
                frame[
                    [
                        "oracle_margin_cny",
                        "d_mv",
                        "mean_relative_error",
                        "q99_relative_error",
                        "stoploss_relative_error",
                        "regret_cny_per_month",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "d_mv_certificate_no_violations": bool(safe["action_correct"].astype(bool).all()),
        "deterministic_final_functionals": bool(
            (frame["draw_count_for_final_functionals"] == 0).all()
        ),
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": mode,
        "signature": run_signature,
        "replications_per_rate": total,
        "tail_rates": rates,
        "rows": int(len(frame)),
        "methods": sorted(frame["method_id"].unique()),
        "cells": int(frame["cell_id"].nunique()),
        "duration_seconds": time.perf_counter() - started,
        "checks": checks,
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError(f"verification failed: {checks}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "frequency"], required=True)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    run(args.mode)


if __name__ == "__main__":
    main()
