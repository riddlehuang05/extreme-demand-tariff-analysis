"""Monthly-maximum baselines for Gate 4."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .base import FitResult


def _sample(maxima_kw: np.ndarray) -> np.ndarray:
    sample = np.asarray(maxima_kw, dtype=float)
    if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)) or np.any(sample < 0):
        raise ValueError("maxima_kw must be a non-empty finite non-negative vector")
    return sample


def fit_point(maxima_kw: np.ndarray, n_draws: int) -> FitResult:
    sample = _sample(maxima_kw)
    center = float(sample.mean())
    return FitResult("POINT-3M", np.full(n_draws, center), diagnostics={"center": "mean"})


def fit_empirical(maxima_kw: np.ndarray, n_draws: int, rng: np.random.Generator) -> FitResult:
    sample = _sample(maxima_kw)
    return FitResult(
        "EMP-JOINT",
        rng.choice(sample, size=n_draws, replace=True),
        diagnostics={"training_rows": int(sample.size)},
    )


def fit_kde(maxima_kw: np.ndarray, n_draws: int, rng: np.random.Generator) -> FitResult:
    sample = _sample(maxima_kw)
    if sample.size < 2 or np.isclose(sample.std(), 0.0):
        result = fit_empirical(sample, n_draws, rng)
        return FitResult("KDE-MC-JOINT", result.draws_kw, "FALLBACK_EMPIRICAL", diagnostics={"reason": "degenerate_training_sample"})
    center = float(sample.mean())
    residuals = sample - center
    bandwidth = float(1.06 * residuals.std(ddof=1) * sample.size ** (-1.0 / 5.0))
    bandwidth = max(bandwidth, np.finfo(float).eps * max(center, 1.0))
    accepted: list[np.ndarray] = []
    count = 0
    batches = 0
    while count < n_draws and batches < 100:
        size = max(n_draws - count, 256)
        base = rng.choice(residuals, size=size, replace=True)
        proposed = center + base + rng.normal(0.0, bandwidth, size=size)
        valid = proposed[proposed >= 0.0]
        accepted.append(valid)
        count += valid.size
        batches += 1
    if count < n_draws:
        raise RuntimeError("boundary-corrected KDE rejection sampler did not converge")
    draws = np.concatenate(accepted)[:n_draws]
    return FitResult(
        "KDE-MC-JOINT",
        draws,
        diagnostics={
            "center_kw": center,
            "bandwidth_kw": bandwidth,
            "boundary": "reject_below_zero",
            "rejection_batches": batches,
        },
    )


def fit_gev(
    maxima_kw: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    physical_bound_kw: float,
    truncation_fraction_gate: float,
) -> FitResult:
    sample = _sample(maxima_kw)
    if sample.size < 6 or np.isclose(sample.std(), 0.0):
        fallback = fit_empirical(sample, n_draws, rng)
        return FitResult("GEV-JOINT", fallback.draws_kw, "FALLBACK_EMPIRICAL", diagnostics={"reason": "insufficient_or_degenerate_months"})
    try:
        shape_c, loc, scale = (float(value) for value in stats.genextreme.fit(sample))
        if not np.isfinite([shape_c, loc, scale]).all() or scale <= 0.0:
            raise ValueError("invalid GEV parameters")
        raw = stats.genextreme.rvs(shape_c, loc=loc, scale=scale, size=n_draws, random_state=rng)
        invalid = ~np.isfinite(raw)
        raw[invalid] = physical_bound_kw
        truncation = float(np.mean((raw < 0.0) | (raw > physical_bound_kw)))
        if truncation > truncation_fraction_gate:
            fallback = fit_empirical(sample, n_draws, rng)
            return FitResult(
                "GEV-JOINT",
                fallback.draws_kw,
                "FALLBACK_EMPIRICAL",
                diagnostics={
                    "reason": "truncation_gate",
                    "scipy_shape_c": shape_c,
                    "loc_kw": loc,
                    "sigma_kw": scale,
                    "truncation_fraction": truncation,
                    "truncation_fraction_gate": truncation_fraction_gate,
                },
            )
        draws = np.clip(raw, 0.0, physical_bound_kw)
        return FitResult(
            "GEV-JOINT",
            draws,
            "PASS",
            xi=-shape_c,
            sigma=scale,
            diagnostics={"scipy_shape_c": shape_c, "loc_kw": loc, "truncation_fraction": truncation},
        )
    except Exception as exc:
        fallback = fit_empirical(sample, n_draws, rng)
        return FitResult(
            "GEV-JOINT",
            fallback.draws_kw,
            "FALLBACK_EMPIRICAL",
            diagnostics={"reason": type(exc).__name__, "message": str(exc)},
        )
