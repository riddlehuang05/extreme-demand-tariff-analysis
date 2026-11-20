"""Billing-window aggregation with explicit complete-window semantics."""

from __future__ import annotations

import numpy as np


def _validated_power(power_mw: np.ndarray) -> np.ndarray:
    values = np.asarray(power_mw, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("power_mw must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("power_mw must contain finite non-negative values")
    return values


def fixed_nonoverlapping_average(power_mw: np.ndarray, window_minutes: int = 15) -> np.ndarray:
    values = _validated_power(power_mw)
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    complete = values.size // window_minutes
    if complete == 0:
        return np.empty(0, dtype=float)
    return values[: complete * window_minutes].reshape(complete, window_minutes).mean(axis=1)


def sliding_average(power_mw: np.ndarray, window_minutes: int = 15) -> np.ndarray:
    values = _validated_power(power_mw)
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    if values.size < window_minutes:
        return np.empty(0, dtype=float)
    cumulative = np.r_[0.0, np.cumsum(values, dtype=float)]
    return (cumulative[window_minutes:] - cumulative[:-window_minutes]) / window_minutes
