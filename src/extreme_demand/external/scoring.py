"""Proper and decision-oriented scores for the Korean holdout months."""

from __future__ import annotations

import numpy as np
from scipy import stats


def empirical_crps(draws: np.ndarray, observation: float) -> float:
    sample = np.sort(np.asarray(draws, dtype=float))
    n = sample.size
    first = float(np.mean(np.abs(sample - observation)))
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    pair_term = float(np.dot(weights, sample) / (n * n))
    return first - pair_term


def kde_negative_log_score(draws: np.ndarray, observation: float) -> float:
    sample = np.asarray(draws, dtype=float)
    if sample.size < 2 or np.isclose(sample.std(), 0.0):
        scale = max(abs(float(sample.mean())) * 1e-6, 1e-8)
        density = stats.norm.pdf(observation, loc=float(sample.mean()), scale=scale)
    else:
        density = float(stats.gaussian_kde(sample).evaluate([observation])[0])
    return -float(np.log(max(density, np.finfo(float).tiny)))


def stop_loss_absolute_score(
    draws: np.ndarray, observation: float, thresholds: list[float]
) -> float:
    sample = np.asarray(draws, dtype=float)
    losses = [
        abs(float(np.maximum(sample - threshold, 0.0).mean()) - max(observation - threshold, 0.0))
        for threshold in thresholds
    ]
    return float(np.mean(losses))
