"""Small distribution adapters used by deterministic theory experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiniteDistribution:
    """A finite distribution with exact means, quantiles, and stop-loss values."""

    values: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if values.ndim != 1 or probabilities.ndim != 1 or values.size != probabilities.size:
            raise ValueError("values and probabilities must be equally sized one-dimensional arrays")
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("distribution values must be finite and non-empty")
        if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must be finite and non-negative")
        total = float(probabilities.sum())
        if not np.isclose(total, 1.0, atol=1e-12):
            raise ValueError(f"probabilities must sum to one, got {total}")
        order = np.argsort(values)
        object.__setattr__(self, "values", values[order])
        object.__setattr__(self, "probabilities", probabilities[order])

    @property
    def support_points(self) -> np.ndarray:
        return self.values.copy()

    def mean(self) -> float:
        return float(np.dot(self.values, self.probabilities))

    def ppf(self, probability: float) -> float:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        cumulative = np.cumsum(self.probabilities)
        index = int(np.searchsorted(cumulative, probability, side="left"))
        return float(self.values[min(index, self.values.size - 1)])

    def stop_loss(self, threshold: float) -> float:
        return float(np.dot(np.maximum(self.values - threshold, 0.0), self.probabilities))


def point_mass(value: float) -> FiniteDistribution:
    return FiniteDistribution(np.array([value]), np.array([1.0]))


def rare_upper_mean_preserving_spread(
    mean: float, upper_probability: float, upper_amplitude: float
) -> FiniteDistribution:
    """Spread a point mass while preserving its mean exactly.

    The upper point is ``mean + upper_amplitude``. The lower point offsets the
    upper probability so that the expected value remains ``mean``.
    """

    if not 0.0 < upper_probability < 1.0:
        raise ValueError("upper_probability must lie strictly between zero and one")
    if upper_amplitude <= 0.0:
        raise ValueError("upper_amplitude must be positive")
    lower = mean - upper_probability * upper_amplitude / (1.0 - upper_probability)
    if lower < 0.0:
        raise ValueError("spread construction produced negative demand")
    return FiniteDistribution(
        np.array([lower, mean + upper_amplitude]),
        np.array([1.0 - upper_probability, upper_probability]),
    )
