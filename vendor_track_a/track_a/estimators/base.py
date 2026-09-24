"""Common estimator result contract with machine-readable diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FitResult:
    method_id: str
    draws_kw: np.ndarray
    fit_status: str = "PASS"
    threshold: float | None = None
    xi: float | None = None
    sigma: float | None = None
    cluster_rate: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        draws = np.asarray(self.draws_kw, dtype=float)
        if draws.ndim != 1 or draws.size == 0:
            raise ValueError("draws_kw must be a non-empty one-dimensional sample")
        if not np.all(np.isfinite(draws)) or np.any(draws < 0.0):
            raise ValueError("draws_kw must be finite and non-negative")
        object.__setattr__(self, "draws_kw", draws)

    def as_row(self, **keys: Any) -> dict[str, Any]:
        return {
            **keys,
            "method_id": self.method_id,
            "fit_status": self.fit_status,
            "threshold": self.threshold,
            "xi": self.xi,
            "sigma": self.sigma,
            "cluster_rate": self.cluster_rate,
            "draw_count": int(self.draws_kw.size),
            "draw_mean_kw": float(self.draws_kw.mean()),
            "draw_q95_kw": float(np.quantile(self.draws_kw, 0.95)),
            "draw_q99_kw": float(np.quantile(self.draws_kw, 0.99)),
            "diagnostics": json.dumps(self.diagnostics, sort_keys=True, separators=(",", ":")),
        }
