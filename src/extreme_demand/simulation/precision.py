"""Pre-registered adaptive Monte Carlo precision rules."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def wilson_halfwidth(successes: int, n: int, confidence_level: float) -> float:
    if n <= 0:
        return float("inf")
    z = float(stats.norm.ppf(0.5 + confidence_level / 2.0))
    p = successes / n
    denominator = 1.0 + z * z / n
    return float(
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )


def precision_rows(
    paired_results: pd.DataFrame,
    capacity_fee_monthly: dict[str, float],
    comparison_baselines: list[str],
    precision_methods: list[str],
    confidence_level: float,
    mode_halfwidth_target: float,
    gain_relative_halfwidth: float,
    gain_floor_fraction: float,
) -> list[dict[str, Any]]:
    required = {"replication_id", "scenario_id", "training_months", "method_id", "regret", "mode_error"}
    missing = required - set(paired_results.columns)
    if missing:
        raise ValueError(f"paired result table missing precision columns: {sorted(missing)}")
    z = float(stats.norm.ppf(0.5 + confidence_level / 2.0))
    rows: list[dict[str, Any]] = []
    for (scenario_id, training_months), group in paired_results.groupby(["scenario_id", "training_months"]):
        pivot = group.pivot(index="replication_id", columns="method_id", values="regret")
        n = int(pivot.shape[0])
        for method_id in precision_methods:
            method = group[group["method_id"] == method_id]
            halfwidth = wilson_halfwidth(int(method["mode_error"].sum()), len(method), confidence_level)
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "training_months": int(training_months),
                    "criterion": "mode_error_halfwidth",
                    "method_id": method_id,
                    "baseline_id": None,
                    "replications": len(method),
                    "estimate": float(method["mode_error"].mean()),
                    "halfwidth": halfwidth,
                    "target": mode_halfwidth_target,
                    "passed": halfwidth <= mode_halfwidth_target,
                }
            )
        for baseline in comparison_baselines:
            paired = pivot[[baseline, "TAIL-JOINT"]].dropna()
            gains = (paired[baseline] - paired["TAIL-JOINT"]).to_numpy(dtype=float)
            mean_gain = float(gains.mean()) if gains.size else float("nan")
            halfwidth = (
                float(z * gains.std(ddof=1) / math.sqrt(gains.size))
                if gains.size > 1
                else float("inf")
            )
            target = max(
                gain_relative_halfwidth * abs(mean_gain),
                gain_floor_fraction * capacity_fee_monthly[str(scenario_id)],
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "training_months": int(training_months),
                    "criterion": "paired_regret_gain_halfwidth",
                    "method_id": "TAIL-JOINT",
                    "baseline_id": baseline,
                    "replications": int(gains.size),
                    "estimate": mean_gain,
                    "halfwidth": halfwidth,
                    "target": target,
                    "passed": bool(np.isfinite(halfwidth) and halfwidth <= target),
                }
            )
    return rows
