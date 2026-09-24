"""Tie-safe out-of-sample economic metrics for paired simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .decision import EstimatedDecision
from .policy import ACTUAL, CAPACITY, CONTRACT, PolicyConfig, capacity_fee, contract_fee


def expected_test_fee(
    decision: EstimatedDecision,
    truth_kw: np.ndarray,
    S_kva: float,
    cfg: PolicyConfig,
) -> float:
    truth = np.asarray(truth_kw, dtype=float)
    if decision.selected_mode == CAPACITY:
        return float(capacity_fee(S_kva, cfg))
    if decision.selected_mode == ACTUAL:
        return cfg.demand_rate * float(truth.mean())
    if decision.D_kw is None:
        raise ValueError("contract decision requires D_kw")
    return float(np.asarray(contract_fee(decision.D_kw, truth, S_kva, cfg)).mean())


def paired_result_row(
    decision: EstimatedDecision,
    oracle: EstimatedDecision,
    truth_kw: np.ndarray,
    capacity_kva: float,
    cfg: PolicyConfig,
    **keys: Any,
) -> dict[str, Any]:
    truth = np.asarray(truth_kw, dtype=float)
    test_cost = expected_test_fee(decision, truth, capacity_kva, cfg)
    oracle_cost = expected_test_fee(oracle, truth, capacity_kva, cfg)
    regret = test_cost - oracle_cost
    mode_error = decision.selected_mode not in oracle.tie_set
    if decision.selected_mode == CONTRACT and decision.D_kw is not None:
        excess_probability = float(np.mean(truth > cfg.alpha * decision.D_kw))
    else:
        excess_probability = 0.0
    if (
        decision.selected_mode == CONTRACT
        and oracle.selected_mode == CONTRACT
        and decision.D_kw is not None
        and oracle.D_kw is not None
    ):
        D_error = abs(decision.D_kw - oracle.D_kw) / capacity_kva
    else:
        D_error = float("nan")
    return {
        **keys,
        "method_id": decision.method_id,
        "test_cost": test_cost,
        "regret": max(0.0, regret) if regret > -1e-7 else regret,
        "mode_error": bool(mode_error),
        "excess_probability": excess_probability,
        "D_error": D_error,
        "oracle_cost": oracle_cost,
        "selected_mode": decision.selected_mode,
        "oracle_mode": oracle.selected_mode,
    }
