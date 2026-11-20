"""Gate 4 decision adapters for probabilistic and deterministic estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .policy import (
    ACTUAL,
    CAPACITY,
    CONTRACT,
    PolicyConfig,
    capacity_fee,
    contract_fee,
    select_mode,
)


@dataclass(frozen=True)
class EstimatedDecision:
    method_id: str
    selected_mode: str
    D_kw: float | None
    estimated_cost: float
    mode_costs: dict[str, float]
    tie_set: tuple[str, ...]


def decision_from_draws(
    method_id: str,
    draws_kw: np.ndarray,
    S_kva: float,
    cfg: PolicyConfig,
    tie_epsilon: float,
    force_contract: bool = False,
) -> EstimatedDecision:
    sample = np.asarray(draws_kw, dtype=float)
    mode = select_mode(sample, S_kva, cfg, tie_epsilon=tie_epsilon)
    selected = CONTRACT if force_contract else mode.selected_mode
    cost = mode.mode_costs[selected]
    contract_D = mode.contract_D_kw_by_month[0] if selected == CONTRACT else None
    return EstimatedDecision(
        method_id=method_id,
        selected_mode=selected,
        D_kw=contract_D,
        estimated_cost=float(cost),
        mode_costs=mode.mode_costs,
        tie_set=(CONTRACT,) if force_contract else mode.tie_set,
    )


def patent_history_decision(
    maxima_kw: np.ndarray,
    S_kva: float,
    cfg: PolicyConfig,
    candidate_summaries: Iterable[str],
    tie_epsilon: float,
) -> EstimatedDecision:
    sample = np.asarray(maxima_kw, dtype=float)
    summaries = {
        "mean": float(sample.mean()),
        "median": float(np.median(sample)),
        "q90": float(np.quantile(sample, 0.90)),
        "maximum": float(sample.max()),
    }
    lower, upper = cfg.delta * S_kva, cfg.contract_upper_ratio * S_kva
    candidates = np.unique(
        np.clip([summaries[name] / cfg.alpha for name in candidate_summaries], lower, upper)
    )
    contract_costs = np.array(
        [np.asarray(contract_fee(d, sample, S_kva, cfg), dtype=float).mean() for d in candidates]
    )
    best_index = int(np.argmin(contract_costs))
    best_d = float(candidates[best_index])
    mode_costs = {
        CAPACITY: float(capacity_fee(S_kva, cfg)),
        ACTUAL: cfg.demand_rate * float(sample.mean()),
        CONTRACT: float(contract_costs[best_index]),
    }
    minimum = min(mode_costs.values())
    tie_set = tuple(
        mode for mode, value in mode_costs.items() if value <= minimum + tie_epsilon + 1e-12
    )
    selected = min(tie_set, key=lambda mode: (mode_costs[mode], mode))
    return EstimatedDecision(
        method_id="PATENT-HIST-3M",
        selected_mode=selected,
        D_kw=best_d if selected == CONTRACT else None,
        estimated_cost=float(mode_costs[selected]),
        mode_costs=mode_costs,
        tie_set=tie_set,
    )


def realized_fee(decision: EstimatedDecision, realized_M_kw: float, S_kva: float, cfg: PolicyConfig) -> float:
    if decision.selected_mode == CAPACITY:
        return float(capacity_fee(S_kva, cfg))
    if decision.selected_mode == ACTUAL:
        return cfg.demand_rate * realized_M_kw
    if decision.D_kw is None:
        raise ValueError("contract decision requires D_kw")
    return float(contract_fee(decision.D_kw, realized_M_kw, S_kva, cfg))


def tai_margin_decision(
    maxima_kw: np.ndarray,
    S_kva: float,
    cfg: PolicyConfig,
    beta_grid: Iterable[float],
    residual_quantile: float,
    minimum_inner_train_months: int,
    tie_epsilon: float,
) -> tuple[EstimatedDecision, dict[str, Any]]:
    sample = np.asarray(maxima_kw, dtype=float)
    if sample.size < minimum_inner_train_months + 1:
        selected_beta = 0.0
        scores = {str(float(beta)): None for beta in beta_grid}
    else:
        losses: dict[float, list[float]] = {float(beta): [] for beta in beta_grid}
        for index in range(minimum_inner_train_months, sample.size):
            inner = sample[:index]
            center = float(inner.mean())
            expanding_forecasts = np.array([inner[:j].mean() for j in range(1, inner.size)])
            residuals = inner[1:] - expanding_forecasts
            margin = float(np.quantile(np.abs(residuals), residual_quantile))
            for beta in losses:
                forecast = max(0.0, center + beta * margin)
                decision = decision_from_draws(
                    "TAI-MARGIN", np.array([forecast]), S_kva, cfg, tie_epsilon
                )
                losses[beta].append(realized_fee(decision, float(sample[index]), S_kva, cfg))
        scores = {str(beta): float(np.mean(values)) for beta, values in losses.items()}
        selected_beta = min(losses, key=lambda beta: (scores[str(beta)], beta))
    center = float(sample.mean())
    if sample.size > 1:
        forecasts = np.array([sample[:j].mean() for j in range(1, sample.size)])
        margin = float(np.quantile(np.abs(sample[1:] - forecasts), residual_quantile))
    else:
        margin = 0.0
    adjusted = max(0.0, center + selected_beta * margin)
    decision = decision_from_draws(
        "TAI-MARGIN", np.array([adjusted]), S_kva, cfg, tie_epsilon
    )
    return decision, {
        "selected_beta": float(selected_beta),
        "center_kw": center,
        "margin_kw": margin,
        "adjusted_kw": adjusted,
        "validation_scores": scores,
    }
