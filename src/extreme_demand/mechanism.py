"""Gate 1 equal-mean convex-order mechanism experiment."""

from __future__ import annotations

from typing import Any

import numpy as np

from .distributions import point_mass, rare_upper_mean_preserving_spread
from .policy import (
    ACTUAL,
    CAPACITY,
    CONTRACT,
    PolicyConfig,
    distribution_mean,
    evaluate_decision,
    select_mode,
)


def normalized_policy() -> PolicyConfig:
    return PolicyConfig(
        demand_rate=1.0,
        capacity_rate=0.625,
        alpha=1.05,
        kappa=2.0,
        delta=0.40,
        contract_upper_ratio=1.0,
        voltage_level="normalized",
        jurisdiction="theory",
        effective_rate_date="not_applicable",
        contract_evidence_status="historical_formal_rule_scenario",
    )


def mechanism_row(mean: float, upper_probability: float, amplitude: float) -> dict[str, Any]:
    cfg = normalized_policy()
    S_kva = 1.0
    base = point_mass(mean)
    spread = rare_upper_mean_preserving_spread(mean, upper_probability, amplitude)
    base_decision = select_mode(base, S_kva, cfg)
    spread_decision = select_mode(spread, S_kva, cfg)
    transported_base_cost = evaluate_decision(base_decision, spread, S_kva, cfg)
    value_of_tail_information = transported_base_cost - spread_decision.expected_cost
    base_contract = base_decision.mode_costs[CONTRACT]
    spread_contract = spread_decision.mode_costs[CONTRACT]

    return {
        "experiment_id": "E1",
        "mean_target": mean,
        "upper_probability": upper_probability,
        "upper_amplitude": amplitude,
        "base_mean": distribution_mean(base),
        "spread_mean": distribution_mean(spread),
        "mean_error": distribution_mean(spread) - distribution_mean(base),
        "base_capacity_cost": base_decision.mode_costs[CAPACITY],
        "spread_capacity_cost": spread_decision.mode_costs[CAPACITY],
        "capacity_cost_change": spread_decision.mode_costs[CAPACITY] - base_decision.mode_costs[CAPACITY],
        "base_actual_cost": base_decision.mode_costs[ACTUAL],
        "spread_actual_cost": spread_decision.mode_costs[ACTUAL],
        "actual_cost_change": spread_decision.mode_costs[ACTUAL] - base_decision.mode_costs[ACTUAL],
        "base_contract_cost": base_contract,
        "spread_contract_cost": spread_contract,
        "contract_cost_change": spread_contract - base_contract,
        "base_mode": base_decision.selected_mode,
        "spread_mode": spread_decision.selected_mode,
        "mode_switch": base_decision.selected_mode != spread_decision.selected_mode,
        "base_D_over_S": base_decision.contract_D_kw_by_month[0],
        "spread_D_over_S": spread_decision.contract_D_kw_by_month[0],
        "base_decision_cost_under_spread": transported_base_cost,
        "spread_oracle_cost": spread_decision.expected_cost,
        "value_of_tail_information": value_of_tail_information,
        "nonnegative_vti_check": value_of_tail_information >= -1e-12,
        "convex_order_contract_check": spread_contract + 1e-12 >= base_contract,
        "invariance_check": (
            abs(distribution_mean(spread) - distribution_mean(base)) <= 1e-12
            and abs(spread_decision.mode_costs[CAPACITY] - base_decision.mode_costs[CAPACITY]) <= 1e-12
            and abs(spread_decision.mode_costs[ACTUAL] - base_decision.mode_costs[ACTUAL]) <= 1e-12
        ),
    }


def mechanism_grid(smoke: bool = False) -> list[dict[str, Any]]:
    if smoke:
        means = np.array([0.45, 0.625, 0.80])
        probabilities = (0.01, 0.10)
        amplitudes = (0.10, 0.25)
    else:
        means = np.round(np.arange(0.45, 0.8000001, 0.025), 3)
        probabilities = (0.005, 0.01, 0.02, 0.05, 0.10)
        amplitudes = (0.05, 0.10, 0.15, 0.25)
    return [
        mechanism_row(float(mean), probability, amplitude)
        for mean in means
        for probability in probabilities
        for amplitude in amplitudes
    ]
