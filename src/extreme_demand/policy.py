"""Pure policy-cost and three-mode decision functions for Track A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


CAPACITY = "capacity"
ACTUAL = "actual_maximum_demand"
CONTRACT = "contracted_maximum_demand"


@dataclass(frozen=True)
class PolicyConfig:
    demand_rate: float
    capacity_rate: float
    alpha: float
    kappa: float
    delta: float
    contract_upper_ratio: float
    voltage_level: str
    jurisdiction: str
    effective_rate_date: str
    contract_evidence_status: str

    def __post_init__(self) -> None:
        for name in ("demand_rate", "capacity_rate", "alpha", "kappa"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.delta <= self.contract_upper_ratio:
            raise ValueError("contract ratios must satisfy 0 <= delta <= upper ratio")
        if self.contract_upper_ratio > 1.0:
            raise ValueError("contract upper ratio cannot exceed one")
        if not 0.0 < 1.0 - 1.0 / (self.kappa * self.alpha) < 1.0:
            raise ValueError("kappa*alpha must imply an interior quantile probability")

    @property
    def contract_quantile_probability(self) -> float:
        return 1.0 - 1.0 / (self.kappa * self.alpha)

    @classmethod
    def from_yaml(cls, path: str | Path, voltage_level: str | None = None) -> "PolicyConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        policy = document["policy"]
        selected_voltage = voltage_level or policy["voltage_level_main"]
        rates = policy["rates"][selected_voltage]
        rule = policy["contracted_rule"]
        return cls(
            demand_rate=float(rates["demand_cny_per_kw_month"]),
            capacity_rate=float(rates["capacity_cny_per_kva_month"]),
            alpha=float(rule["tolerance_alpha"]),
            kappa=float(rule["excess_total_multiplier_kappa"]),
            delta=float(rule["minimum_contract_ratio_delta"]),
            contract_upper_ratio=float(rule["contract_upper_ratio"]),
            voltage_level=selected_voltage,
            jurisdiction=str(policy["jurisdiction"]),
            effective_rate_date=str(policy["effective_rate_date"]),
            contract_evidence_status=str(rule["evidence_status"]),
        )


@dataclass(frozen=True)
class ContractDecision:
    D_kw: float
    expected_cost: float
    quantile_probability: float
    bound_status: str


@dataclass(frozen=True)
class ModeDecision:
    selected_mode: str
    contract_D_kw_by_month: tuple[float, ...]
    expected_cost: float
    mode_costs: dict[str, float]
    tie_set: tuple[str, ...]
    tie_epsilon: float


def _nonnegative_finite(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain only finite non-negative values")
    return array


def _return_scalar_if_all_scalar(originals: tuple[Any, ...], result: np.ndarray) -> float | np.ndarray:
    return float(result) if all(np.asarray(value).ndim == 0 for value in originals) else result


def capacity_fee(S_kva: Any, cfg: PolicyConfig) -> float | np.ndarray:
    capacity = _nonnegative_finite(S_kva, "S_kva")
    return _return_scalar_if_all_scalar((S_kva,), cfg.capacity_rate * capacity)


def actual_fee(M_kw: Any, cfg: PolicyConfig) -> float | np.ndarray:
    demand = _nonnegative_finite(M_kw, "M_kw")
    return _return_scalar_if_all_scalar((M_kw,), cfg.demand_rate * demand)


def contract_fee(D_kw: Any, M_kw: Any, S_kva: float, cfg: PolicyConfig) -> float | np.ndarray:
    contract = _nonnegative_finite(D_kw, "D_kw")
    demand = _nonnegative_finite(M_kw, "M_kw")
    capacity = _nonnegative_finite(S_kva, "S_kva")
    if capacity.ndim != 0 or float(capacity) <= 0.0:
        raise ValueError("S_kva must be a positive scalar")
    lower = cfg.delta * float(capacity)
    upper = cfg.contract_upper_ratio * float(capacity)
    if np.any(contract < lower - 1e-12) or np.any(contract > upper + 1e-12):
        raise ValueError(f"D_kw must be within [{lower}, {upper}]")
    result = cfg.demand_rate * contract + cfg.kappa * cfg.demand_rate * np.maximum(
        demand - cfg.alpha * contract, 0.0
    )
    return _return_scalar_if_all_scalar((D_kw, M_kw), np.asarray(result))


def distribution_mean(distribution: Any) -> float:
    if isinstance(distribution, np.ndarray):
        if distribution.size == 0:
            raise ValueError("empirical distribution cannot be empty")
        return float(_nonnegative_finite(distribution, "distribution").mean())
    mean = distribution.mean()
    value = float(np.asarray(mean))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("distribution mean must be finite and non-negative")
    return value


def expected_stop_loss(distribution: Any, threshold: float) -> float:
    if isinstance(distribution, np.ndarray):
        sample = _nonnegative_finite(distribution, "distribution")
        if sample.size == 0:
            raise ValueError("empirical distribution cannot be empty")
        return float(np.maximum(sample - threshold, 0.0).mean())
    if hasattr(distribution, "stop_loss"):
        return float(distribution.stop_loss(threshold))
    if hasattr(distribution, "expect"):
        return float(distribution.expect(lambda x: np.maximum(x - threshold, 0.0)))
    raise TypeError("distribution must provide stop_loss(), expect(), or be a numpy sample")


def _expected_contract_cost(D_kw: float, distribution: Any, cfg: PolicyConfig) -> float:
    return cfg.demand_rate * D_kw + cfg.kappa * cfg.demand_rate * expected_stop_loss(
        distribution, cfg.alpha * D_kw
    )


def optimal_contract_from_distribution(
    distribution: Any, S_kva: float, cfg: PolicyConfig
) -> ContractDecision:
    capacity_array = _nonnegative_finite(S_kva, "S_kva")
    if capacity_array.ndim != 0 or float(capacity_array) <= 0.0:
        raise ValueError("S_kva must be a positive scalar")
    capacity = float(capacity_array)
    lower = cfg.delta * capacity
    upper = cfg.contract_upper_ratio * capacity
    rho = cfg.contract_quantile_probability

    if isinstance(distribution, np.ndarray):
        support = np.unique(_nonnegative_finite(distribution, "distribution"))
    elif hasattr(distribution, "support_points"):
        support = _nonnegative_finite(distribution.support_points, "support_points")
    else:
        support = None

    if support is not None:
        candidates = np.unique(np.clip(np.r_[lower, upper, support / cfg.alpha], lower, upper))
        costs = np.array([_expected_contract_cost(float(d), distribution, cfg) for d in candidates])
        D_kw = float(candidates[int(np.argmin(costs))])
    else:
        quantile = float(np.asarray(distribution.ppf(rho)))
        if not np.isfinite(quantile):
            raise ValueError("distribution quantile must be finite")
        D_kw = float(np.clip(quantile / cfg.alpha, lower, upper))

    if np.isclose(D_kw, lower, atol=1e-10):
        bound_status = "lower"
    elif np.isclose(D_kw, upper, atol=1e-10):
        bound_status = "upper"
    else:
        bound_status = "interior"
    return ContractDecision(
        D_kw=D_kw,
        expected_cost=_expected_contract_cost(D_kw, distribution, cfg),
        quantile_probability=rho,
        bound_status=bound_status,
    )


def _as_months(distributions_by_month: Any) -> list[Any]:
    if isinstance(distributions_by_month, (list, tuple)):
        if not distributions_by_month:
            raise ValueError("at least one monthly distribution is required")
        return list(distributions_by_month)
    return [distributions_by_month]


def select_mode(
    distributions_by_month: Any,
    S_kva: float,
    cfg: PolicyConfig,
    tie_epsilon: float = 0.0,
) -> ModeDecision:
    if tie_epsilon < 0.0 or not np.isfinite(tie_epsilon):
        raise ValueError("tie_epsilon must be finite and non-negative")
    months = _as_months(distributions_by_month)
    contract_decisions = [optimal_contract_from_distribution(month, S_kva, cfg) for month in months]
    costs = {
        CAPACITY: len(months) * float(capacity_fee(S_kva, cfg)),
        ACTUAL: sum(cfg.demand_rate * distribution_mean(month) for month in months),
        CONTRACT: sum(decision.expected_cost for decision in contract_decisions),
    }
    minimum = min(costs.values())
    tie_set = tuple(mode for mode, cost in costs.items() if cost <= minimum + tie_epsilon + 1e-12)
    selected_mode = min(tie_set, key=lambda mode: (costs[mode], mode))
    return ModeDecision(
        selected_mode=selected_mode,
        contract_D_kw_by_month=tuple(decision.D_kw for decision in contract_decisions),
        expected_cost=costs[selected_mode],
        mode_costs=costs,
        tie_set=tie_set,
        tie_epsilon=tie_epsilon,
    )


def evaluate_decision(decision: ModeDecision, truth_by_month: Any, S_kva: float, cfg: PolicyConfig) -> float:
    months = _as_months(truth_by_month)
    if decision.selected_mode == CAPACITY:
        return len(months) * float(capacity_fee(S_kva, cfg))
    if decision.selected_mode == ACTUAL:
        return sum(cfg.demand_rate * distribution_mean(month) for month in months)
    if len(decision.contract_D_kw_by_month) != len(months):
        raise ValueError("contract decision and truth must contain the same number of months")
    return sum(
        _expected_contract_cost(D_kw, month, cfg)
        for D_kw, month in zip(decision.contract_D_kw_by_month, months, strict=True)
    )
