from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy import stats


EXP = Path(__file__).resolve().parents[1]
ENGINE_ROOT = EXP / "code" / "legacy_engine"
LEGACY_ROOT = EXP / "code" / "legacy"
PROTOCOL_PATH = EXP / "config" / "exp1_protocol.json"


def _load_module(name: str, path: Path, import_root: Path | None = None) -> Any:
    if import_root is not None and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime() -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    engine = _load_module(
        "exp1_v12_engine",
        ENGINE_ROOT / "scripts" / "run_revision_experiment.py",
        ENGINE_ROOT / "src",
    )
    helper = _load_module("exp1_v12_no_upper", LEGACY_ROOT / "run_support_policy_audit.py")
    nb_module = _load_module("exp1_v12_nb_weibull", LEGACY_ROOT / "nonpoisson_gpd_challenge.py")
    settings = engine.load_settings()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    settings["seed_root"] = int(protocol["seed_root"])
    return engine, helper, nb_module, settings, protocol


def _rng(seed_root: int, *namespace: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed_root), *map(int, namespace)]))


def generate_history(
    dgp_id: str,
    replication_id: int,
    n_months: int,
    engine: Any,
    nb_module: Any,
    settings: dict[str, Any],
    seed_root: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate one path using V1.2's exact marginal mechanisms and extend to n months."""
    dgp_index = {"poisson_gpd": 1, "negative_binomial_weibull": 2}[dgp_id]
    rng = _rng(seed_root, dgp_index, int(replication_id), 0)
    if dgp_id == "poisson_gpd":
        dgp = engine.dgp_from_settings(settings)
        monthly, events = engine.generate_event_history(n_months, dgp, rng)
        return monthly, events, {
            "dgp_id": dgp_id,
            "replication_id": int(replication_id),
            "n_months": int(n_months),
            "seed_components": [int(seed_root), dgp_index, int(replication_id), 0],
            "tail_count_mean_empirical": float(events.groupby("month_id").size().mean() - dgp.baseline_events),
            "generator_source": "V1.2 R5 engine generate_event_history, called from the EXP1 paired seed stream",
        }

    spec = nb_module.DGP
    mean = float(spec["tail_count_mean_per_month"])
    size = float(spec["tail_count_negative_binomial_size"])
    probability = size / (size + mean)
    lower = float(spec["baseline_lower_kw"])
    threshold = float(spec["threshold_kw"])
    bulk_n = int(spec["baseline_events_per_month"])
    beta_a = float(spec["baseline_beta_a"])
    beta_b = float(spec["baseline_beta_b"])
    weibull_shape = float(spec["tail_excess_weibull_shape"])
    weibull_scale = float(spec["tail_excess_weibull_scale_kw"])
    monthly_rows: list[dict[str, float | int]] = []
    event_rows: list[dict[str, float | int]] = []
    tail_counts: list[int] = []
    for month_id in range(n_months):
        bulk = lower + (threshold - lower) * rng.beta(beta_a, beta_b, size=bulk_n)
        count = int(rng.negative_binomial(size, probability))
        tail = threshold + rng.weibull(weibull_shape, size=count) * weibull_scale
        values = np.r_[bulk, tail]
        monthly_rows.append({"month_id": month_id, "M_fixed15_kw": float(np.max(values))})
        tail_counts.append(count)
        event_rows.extend(
            {"month_id": month_id, "heat_id": event_id, "cluster_max_kw": float(value)}
            for event_id, value in enumerate(values)
        )
    return pd.DataFrame(monthly_rows), pd.DataFrame(event_rows), {
        "dgp_id": dgp_id,
        "replication_id": int(replication_id),
        "n_months": int(n_months),
        "seed_components": [int(seed_root), dgp_index, int(replication_id), 0],
        "tail_count_mean_empirical": float(np.mean(tail_counts)),
        "tail_count_variance_empirical": float(np.var(tail_counts, ddof=1)) if len(tail_counts) > 1 else None,
        "generator_source": "V1.2 nonpoisson_gpd_challenge.py DGP, extended to 60 months with the same NB(size=1, mean=30) and Weibull(1.5, 30000) parameterization",
    }


def build_oracle(dgp_id: str, engine: Any, nb_module: Any, settings: dict[str, Any]) -> Any:
    if dgp_id == "poisson_gpd":
        return engine.OracleDistribution(engine.dgp_from_settings(settings))
    return nb_module.NonPoissonWeibullOracle()


def build_tariff_cells(oracle: Any, engine: Any, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for family in protocol["tariff_families"]:
        policy = engine.Policy(
            family["id"],
            float(family["demand_rate_cny_kw_month"]),
            float(family["capacity_rate_cny_kva_month"]),
            float(family["alpha"]),
            float(family["kappa"]),
            float(family["delta"]),
            1.0,
        )
        p_star = float(family["p_star"])
        if not math.isclose(policy.contract_probability, p_star, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"Tariff p* mismatch in {family['id']}")
        for utilization in family["utilizations"]:
            u = float(utilization)
            capacity = float(oracle.mean() / u)
            oracle_decision = engine.optimized_decision(oracle, capacity, policy)
            cell_id = f"{family['id']}__u{u:.3f}"
            cells.append({
                "family_id": family["id"],
                "cell_id": cell_id,
                "utilization": u,
                "capacity_kva": capacity,
                "p_star": p_star,
                "policy": policy,
                "oracle_mode": oracle_decision["selected_mode"],
                "oracle_contract_d_kw": oracle_decision["contract_D_kw"],
                "oracle_cost_cny": oracle_decision["best_value_cny"],
                "oracle_margin_cny": oracle_decision["margin_cny"],
                "oracle_mode_costs": oracle_decision["mode_costs"],
            })
    modes = {cell["oracle_mode"] for cell in cells}
    if modes != {engine.ACTUAL, engine.CONTRACT, engine.CAPACITY}:
        raise AssertionError(f"Frozen cell design must contain all three oracle modes; got {modes}")
    return cells


@dataclass
class CandidateState:
    candidate: Any
    subthreshold: np.ndarray
    law: Any | None
    fit_exception: str | None = None


def fit_threshold_candidates(
    monthly: pd.DataFrame,
    events: pd.DataFrame,
    settings: dict[str, Any],
    engine: Any,
    helper: Any,
    seed_components: tuple[int, ...],
) -> tuple[list[CandidateState], dict[str, Any]]:
    months = np.sort(monthly["month_id"].unique().astype(int))
    holdout = float(settings["tail"]["holdout_fraction"])
    split = max(1, min(months.size - 1, int(np.floor(months.size * (1.0 - holdout)))))
    fit_months = months[:split]
    validation_months = months[split:]
    fit_events = events[events["month_id"].isin(fit_months)].copy()
    validation = monthly[monthly["month_id"].isin(validation_months)]["M_fixed15_kw"].to_numpy(float)
    g = engine.fit_process_tail.__globals__
    rng = _rng(int(settings["seed_root"]), *seed_components, 20, 1)
    bootstrap_rng = _rng(int(settings["seed_root"]), *seed_components, 20, 2)
    cfg = settings["tail"]
    states: list[CandidateState] = []
    exceptions: list[dict[str, Any]] = []
    for quantile in [float(q) for q in cfg["threshold_quantiles"]]:
        try:
            candidate, subthreshold = helper.no_upper_candidate(
                g,
                quantile,
                fit_events,
                fit_months,
                validation,
                int(settings["estimator_bootstrap_replicates"]["full"]),
                int(settings["validation_draws"]["full"]),
                int(cfg["minimum_exceedances_main"]),
                float(cfg["shape_ci_width_gate"]),
                rng,
                bootstrap_rng,
            )
            law = None
            if candidate.valid_main and np.isfinite(candidate.xi) and candidate.xi < 1.0:
                law = engine.TailProcessDistribution(
                    candidate.threshold,
                    candidate.xi,
                    candidate.sigma,
                    subthreshold,
                    candidate.count_model,
                    candidate.count_mean,
                    candidate.count_variance,
                    candidate.nb_size,
                    candidate.nb_probability,
                )
            states.append(CandidateState(candidate, np.asarray(subthreshold, dtype=float), law))
        except Exception as exc:  # retain candidate-level failure as data
            exceptions.append({"quantile": quantile, "error": f"{type(exc).__name__}: {exc}"})
    metadata = {
        "n_months": int(months.size),
        "fit_months": int(fit_months.size),
        "validation_months": int(validation_months.size),
        "fit_month_ids": fit_months.tolist(),
        "validation_month_ids": validation_months.tolist(),
        "n_fit_events": int(len(fit_events)),
        "n_validation_maxima": int(validation.size),
        "candidate_fit_exceptions": exceptions,
    }
    return states, metadata


def _main_candidates(states: list[CandidateState]) -> list[CandidateState]:
    return [
        state for state in states
        if state.law is not None
        and bool(state.candidate.valid_main)
        and np.isfinite(float(state.candidate.xi))
        and float(state.candidate.xi) < 1.0
    ]


def _score_fz0(law: Any, probability: float, observations: np.ndarray) -> float:
    q = float(law.ppf(probability))
    es = q + float(law.stop_loss(q)) / (1.0 - probability)
    if not np.isfinite(es) or es <= 0.0:
        return float("nan")
    # Upper-tail FZ0 score obtained by applying the lower-tail FZ0 score to -M.
    excess = np.maximum(observations - q, 0.0)
    return float(np.mean(excess / ((1.0 - probability) * es) + q / es + np.log(es) - 1.0))


def _score_tqcrps(law: Any, probability: float, observations: np.ndarray) -> float:
    levels = np.linspace(probability, 0.99, 41)
    quantiles = np.asarray([law.ppf(float(tau)) for tau in levels], dtype=float)
    if not np.isfinite(quantiles).all():
        return float("nan")
    delta = observations[:, None] - quantiles[None, :]
    pinball = np.maximum(levels[None, :] * delta, (levels[None, :] - 1.0) * delta)
    per_obs = 2.0 * np.trapezoid(pinball, levels, axis=1) / max(0.99 - probability, 1e-12)
    return float(np.mean(per_obs))


def _realized_bill(mode: str, contract_d: float, observation: float, cell: dict[str, Any], engine: Any) -> float:
    policy = cell["policy"]
    if mode == engine.ACTUAL:
        return float(policy.demand_rate * observation)
    if mode == engine.CAPACITY:
        return float(policy.capacity_rate * cell["capacity_kva"])
    return float(policy.demand_rate * (contract_d + policy.kappa * max(observation - policy.alpha * contract_d, 0.0)))


def _normalized_shape_penalty(candidates: list[CandidateState], weight: float) -> tuple[float, float]:
    xi = np.asarray([float(row.candidate.xi) for row in candidates], dtype=float)
    median = float(np.median(xi))
    mad = float(np.median(np.abs(xi - median)))
    scale = max(1.4826 * mad, 0.05)
    return median, float(weight / scale)


def _zscore(values: dict[float, float], higher_is_better: bool) -> dict[float, float]:
    finite = {q: (-float(value) if higher_is_better else float(value)) for q, value in values.items() if np.isfinite(value)}
    if not finite:
        return {}
    x = np.asarray(list(finite.values()), dtype=float)
    center = float(np.mean(x))
    scale = float(np.std(x, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return {q: 0.0 for q in finite}
    return {q: float(-(value - center) / scale) for q, value in finite.items()}


def score_and_select(
    states: list[CandidateState],
    metadata: dict[str, Any],
    monthly: pd.DataFrame,
    engine: Any,
    cells: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[dict[str, CandidateState | None], list[dict[str, Any]], dict[str, Any]]:
    valid = _main_candidates(states)
    validation_ids = metadata["validation_month_ids"]
    observations = monthly[monthly["month_id"].isin(validation_ids)]["M_fixed15_kw"].to_numpy(float)
    family_ids = list(dict.fromkeys(cell["family_id"] for cell in cells))
    p_by_family = {
        family: next(cell["p_star"] for cell in cells if cell["family_id"] == family)
        for family in family_ids
    }
    score_by_method: dict[str, dict[float, float]] = {
        "TAIL_LEGACY_LOG": {},
        "CV_LOG_ZPEN": {},
        "CV_FZ0_ZPEN": {},
        "CV_TQCRPS_ZPEN": {},
        "CV_COST_ZPEN": {},
    }
    candidate_records: list[dict[str, Any]] = []
    if not valid:
        for state in states:
            c = state.candidate
            candidate_records.append({
                "threshold_quantile": float(c.quantile), "threshold_kw": float(c.threshold),
                "xi": float(c.xi), "sigma_kw": float(c.sigma), "n_exceedances_train": int(c.exceedances),
                "shape_ci_width": float(c.shape_ci_width), "ks_pvalue": float(c.ks_pvalue),
                "valid_main_gate": bool(c.valid_main), "rejection_reasons": list(c.rejection_reasons),
                "validation_log_score": float(c.validation_log_score), "eligible_finite_mean": False,
            })
        fallback = _select_fallback(states, int(protocol["pot"]["minimum_fallback_exceedances"]))
        return ({name: fallback for name in score_by_method}, candidate_records, {"n_main_candidates": 0})

    median_xi, penalty_factor = _normalized_shape_penalty(valid, float(protocol["common_standardized_stability_penalty"]["weight"]))
    raw: dict[float, dict[str, float]] = {}
    for state in valid:
        c, law = state.candidate, state.law
        assert law is not None
        row: dict[str, float] = {"log_score": float(c.validation_log_score)}
        for family in family_ids:
            pstar = p_by_family[family]
            row[f"fz0::{family}"] = _score_fz0(law, pstar, observations)
            row[f"tqcrps::{family}"] = _score_tqcrps(law, pstar, observations)
        family_costs: list[float] = []
        for family in family_ids:
            family_rows: list[float] = []
            for cell in [x for x in cells if x["family_id"] == family]:
                action = engine.optimized_decision(law, float(cell["capacity_kva"]), cell["policy"])
                realized = [
                    _realized_bill(action["selected_mode"], float(action["contract_D_kw"]), float(y), cell, engine)
                    for y in observations
                ]
                family_rows.append(float(np.mean(realized)))
            family_costs.append(float(np.mean(family_rows)))
        row["cost_cv"] = float(np.mean(family_costs))
        raw[float(c.quantile)] = row

    # Preserve the V1.2 raw-score rule as a separate historical baseline.
    log_finite = {
        q: row["log_score"] for q, row in raw.items() if np.isfinite(row["log_score"])
    }
    legacy_candidates = [state for state in valid if float(state.candidate.quantile) in log_finite]
    if legacy_candidates:
        legacy_median = float(np.median([float(state.candidate.xi) for state in legacy_candidates]))
        legacy = max(
            legacy_candidates,
            key=lambda state: (
                float(state.candidate.validation_log_score)
                - float(protocol["common_standardized_stability_penalty"]["weight"])
                * abs(float(state.candidate.xi) - legacy_median),
                -float(state.candidate.quantile),
            ),
        )
    else:
        legacy = _select_fallback(states, int(protocol["pot"]["minimum_fallback_exceedances"]))
    score_by_method["TAIL_LEGACY_LOG"][float(legacy.candidate.quantile)] = 0.0 if legacy is not None else float("nan")

    criterion_specs = {
        "CV_LOG_ZPEN": ("log_score", True),
        "CV_FZ0_ZPEN": ("fz0", False),
        "CV_TQCRPS_ZPEN": ("tqcrps", False),
        "CV_COST_ZPEN": ("cost_cv", False),
    }
    selected: dict[str, CandidateState | None] = {"TAIL_LEGACY_LOG": legacy}
    for method, (criterion, higher_is_better) in criterion_specs.items():
        candidate_values: dict[float, float] = {}
        for q, row in raw.items():
            if criterion == "fz0":
                values = [row.get(f"fz0::{family}", float("nan")) for family in family_ids]
            elif criterion == "tqcrps":
                values = [row.get(f"tqcrps::{family}", float("nan")) for family in family_ids]
            else:
                values = [row.get(criterion, float("nan"))]
            candidate_values[q] = float(np.mean(values)) if values and np.isfinite(values).all() else float("nan")
        z = _zscore(candidate_values, higher_is_better)
        adjusted = {
            q: score - penalty_factor * abs(float(next(s.candidate.xi for s in valid if float(s.candidate.quantile) == q)) - median_xi)
            for q, score in z.items()
        }
        if adjusted:
            chosen_q = max(adjusted, key=lambda q: (adjusted[q], -q))
            selected[method] = next(s for s in valid if float(s.candidate.quantile) == chosen_q)
            score_by_method[method][chosen_q] = float(adjusted[chosen_q])
        else:
            selected[method] = _select_fallback(states, int(protocol["pot"]["minimum_fallback_exceedances"]))

    for state in states:
        c = state.candidate
        q = float(c.quantile)
        record: dict[str, Any] = {
            "threshold_quantile": q,
            "threshold_kw": float(c.threshold),
            "xi": float(c.xi),
            "sigma_kw": float(c.sigma),
            "n_exceedances_train": int(c.exceedances),
            "shape_ci_width": float(c.shape_ci_width),
            "ks_pvalue": float(c.ks_pvalue),
            "endpoint_kw": None if c.endpoint_kw is None else float(c.endpoint_kw),
            "count_model": str(c.count_model),
            "count_mean": float(c.count_mean),
            "count_variance": float(c.count_variance),
            "valid_main_gate": bool(c.valid_main),
            "rejection_reasons": list(c.rejection_reasons),
            "validation_log_score": float(c.validation_log_score),
            "eligible_finite_mean": bool(state in valid),
        }
        if q in raw:
            record.update({key: (float(value) if np.isscalar(value) else value) for key, value in raw[q].items()})
        candidate_records.append(record)
    return selected, candidate_records, {
        "n_main_candidates": len(valid),
        "n_candidates_total": len(states),
        "median_xi_for_common_penalty": median_xi,
        "robust_xi_scale": (1.0 / penalty_factor * float(protocol["common_standardized_stability_penalty"]["weight"])) if penalty_factor else None,
    }


def _select_fallback(states: list[CandidateState], minimum: int) -> CandidateState | None:
    eligible = [state for state in states if int(state.candidate.exceedances) >= minimum]
    if not eligible:
        return None
    return max(eligible, key=lambda state: (int(state.candidate.exceedances), -float(state.candidate.quantile)))


def refit_selected(
    selected: CandidateState | None,
    monthly: pd.DataFrame,
    events: pd.DataFrame,
    engine: Any,
    settings: dict[str, Any],
    fallback_minimum: int,
) -> tuple[Any, dict[str, Any]]:
    maxima = monthly["M_fixed15_kw"].to_numpy(float)
    if selected is None:
        return engine.DiscreteDistribution(maxima), {
            "fit_status": "FALLBACK_EMPIRICAL",
            "fallback_reason": "no_threshold_with_minimum_fallback_exceedances",
            "selected_threshold_quantile": None,
            "threshold_kw": None,
            "xi_hat": None,
            "sigma_hat_kw": None,
            "n_exceedances_final": 0,
            "refit_used_all_months": True,
        }
    c = selected.candidate
    threshold = float(c.threshold)  # numerical training threshold is fixed for the all-history refit
    values = events["cluster_max_kw"].to_numpy(float)
    excesses = values[values > threshold] - threshold
    if excesses.size < fallback_minimum:
        return engine.DiscreteDistribution(maxima), {
            "fit_status": "FALLBACK_EMPIRICAL",
            "fallback_reason": "insufficient_exceedances_after_full_history_refit",
            "selected_threshold_quantile": float(c.quantile),
            "threshold_kw": threshold,
            "xi_hat": None,
            "sigma_hat_kw": None,
            "n_exceedances_final": int(excesses.size),
            "refit_used_all_months": True,
        }
    cfg = settings["tail"]
    force_exponential = not bool(c.valid_main) or not np.isfinite(float(c.xi)) or float(c.xi) >= 1.0
    try:
        xi, _, sigma = (float(v) for v in stats.genpareto.fit(excesses, floc=0.0))
    except Exception:
        xi, sigma = float("nan"), float("nan")
    invalid = (
        force_exponential
        or not np.isfinite(xi)
        or not np.isfinite(sigma)
        or sigma <= 0.0
        or xi >= 1.0
        or (xi < 0.0 and threshold - sigma / xi < float(values.max()) - 1e-8)
    )
    fallback_reason = None
    fit_status = "PASS"
    if invalid:
        xi = 0.0
        sigma = float(excesses.mean())
        fit_status = "FALLBACK_EXPONENTIAL"
        fallback_reason = "selection_or_full_refit_failed_finite_mean_or_support_gate"
    months = np.sort(monthly["month_id"].unique().astype(int))
    counts = (
        events.assign(exceed=events["cluster_max_kw"] > threshold)
        .groupby("month_id")["exceed"]
        .sum()
        .reindex(months, fill_value=0)
        .to_numpy(dtype=int)
    )
    count_model, count_mean, count_variance, nb_size, nb_probability = engine.fit_process_tail.__globals__["_count_parameters"](counts)
    below = events[events["cluster_max_kw"] <= threshold]
    baseline = (
        below.groupby("month_id")["cluster_max_kw"]
        .max()
        .reindex(months, fill_value=0.0)
        .to_numpy(float)
    )
    law = engine.TailProcessDistribution(
        threshold,
        xi,
        sigma,
        baseline,
        count_model,
        count_mean,
        count_variance,
        nb_size,
        nb_probability,
    )
    return law, {
        "fit_status": fit_status,
        "fallback_reason": fallback_reason,
        "selected_threshold_quantile": float(c.quantile),
        "threshold_kw": threshold,
        "selection_xi_hat": float(c.xi),
        "xi_hat": xi,
        "sigma_hat_kw": sigma,
        "n_exceedances_final": int(excesses.size),
        "count_model": count_model,
        "count_mean": count_mean,
        "count_variance": count_variance,
        "finite_mean_gate_pass": bool(xi < 1.0),
        "finite_variance_flag": bool(xi < 0.5),
        "refit_used_all_months": True,
    }


def tariff_regret(law: Any, oracle: Any, cells: list[dict[str, Any]], engine: Any) -> tuple[list[dict[str, Any]], float, float]:
    rows: list[dict[str, Any]] = []
    family_values: dict[str, list[float]] = {}
    raw_family_values: dict[str, list[float]] = {}
    for cell in cells:
        decision = engine.optimized_decision(law, float(cell["capacity_kva"]), cell["policy"])
        true_cost = engine.true_cost_of_estimated_action(
            decision["selected_mode"], float(decision["contract_D_kw"]),
            oracle, float(cell["capacity_kva"]), cell["policy"],
        )
        regret = float(true_cost - float(cell["oracle_cost_cny"]))
        norm = regret / max(float(cell["oracle_cost_cny"]), 1e-12)
        rows.append({
            "family_id": cell["family_id"],
            "cell_id": cell["cell_id"],
            "utilization": float(cell["utilization"]),
            "p_star": float(cell["p_star"]),
            "oracle_mode": cell["oracle_mode"],
            "selected_mode": decision["selected_mode"],
            "mode_correct": bool(decision["selected_mode"] == cell["oracle_mode"]),
            "contract_D_kw": float(decision["contract_D_kw"]),
            "oracle_contract_D_kw": float(cell["oracle_contract_d_kw"]),
            "estimated_action_true_expected_bill_cny": float(true_cost),
            "oracle_expected_bill_cny": float(cell["oracle_cost_cny"]),
            "regret_cny_month": regret,
            "relative_regret": norm,
            "oracle_mode_margin_cny": float(cell["oracle_margin_cny"]),
        })
        family_values.setdefault(cell["family_id"], []).append(norm)
        raw_family_values.setdefault(cell["family_id"], []).append(regret)
    family_means = [float(np.mean(v)) for v in family_values.values()]
    raw_family_means = [float(np.mean(v)) for v in raw_family_values.values()]
    return rows, float(np.mean(family_means)), float(np.mean(raw_family_means))


def observed_validation_cost_score(
    candidate_law: Any,
    observations: np.ndarray,
    cells: list[dict[str, Any]],
    engine: Any,
) -> float:
    family_means: list[float] = []
    for family_id in dict.fromkeys(cell["family_id"] for cell in cells):
        cell_means: list[float] = []
        for cell in [row for row in cells if row["family_id"] == family_id]:
            action = engine.optimized_decision(candidate_law, float(cell["capacity_kva"]), cell["policy"])
            bills = [
                _realized_bill(action["selected_mode"], float(action["contract_D_kw"]), float(y), cell, engine)
                for y in observations
            ]
            cell_means.append(float(np.mean(bills)))
        family_means.append(float(np.mean(cell_means)))
    return float(np.mean(family_means))


def validate_legacy_equivalence(
    monthly: pd.DataFrame,
    events: pd.DataFrame,
    states: list[CandidateState],
    selection: dict[str, CandidateState | None],
    settings: dict[str, Any],
    engine: Any,
    helper: Any,
    replication_id: int,
) -> dict[str, Any]:
    """Compare the EXP1 raw LOG choice/refit to the unchanged V1.2 no-upper helper once."""
    reference, reference_diag, _ = helper.fit_tail(
        engine,
        monthly,
        events,
        settings,
        int(replication_id),
        "no_upper",
        float(np.finfo(float).max),
    )
    ours_state = selection["TAIL_LEGACY_LOG"]
    ours, ours_diag = refit_selected(
        ours_state, monthly, events, engine, settings,
        int(settings["tail"]["minimum_exceedances_fallback"]),
    )
    ref_q = reference_diag.get("selected_threshold_quantile")
    our_q = ours_diag.get("selected_threshold_quantile")
    same_q = (ref_q is None and our_q is None) or (
        ref_q is not None and our_q is not None and math.isclose(float(ref_q), float(our_q), abs_tol=0.0, rel_tol=0.0)
    )
    checks = {
        "reference_selected_quantile": ref_q,
        "exp1_selected_quantile": our_q,
        "same_selected_quantile": same_q,
        "reference_fit_status": reference_diag.get("fit_status"),
        "exp1_fit_status": ours_diag.get("fit_status"),
        "reference_xi_hat": reference_diag.get("xi_hat"),
        "exp1_xi_hat": ours_diag.get("xi_hat"),
        "reference_sigma_hat_kw": reference_diag.get("sigma_hat_kw"),
        "exp1_sigma_hat_kw": ours_diag.get("sigma_hat_kw"),
        "reference_candidate_count": len((reference_diag.get("candidate_diagnostics") or [])),
        "exp1_candidate_count": len(states),
    }
    if not same_q:
        raise AssertionError(f"EXP1 legacy reproduction disagrees with V1.2 no-upper helper: {checks}")
    for field in ("xi_hat", "sigma_hat_kw"):
        a = reference_diag.get(field)
        b = ours_diag.get(field)
        if a is not None and b is not None and not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-8):
            raise AssertionError(f"EXP1 legacy parameter mismatch for {field}: {checks}")
    return checks

