#!/usr/bin/env python3
"""Five-method NB-Weibull challenge with four-fold CV-CRPS TAIL selection.

This is a new DGP challenge, distinct from the earlier KDE-versus-EMP-only
simulation and from the old archived/failed experiments.  It changes the
baseline Poisson count and GPD excess assumptions at the same time:

* each month has 200 baseline Beta bulk events on [45,000, 80,000] kW;
* the number of threshold events is negative binomial (mean 30, size 1);
* threshold excesses are Weibull (shape 1.5, scale 30,000 kW), not GPD.

All five fitted methods are the frozen project implementations: TAIL,
EVENT-EMP, GEV, KDE, EMP.  The monthly-maximum oracle is analytic, allowing
exact tariff-mode and regret evaluation.  Per-replication seeds are stable
across worker count and chunk order.  Output is chunked and resumable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

# Keep each process-pool worker to one numerical-library thread.  Without this
# cap, NumPy's OpenBLAS defaults to all 16 local CPUs in every worker, causing
# severe oversubscription (and potentially worker termination) in a 10-worker
# run.  Set these before importing NumPy/SciPy/Pandas so spawned workers inherit
# the same deterministic limit.
NUMERICAL_THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
for _thread_env_name in NUMERICAL_THREAD_ENV:
    os.environ[_thread_env_name] = "1"

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, optimize, stats
import yaml


HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
DEFAULT_BASE = ROOT / "exp1" / "code" / "legacy_engine"
DEFAULT_OUT = ROOT / "CSSC_REVIEW_REVISION_20261112" / "nb_weibull_cv_crps_20261115"
METHODS = ("TAIL", "EVENT-EMP", "GEV", "KDE", "EMP")
ORACLE_REGION = {
    "actual_maximum_demand": "actual",
    "contracted_maximum_demand": "contract",
    "capacity": "capacity",
}

DGP = {
    "name": "negative_binomial_size1_counts_weibull_excess_with_active_bulk",
    "baseline_events_per_month": 200,
    "baseline_lower_kw": 45_000.0,
    "threshold_kw": 80_000.0,
    "baseline_beta_a": 8.0,
    "baseline_beta_b": 2.0,
    "tail_count_mean_per_month": 30.0,
    "tail_count_negative_binomial_size": 1.0,
    "tail_excess_distribution": "Weibull(shape=1.5, scale=30000 kW)",
    "tail_excess_weibull_shape": 1.5,
    "tail_excess_weibull_scale_kw": 30_000.0,
    "non_poisson_certificate": "tail count variance = mean + mean^2/size = 930, 31 times the Poisson variance at the same mean; zero-tail month probability = 1/31",
    "non_gpd_certificate": "Weibull survival exp(-(x/scale)^shape) with shape 1.5 is not a generalized Pareto survival law; the survival log has nonzero curvature in x, unlike exponential xi=0 and unlike log(1+xi*x/sigma) GPD log-survival",
    "baseline_relation": "Retains the fixed 200-event Beta bulk and threshold; replaces the Poisson count by NB(size=1, mean=30) and GPD excesses by Weibull(shape=1.5, scale=30000). In zero-tail months (probability 1/31), the baseline bulk determines the monthly maximum. This is a joint misspecification challenge; count and excess-law effects are not separately identified.",
}

SEED_ROOT = 20261002
SEED_NAMESPACE = 95021
N_MONTHS = 24
GRID_STEP_KW = 10.0
GRID_MAX_KW = 1_000_000.0

_ENGINE: Any = None
_SETTINGS: dict[str, Any] | None = None
_ORACLE: Any = None
_PLAN: list[dict[str, Any]] | None = None
_SUPPORT: Any = None
_BASE_REPO: str | None = None
_ACTIVE_SEED_ROOT = SEED_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class NonPoissonWeibullOracle:
    """Exact monthly-max CDF and numerically integrated stop-loss oracle."""

    def __init__(self, grid_step_kw: float = GRID_STEP_KW,
                 grid_max_kw: float = GRID_MAX_KW) -> None:
        self.threshold = float(DGP["threshold_kw"])
        self.lower = float(DGP["baseline_lower_kw"])
        self.bulk_n = int(DGP["baseline_events_per_month"])
        self.a = float(DGP["baseline_beta_a"])
        self.b = float(DGP["baseline_beta_b"])
        self.tail_mean = float(DGP["tail_count_mean_per_month"])
        self.tail_size = float(DGP["tail_count_negative_binomial_size"])
        self.tail_shape = float(DGP["tail_excess_weibull_shape"])
        self.tail_scale = float(DGP["tail_excess_weibull_scale_kw"])
        self.tail_p = self.tail_size / (self.tail_size + self.tail_mean)
        self.p_zero = self.tail_p ** self.tail_size
        self.grid_step_kw = float(grid_step_kw)
        self.grid_max_kw = float(grid_max_kw)
        self._mean = self._integrate_exact(0.0)
        # Dense monotone interpolation accelerates the hundreds of thousands
        # of true-action costs while retaining sub-cent tariff-value accuracy.
        x = np.arange(0.0, self.grid_max_kw + self.grid_step_kw,
                      self.grid_step_kw, dtype=np.float64)
        survival = 1.0 - np.asarray(self.cdf(x), dtype=np.float64)
        survival = np.clip(survival, 0.0, 1.0)
        dx = np.diff(x)
        area = 0.5 * (survival[:-1] + survival[1:]) * dx
        h = np.r_[np.cumsum(area[::-1])[::-1], 0.0]
        self._stoploss_interp = interpolate.PchipInterpolator(x, h, extrapolate=False)
        self._grid_x = x
        self._grid_h = h

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(values)
        positive = values >= self.lower
        below = positive & (values < self.threshold)
        if np.any(below):
            z = np.clip((values[below] - self.lower) /
                        (self.threshold - self.lower), 0.0, 1.0)
            bulk_cdf = stats.beta.cdf(z, self.a, self.b) ** self.bulk_n
            out[below] = self.p_zero * bulk_cdf
        above = values >= self.threshold
        if np.any(above):
            excess = np.maximum(values[above] - self.threshold, 0.0)
            tail_survival = np.exp(-np.power(excess / self.tail_scale,
                                             self.tail_shape))
            # PGF of NB(r, mean): E[z^N]=(1 + mean/r*(1-z))^-r.
            out[above] = np.power(
                1.0 + (self.tail_mean / self.tail_size) * tail_survival,
                -self.tail_size,
            )
        out = np.clip(out, 0.0, 1.0)
        return float(out) if values.ndim == 0 else out

    def ppf(self, probability: float) -> float:
        p = float(probability)
        if not 0.0 < p < 1.0:
            raise ValueError("probability must be in (0,1)")
        if p <= self.p_zero:
            # This probability range lies in the baseline maximum component.
            return float(optimize.brentq(lambda z: self.cdf(z) - p,
                                         self.lower, self.threshold))
        high = self.threshold + self.tail_scale
        while self.cdf(high) < p:
            high = self.threshold + 2.0 * (high - self.threshold)
        return float(optimize.brentq(lambda z: self.cdf(z) - p,
                                     self.threshold, high))

    def _integrate_exact(self, threshold: float) -> float:
        t = max(0.0, float(threshold))
        fn = lambda z: 1.0 - float(self.cdf(z))
        # For the chosen Weibull tail, survival is below machine precision by
        # 1,000,000 kW. Finite-interval quadrature avoids the unreliable
        # infinite-interval transform when survival stays near one to 100kW.
        upper = max(self.grid_max_kw, 1_000_000.0)
        if t >= upper:
            return 0.0
        if t < self.threshold:
            below = integrate.quad(fn, t, self.threshold,
                                   epsabs=1e-7, epsrel=2e-10, limit=200)[0]
            above = integrate.quad(fn, self.threshold, upper,
                                   epsabs=1e-7, epsrel=2e-10, limit=300)[0]
            return float(below + above)
        return float(integrate.quad(fn, t, upper,
                                    epsabs=1e-7, epsrel=2e-10, limit=300)[0])

    def stop_loss(self, threshold: float) -> float:
        t = max(0.0, float(threshold))
        if t >= self.grid_max_kw:
            return self._integrate_exact(t)
        value = float(self._stoploss_interp(t))
        return max(0.0, value)

    def mean(self) -> float:
        return self._mean

    def validate_grid(self) -> dict[str, Any]:
        grid = [0.0, 35_000.0, 79_000.0, 80_000.0, 80_001.0,
                95_000.0, 125_000.0, 200_000.0, 450_000.0]
        errors = []
        for t in grid:
            exact = self._integrate_exact(t)
            errors.append(abs(self.stop_loss(t) - exact))
        return {
            "grid_step_kw": self.grid_step_kw,
            "grid_max_kw": self.grid_max_kw,
            "oracle_mean_kw_exact_quad": self._mean,
            "stoploss_grid_vs_quad_max_abs_error_kw": max(errors),
            "validation_thresholds_kw": grid,
            "stoploss_validation_errors_kw": errors,
        }


def load_engine(base_repo: Path) -> Any:
    script = base_repo / "scripts" / "run_revision_experiment.py"
    source = base_repo / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("nonpoisson_revision_engine", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tariff engine {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_support_policy_module() -> Any:
    """Load the audited no-upper TAIL/GEV fit routines used by the 10k reanalysis."""
    source = HERE.parent / "run_primary_no_nominal_ks.py"
    spec = importlib.util.spec_from_file_location("challenge_no_upper_support_policy", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load corrected no-upper fit routines: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_settings(base_repo: Path) -> dict[str, Any]:
    with (base_repo / "configs" / "revision.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)["revision"]


def build_plan(settings: dict[str, Any], engine: Any,
               oracle: NonPoissonWeibullOracle) -> list[dict[str, Any]]:
    tariff = settings["tariff"]
    plan: list[dict[str, Any]] = []
    for policy_doc in tariff["policies"]:
        policy = engine.Policy(
            str(policy_doc["id"]), float(tariff["demand_rate"]),
            float(tariff["capacity_rate"]), float(policy_doc["alpha"]),
            float(policy_doc["kappa"]), float(tariff["delta"]),
            float(tariff["contract_upper_ratio"]),
        )
        for cell in policy_doc["cells"]:
            u = float(cell["utilization"])
            s_kva = oracle.mean() / u
            decision = engine.optimized_decision(oracle, s_kva, policy)
            plan.append({
                "policy": policy,
                "policy_id": policy.policy_id,
                "cell_id": str(cell["id"]),
                "utilization": u,
                "S_kva": s_kva,
                "oracle": decision,
            })
    return plan


def preflight(base_repo: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    engine = load_engine(base_repo)
    settings = read_settings(base_repo)
    oracle = NonPoissonWeibullOracle()
    plan = build_plan(settings, engine, oracle)
    modes = {str(x["oracle"]["selected_mode"]) for x in plan}
    expected = set(ORACLE_REGION)
    frame = pd.DataFrame([{
        "cell_id": x["cell_id"], "policy_id": x["policy_id"],
        "utilization": x["utilization"], "S_kva": x["S_kva"],
        "oracle_mode": x["oracle"]["selected_mode"],
        "oracle_region": ORACLE_REGION[x["oracle"]["selected_mode"]],
        "oracle_margin_cny": x["oracle"]["margin_cny"],
        "oracle_contract_D_kw": x["oracle"]["contract_D_kw"],
        "p_star": x["policy"].contract_probability,
        "q_p_star_kw": oracle.ppf(x["policy"].contract_probability),
        "mean_kw": oracle.mean(),
        "actual_cost_cny": x["oracle"]["mode_costs"][engine.ACTUAL],
        "contract_cost_cny": x["oracle"]["mode_costs"][engine.CONTRACT],
        "capacity_cost_cny": x["oracle"]["mode_costs"][engine.CAPACITY],
    } for x in plan]).sort_values("cell_id").reset_index(drop=True)
    record = {
        "status": "PASS" if modes == expected and len(frame) == 9 else "FAIL",
        "experiment": "five-method non-Poisson-count and non-GPD-excess misspecification challenge",
        "dgp": DGP,
        "oracle_regions": sorted({str(x) for x in frame.oracle_region}),
        "oracle_mode_counts_by_cell": frame.groupby("oracle_region").size().astype(int).to_dict(),
        "tariff_cells": int(len(frame)),
        "methods": list(METHODS),
        "methods_are_frozen_project_estimators": True,
        "grid_validation": oracle.validate_grid(),
        "tariff_source": str(base_repo / "configs" / "revision.yaml"),
        "tariff_source_sha256": sha256(base_repo / "configs" / "revision.yaml"),
        "engine_source": str(base_repo / "scripts" / "run_revision_experiment.py"),
        "engine_source_sha256": sha256(base_repo / "scripts" / "run_revision_experiment.py"),
        "no_upper_fit_source": str(HERE.parent / "run_primary_no_nominal_ks.py"),
        "no_upper_fit_source_sha256": sha256(HERE.parent / "run_primary_no_nominal_ks.py"),
        "support_policy": "no_upper; TAIL uses fit_tail(kind='no_upper') hooks; GEV uses make_no_upper_gev/NonnegativeGEV; no oracle-derived or artificial upper bound enters either fit",
        "all_three_oracle_modes_present": modes == expected,
    }
    if record["grid_validation"]["stoploss_grid_vs_quad_max_abs_error_kw"] > 1e-3:
        raise RuntimeError(
            "oracle stop-loss interpolation exceeds the 1e-3 kW preflight tolerance: "
            f"{record['grid_validation']['stoploss_grid_vs_quad_max_abs_error_kw']}"
        )
    if record["status"] != "PASS":
        raise RuntimeError(f"preflight did not cover all three oracle modes: {record}")
    return record, frame


def _worker_init(base_repo_str: str, seed_root: int) -> None:
    global _ENGINE, _SETTINGS, _ORACLE, _PLAN, _SUPPORT, _BASE_REPO
    global _ACTIVE_SEED_ROOT
    _ACTIVE_SEED_ROOT = int(seed_root)
    base_repo = Path(base_repo_str)
    _BASE_REPO = str(base_repo.resolve())
    _ENGINE = load_engine(base_repo)
    _SETTINGS = read_settings(base_repo)
    _SETTINGS["seed_root"] = _ACTIVE_SEED_ROOT
    _SUPPORT = load_support_policy_module()
    _ORACLE = NonPoissonWeibullOracle()
    _PLAN = build_plan(_SETTINGS, _ENGINE, _ORACLE)


def _worker_init_smoke_task(task_id: int) -> dict[str, Any]:
    """Return initialized worker identity and BLAS thread limits."""
    time.sleep(0.5)  # hold a slot long enough to observe every initialized process
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except Exception as exc:
        pools = [{"diagnostic_error": str(exc)}]
    return {
        "task_id": int(task_id),
        "pid": os.getpid(),
        "engine_loaded": _ENGINE is not None,
        "support_module_loaded": _SUPPORT is not None,
        "oracle_mean_kw": float(_ORACLE.mean()) if _ORACLE is not None else None,
        "thread_pools": pools,
        "thread_environment": {k: os.environ.get(k) for k in NUMERICAL_THREAD_ENV},
    }


def _generate_history(replication_id: int, oracle: NonPoissonWeibullOracle
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seed = np.random.SeedSequence([_ACTIVE_SEED_ROOT, SEED_NAMESPACE, int(replication_id)])
    rng = np.random.default_rng(seed)
    tail_mean = float(DGP["tail_count_mean_per_month"])
    tail_size = float(DGP["tail_count_negative_binomial_size"])
    tail_prob = tail_size / (tail_size + tail_mean)
    lower = float(DGP["baseline_lower_kw"])
    threshold = float(DGP["threshold_kw"])
    baseline_n = int(DGP["baseline_events_per_month"])
    baseline_a = float(DGP["baseline_beta_a"])
    baseline_b = float(DGP["baseline_beta_b"])
    w_shape = float(DGP["tail_excess_weibull_shape"])
    w_scale = float(DGP["tail_excess_weibull_scale_kw"])
    maxima = np.empty(N_MONTHS, dtype=np.float64)
    low_counts = np.empty(N_MONTHS, dtype=np.int32)
    tail_counts = np.empty(N_MONTHS, dtype=np.int32)
    event_rows: list[dict[str, Any]] = []
    for month in range(N_MONTHS):
        bulk = lower + (threshold - lower) * rng.beta(baseline_a, baseline_b,
                                                       size=baseline_n)
        count = int(rng.negative_binomial(tail_size, tail_prob))
        tail = threshold + rng.weibull(w_shape, size=count) * w_scale
        values = np.r_[bulk, tail]
        maxima[month] = float(np.max(values))
        low_counts[month] = int(count == 0)
        tail_counts[month] = count
        event_rows.extend({"month_id": month, "heat_id": i,
                           "cluster_max_kw": float(value)}
                          for i, value in enumerate(values))
    months = pd.DataFrame({
        "replication_id": int(replication_id),
        "month_id": np.arange(N_MONTHS, dtype=np.int16),
        "M_fixed15_kw": maxima,
        "tail_event_count": tail_counts,
        "tail_count_zero": low_counts,
    })
    seed_material = maxima.astype("<f8", copy=False).tobytes()
    count_material = tail_counts.astype("<i4", copy=False).tobytes()
    seed_record = {
        "replication_id": int(replication_id),
        "seed_entropy": json.dumps([_ACTIVE_SEED_ROOT, SEED_NAMESPACE, int(replication_id)]),
        "bit_generator": "PCG64",
        "monthly_maxima_sha256_f64le": hashlib.sha256(seed_material).hexdigest(),
        "tail_counts_sha256_i32le": hashlib.sha256(count_material).hexdigest(),
        "total_tail_events": int(tail_counts.sum()),
        "tail_count_mean_observed": float(tail_counts.mean()),
        "tail_count_variance_observed": float(tail_counts.var(ddof=1)),
        "tail_month_zero_frequency": float(np.mean(tail_counts == 0)),
    }
    events = pd.DataFrame(event_rows)
    return months, events, seed_record


def _history_task(replication_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                 list[dict[str, Any]], dict[str, Any]]:
    assert _ENGINE is not None and _SETTINGS is not None and _ORACLE is not None
    assert _PLAN is not None and _SUPPORT is not None
    month_df, event_df, seed_record = _generate_history(replication_id, _ORACLE)
    tail, tail_diag, tail_candidate_diag = _SUPPORT.fit_tail(
        _ENGINE, month_df, event_df, _SETTINGS, replication_id, "no_upper", None,
    )
    event_emp, event_diag = _ENGINE.fit_event_empirical(event_df)
    maxima = month_df["M_fixed15_kw"].to_numpy(float)
    try:
        gev, gev_diag = _SUPPORT.make_no_upper_gev(_ENGINE, maxima)
    except Exception as exc:
        gev, gev_diag = _ENGINE.DiscreteDistribution(maxima), {
            "fit_status": "FALLBACK_EMPIRICAL", "fallback_reason": str(exc),
            "support_policy": "no_upper",
        }
    kde, kde_diag = _ENGINE.fit_kde_exact(month_df["M_fixed15_kw"].to_numpy(float))
    emp = _ENGINE.DiscreteDistribution(month_df["M_fixed15_kw"].to_numpy(float))
    builds = [
        ("TAIL", tail, tail_diag, tail_candidate_diag),
        ("EVENT-EMP", event_emp, event_diag, {}),
        ("GEV", gev, gev_diag, {}),
        ("KDE", kde, kde_diag, {}),
        ("EMP", emp, {"fit_status": "PASS"}, {}),
    ]
    decision_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for method, dist, diagnostic, full_diagnostic in builds:
        law_parameters = _SUPPORT.law_params(dist) if method in {"TAIL", "GEV"} else {
            "class": type(dist).__name__,
            "support_policy": "no artificial upper bound",
            "mean_kw": float(dist.mean()),
        }
        law_check = _independent_law_checks(method, dist, diagnostic, maxima,
                                             _SUPPORT)
        diagnostic_rows.append({
            "replication_id": int(replication_id),
            "method_id": method,
            "fit_status": str(diagnostic.get("fit_status", "PASS")),
            "fallback_reason": diagnostic.get("fallback_reason", diagnostic.get("reason")),
            "threshold_kw": diagnostic.get("threshold_kw"),
            "selected_threshold_quantile": diagnostic.get("selected_threshold_quantile"),
            "xi_hat": diagnostic.get("xi_hat", diagnostic.get("selection_xi_hat")),
            "sigma_hat_kw": diagnostic.get("sigma_hat_kw"),
            "count_model": diagnostic.get("count_model"),
            "count_mean": diagnostic.get("count_mean"),
            "count_variance": diagnostic.get("count_variance"),
            "gev_xi_hat": diagnostic.get("xi_hat") if method == "GEV" else None,
            "gev_truncation_probability": diagnostic.get("truncation_probability"),
            "kde_bandwidth_kw": diagnostic.get("bandwidth_kw"),
            "n_events": int(len(event_df)) if method == "EVENT-EMP" else None,
            "diagnostics_json": json.dumps(diagnostic, sort_keys=True, default=str),
            "candidate_diagnostics_json": json.dumps(full_diagnostic, sort_keys=True, default=str),
            "law_parameters_json": json.dumps(law_parameters, sort_keys=True, default=str),
            "independent_law_checks_json": json.dumps(law_check, sort_keys=True, default=str),
            "support_policy": "no_upper" if method in {"TAIL", "GEV"} else "not_applicable",
        })
        mean_est = float(dist.mean())
        for cell in _PLAN:
            policy = cell["policy"]
            oracle_decision = cell["oracle"]
            est = _ENGINE.optimized_decision(dist, float(cell["S_kva"]), policy)
            selected = str(est["selected_mode"])
            true_action_cost = _ENGINE.true_cost_of_estimated_action(
                selected, float(est["contract_D_kw"]), _ORACLE,
                float(cell["S_kva"]), policy,
            )
            regret = float(true_action_cost - oracle_decision["best_value_cny"])
            tol = 1e-7 * max(1.0, abs(float(oracle_decision["best_value_cny"])))
            if regret < -tol:
                raise ArithmeticError(f"negative regret={regret}: replication={replication_id}, method={method}, cell={cell['cell_id']}")
            regret = max(0.0, regret)
            mode_loss = float(oracle_decision["mode_costs"][selected] -
                              oracle_decision["best_value_cny"])
            within_loss = 0.0
            if selected == _ENGINE.CONTRACT:
                within_loss = float(
                    policy.demand_rate
                    * (float(est["contract_D_kw"])
                       + policy.kappa * _ORACLE.stop_loss(
                           policy.alpha * float(est["contract_D_kw"])))
                    - oracle_decision["mode_costs"][_ENGINE.CONTRACT]
                )
                if within_loss < -tol:
                    raise ArithmeticError(f"negative within-contract loss={within_loss}")
                within_loss = max(0.0, within_loss)
            if abs(regret - mode_loss - within_loss) > max(1e-5, tol):
                raise ArithmeticError("regret decomposition failed")
            oracle_mode = str(oracle_decision["selected_mode"])
            est_cost = est["mode_costs"]
            true_costs = oracle_decision["mode_costs"]
            decision_rows.append({
                "replication_id": int(replication_id),
                "method_id": method,
                "fit_status": str(diagnostic.get("fit_status", "PASS")),
                "policy_id": cell["policy_id"],
                "cell_id": cell["cell_id"],
                "utilization": float(cell["utilization"]),
                "S_kva": float(cell["S_kva"]),
                "oracle_mode": oracle_mode,
                "oracle_region": ORACLE_REGION[oracle_mode],
                "selected_mode": selected,
                "action_correct": selected == oracle_mode,
                "oracle_margin_cny": float(oracle_decision["margin_cny"]),
                "oracle_best_value_cny": float(oracle_decision["best_value_cny"]),
                "estimated_best_value_cny": float(est["best_value_cny"]),
                "oracle_contract_D_kw": float(oracle_decision["contract_D_kw"]),
                "estimated_contract_D_kw": float(est["contract_D_kw"]),
                "contract_D_error_kw": float(est["contract_D_kw"] - oracle_decision["contract_D_kw"]),
                "estimated_mean_kw": mean_est,
                "true_mean_kw": float(_ORACLE.mean()),
                "mean_error_kw": mean_est - float(_ORACLE.mean()),
                "capacity_value_error_cny": float(est_cost[_ENGINE.CAPACITY] - true_costs[_ENGINE.CAPACITY]),
                "actual_value_error_cny": float(est_cost[_ENGINE.ACTUAL] - true_costs[_ENGINE.ACTUAL]),
                "contract_value_error_cny": float(est_cost[_ENGINE.CONTRACT] - true_costs[_ENGINE.CONTRACT]),
                "selected_action_true_cost_cny": float(true_action_cost),
                "regret_cny_per_month": regret,
                "mode_selection_regret_cny": mode_loss,
                "within_contract_regret_cny": within_loss,
                "regret_decomposition_error_cny": regret - mode_loss - within_loss,
                "oracle_capacity_cost_cny": float(true_costs[_ENGINE.CAPACITY]),
                "oracle_actual_cost_cny": float(true_costs[_ENGINE.ACTUAL]),
                "oracle_contract_cost_cny": float(true_costs[_ENGINE.CONTRACT]),
                "estimated_capacity_cost_cny": float(est_cost[_ENGINE.CAPACITY]),
                "estimated_actual_cost_cny": float(est_cost[_ENGINE.ACTUAL]),
                "estimated_contract_cost_cny": float(est_cost[_ENGINE.CONTRACT]),
                "estimated_distribution_mean_kw": mean_est,
                "estimated_distribution_q_p_star_kw": float(dist.ppf(policy.contract_probability)),
                "oracle_q_p_star_kw": float(_ORACLE.ppf(policy.contract_probability)),
                "p_star": float(policy.contract_probability),
                "oracle_stoploss_at_estimated_threshold_kw": float(_ORACLE.stop_loss(
                    policy.alpha * float(est["contract_D_kw"]))),
                "estimated_stoploss_at_estimated_threshold_kw": float(dist.stop_loss(
                    policy.alpha * float(est["contract_D_kw"]))),
            })
    return decision_rows, diagnostic_rows, month_df.to_dict("records"), seed_record


def _independent_law_checks(method: str, dist: Any, diagnostic: dict[str, Any],
                            observed_maxima: np.ndarray, support_module: Any) -> dict[str, Any]:
    check: dict[str, Any] = {
        "method_id": method,
        "distribution_class": type(dist).__name__,
        "no_oracle_derived_upper_bound_used": method in {"TAIL", "GEV"},
        "cdf_at_zero": float(dist.cdf(0.0)),
        "finite_mean": bool(math.isfinite(float(dist.mean()))),
        "support_policy": "no_upper" if method in {"TAIL", "GEV"} else "not_applicable",
    }
    points = np.array([0.0, float(observed_maxima.min()),
                       float(np.median(observed_maxima)),
                       float(observed_maxima.max()),
                       float(observed_maxima.max() + 100_000.0)])
    cdf_values = np.asarray([float(dist.cdf(float(x))) for x in points])
    check["cdf_probe_points_kw"] = points.tolist()
    check["cdf_probe_values"] = cdf_values.tolist()
    check["cdf_finite_monotone"] = bool(np.isfinite(cdf_values).all()
                                         and np.all(np.diff(cdf_values) >= -1e-10))
    check["quantile_p99_finite"] = bool(math.isfinite(float(dist.ppf(0.99)))
                                         and float(dist.ppf(0.99)) >= 0.0)
    if method == "TAIL":
        if type(dist).__name__ == "TailProcessDistribution":
            natural_endpoint = (float(dist.threshold - dist.sigma / dist.xi)
                                if dist.xi < 0.0 else math.inf)
            check.update({
                "finite_mean_xi_gate": bool(float(dist.xi) < 1.0),
                "finite_natural_endpoint_contains_observed_maxima": bool(
                    natural_endpoint >= float(observed_maxima.max()) - 1e-8),
                "tail_count_model": str(dist.count_model),
                "tail_xi": float(dist.xi),
                "tail_sigma_kw": float(dist.sigma),
                "natural_endpoint_kw": natural_endpoint if math.isfinite(natural_endpoint) else None,
                "artificial_upper_bound": None,
            })
        else:
            check.update({
                "fallback_to_empirical": type(dist).__name__ == "DiscreteDistribution",
                "artificial_upper_bound": None,
            })
    if method == "GEV":
        if isinstance(dist, support_module.NonnegativeGEV):
            check.update({
                "nonnegative_gev": True,
                "finite_mean_xi_gate": bool(float(dist.xi) < 1.0),
                "negative_mass_probability": float(dist.f0),
                "negative_mass_gate_pass": bool(float(dist.f0) <= 0.01),
                "natural_support_upper_kw": (float(dist.upper)
                                             if math.isfinite(float(dist.upper)) else None),
                "artificial_upper_bound": None,
            })
        else:
            check.update({
                "nonnegative_gev": False,
                "fallback_to_empirical": type(dist).__name__ == "DiscreteDistribution",
                "artificial_upper_bound": None,
            })
    check["status"] = "PASS" if check["finite_mean"] and check["cdf_finite_monotone"] and check["quantile_p99_finite"] else "FAIL"
    if method == "TAIL" and type(dist).__name__ == "TailProcessDistribution":
        if not check["finite_mean_xi_gate"] or not check["finite_natural_endpoint_contains_observed_maxima"]:
            check["status"] = "FAIL"
    if method == "GEV" and isinstance(dist, support_module.NonnegativeGEV):
        if not check["finite_mean_xi_gate"] or not check["negative_mass_gate_pass"]:
            check["status"] = "FAIL"
    return check


def _chunk_task(start: int, stop: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                 list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    months: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    for rep in range(start, stop):
        d, f, m, s = _history_task(rep)
        decisions.extend(d)
        fits.extend(f)
        months.extend(m)
        seeds.append(s)
    return decisions, fits, months, seeds


def chunk_paths(out: Path, start: int, stop: int) -> dict[str, Path]:
    stem = f"part_{start:05d}_{stop - 1:05d}"
    return {key: out / "chunks" / key / f"{stem}.parquet"
            for key in ("decisions", "fits", "monthly_maxima", "seeds")}


def write_chunk(out: Path, start: int, stop: int,
                payload: tuple[list[dict[str, Any]], list[dict[str, Any]],
                               list[dict[str, Any]], list[dict[str, Any]]],
                scientific_signature_sha256: str) -> dict[str, Any]:
    decisions, fits, months, seeds = payload
    paths = chunk_paths(out, start, stop)
    frames = {
        "decisions": pd.DataFrame(decisions),
        "fits": pd.DataFrame(fits),
        "monthly_maxima": pd.DataFrame(months),
        "seeds": pd.DataFrame(seeds),
    }
    artifacts = []
    for key, frame in frames.items():
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        tmp = paths[key].with_suffix(".tmp.parquet")
        frame.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(paths[key])
        artifacts.append({"path": str(paths[key]), "sha256": sha256(paths[key]),
                          "rows": int(len(frame))})
    marker = out / "chunks" / "parts" / f"part_{start:05d}_{stop - 1:05d}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"start": start, "stop": stop,
                                  "status": "PASS",
                                  "scientific_signature_sha256": scientific_signature_sha256,
                                  "artifacts": artifacts},
                                 indent=2), encoding="utf-8")
    return json.loads(marker.read_text(encoding="utf-8"))


def valid_chunk(out: Path, start: int, stop: int,
                scientific_signature_sha256: str) -> bool:
    marker = out / "chunks" / "parts" / f"part_{start:05d}_{stop - 1:05d}.json"
    if not marker.exists():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        if (record.get("status") != "PASS" or
                int(record.get("start", -1)) != start or
                int(record.get("stop", -1)) != stop or
                record.get("scientific_signature_sha256") != scientific_signature_sha256):
            return False
        expected_rows = {
            "decisions": (stop - start) * len(METHODS) * 9,
            "fits": (stop - start) * len(METHODS),
            "monthly_maxima": (stop - start) * N_MONTHS,
            "seeds": (stop - start),
        }
        if {Path(a["path"]).parent.name for a in record.get("artifacts", [])} != set(expected_rows):
            return False
        for art in record["artifacts"]:
            p = Path(art["path"])
            key = p.parent.name
            if (key not in expected_rows or int(art.get("rows", -1)) != expected_rows[key] or
                    not p.exists() or sha256(p) != art["sha256"]):
                return False
        if len(record.get("artifacts", [])) != len(expected_rows):
            return False
        return True
    except Exception:
        return False


def read_chunks(out: Path, key: str) -> pd.DataFrame:
    files = sorted((out / "chunks" / key).glob("part_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def summarize(out: Path, replications: int) -> dict[str, Any]:
    decision = read_chunks(out, "decisions")
    fits = read_chunks(out, "fits")
    months = read_chunks(out, "monthly_maxima")
    seeds = read_chunks(out, "seeds")
    expected_decisions = replications * len(METHODS) * 9
    errors = []
    if len(decision) != expected_decisions:
        errors.append(f"decision rows {len(decision)} != {expected_decisions}")
    if decision.duplicated(["replication_id", "method_id", "cell_id"]).any():
        errors.append("duplicate decision key")
    if set(decision.method_id.unique()) != set(METHODS):
        errors.append("method coverage mismatch")
    if decision.groupby("method_id").replication_id.nunique().min() != replications:
        errors.append("history coverage mismatch by method")
    if len(fits) != replications * len(METHODS):
        errors.append(f"fit rows {len(fits)} != {replications * len(METHODS)}")
    if len(months) != replications * N_MONTHS:
        errors.append(f"monthly-maxima rows {len(months)} != {replications * N_MONTHS}")
    if len(seeds) != replications or seeds.replication_id.nunique() != replications:
        errors.append("seed registry coverage mismatch")
    if decision.groupby("cell_id").oracle_mode.nunique().max() != 1:
        errors.append("oracle mode varies across replications")
    law_status_counts: dict[str, dict[str, int]] = {}
    law_failure_examples: list[dict[str, Any]] = []
    if len(fits) and "independent_law_checks_json" in fits:
        parsed_checks = []
        for row in fits[["replication_id", "method_id", "independent_law_checks_json"]].itertuples(index=False):
            try:
                check = json.loads(row.independent_law_checks_json)
            except Exception as exc:
                check = {"status": "FAIL", "failure_reason": f"invalid JSON: {exc}"}
            check_status = str(check.get("status", "MISSING"))
            by_method = law_status_counts.setdefault(str(row.method_id), {})
            by_method[check_status] = by_method.get(check_status, 0) + 1
            parsed_checks.append((str(row.method_id), int(row.replication_id), check))
            if check_status != "PASS":
                law_failure_examples.append({"method_id": str(row.method_id),
                                             "replication_id": int(row.replication_id),
                                             "check": check})
        if any(status_counts.get("PASS", 0) != replications or
               sum(status_counts.values()) != replications
               for status_counts in law_status_counts.values()):
            errors.append("independent per-history law checks are incomplete or failed")
        for method, rep, check in parsed_checks:
            if method in {"TAIL", "GEV"}:
                if check.get("artificial_upper_bound") is not None:
                    errors.append(f"artificial upper bound recorded for {method} history {rep}")
                    break
                if check.get("no_oracle_derived_upper_bound_used") is not True:
                    errors.append(f"oracle-bound audit failed for {method} history {rep}")
                    break
                if method == "GEV" and not (
                    check.get("nonnegative_gev") is True or
                    check.get("fallback_to_empirical") is True
                ):
                    errors.append(f"GEV fit was not NonnegativeGEV or documented empirical fallback at history {rep}")
                    break
    else:
        errors.append("independent per-history law checks missing")

    def summary_rows(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        rows = []
        for keys, g in df.groupby(groups, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(groups, keys))
            for name in ("regret_cny_per_month", "mode_selection_regret_cny",
                         "within_contract_regret_cny", "mean_error_kw",
                         "contract_value_error_cny"):
                values = g[name].to_numpy(float)
                mean = float(values.mean())
                se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else math.nan
                row[f"{name}_mean"] = mean
                row[f"{name}_se"] = se
                row[f"{name}_ci95_low"] = mean - 1.96 * se
                row[f"{name}_ci95_high"] = mean + 1.96 * se
            row["n_history_rows"] = int(g.replication_id.nunique())
            row["action_accuracy"] = float(g.action_correct.mean())
            row["mode_error_rate"] = 1.0 - float(g.action_correct.mean())
            rows.append(row)
        return pd.DataFrame(rows)

    cell_summary = summary_rows(decision, ["method_id", "cell_id", "oracle_region"])
    cell_summary.to_csv(out / "summary" / "five_method_by_cell.csv", index=False)
    # Give every cell equal weight within an oracle region for a history-level
    # comparison, then pair methods on the same replication IDs.
    hist_region = decision.groupby(["replication_id", "method_id", "oracle_region"],
                                   sort=True, as_index=False).agg(
        regret_cny_per_month=("regret_cny_per_month", "mean"),
        mode_selection_regret_cny=("mode_selection_regret_cny", "mean"),
        within_contract_regret_cny=("within_contract_regret_cny", "mean"),
        mean_error_kw=("mean_error_kw", "mean"),
        contract_value_error_cny=("contract_value_error_cny", "mean"),
        action_correct=("action_correct", "mean"),
    )
    # Summary routine expects binary action_correct; here it is the fraction of
    # correct cells for that history, which is the intended regional average.
    region_summary = summary_rows(hist_region, ["method_id", "oracle_region"])
    region_summary.to_csv(out / "summary" / "five_method_by_oracle_region.csv", index=False)

    pair_rows = []
    region_hist = hist_region.pivot(index=["replication_id", "oracle_region"],
                                    columns="method_id", values="regret_cny_per_month")
    for region in sorted(decision.oracle_region.unique()):
        sub = region_hist.xs(region, level="oracle_region")
        for i, left in enumerate(METHODS):
            for right in METHODS[i + 1:]:
                diff = (sub[left] - sub[right]).dropna().to_numpy(float)
                se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else math.nan
                mean = float(diff.mean()) if len(diff) else math.nan
                pair_rows.append({
                    "oracle_region": region, "method_a": left, "method_b": right,
                    "paired_mean_regret_difference_a_minus_b_cny": mean,
                    "paired_se_cny": se,
                    "ci95_low_cny": mean - 1.96 * se,
                    "ci95_high_cny": mean + 1.96 * se,
                    "paired_history_observations_equal_cell_weighted": int(len(diff)),
                })
    pd.DataFrame(pair_rows).to_csv(out / "summary" / "paired_method_contrasts_by_region.csv", index=False)

    fit_summary = fits.groupby(["method_id", "fit_status"], dropna=False).agg(
        n_fits=("replication_id", "size"), unique_histories=("replication_id", "nunique")
    ).reset_index()
    fit_summary.to_csv(out / "summary" / "fit_status_counts.csv", index=False)
    decomposition_max = float(decision.regret_decomposition_error_cny.abs().max()) if len(decision) else math.nan
    diagnostics = {
        "status": "PASS" if not errors and decomposition_max <= 1e-6 else "FAIL",
        "errors": errors,
        "replications": replications,
        "decision_rows": int(len(decision)),
        "expected_decision_rows": expected_decisions,
        "unique_histories": int(decision.replication_id.nunique()) if len(decision) else 0,
        "methods": sorted(decision.method_id.unique().tolist()) if len(decision) else [],
        "cells": sorted(decision.cell_id.unique().tolist()) if len(decision) else [],
        "oracle_regions": sorted(decision.oracle_region.unique().tolist()) if len(decision) else [],
        "unique_decision_keys": bool(not decision.duplicated(["replication_id", "method_id", "cell_id"]).any()) if len(decision) else False,
        "all_metrics_finite": bool(np.isfinite(decision.select_dtypes(include=[np.number]).to_numpy()).all()) if len(decision) else False,
        "max_regret_decomposition_abs_error_cny": decomposition_max,
        "maximum_tail_events_per_history": int(seeds.total_tail_events.max()) if len(seeds) else None,
        "monthly_maxima_rows": int(len(months)),
        "fit_rows": int(len(fits)),
        "seed_rows": int(len(seeds)),
        "fit_status_counts": fit_summary.to_dict("records"),
        "independent_law_check_status_counts_by_method": law_status_counts,
        "independent_law_check_failure_examples": law_failure_examples[:20],
        "no_upper_support_audit": {
            "methods": ["TAIL", "GEV"],
            "artificial_or_oracle_derived_upper_bound_used": False,
            "gev_fit_class": "NonnegativeGEV; empirical fallback recorded only when corrected fit rejects",
            "history_level_checks_required": replications * 2,
        },
    }
    (out / "summary" / "validation.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def main() -> None:
    global SEED_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-repo", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--replications", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--seed-root", type=int, default=SEED_ROOT)
    ap.add_argument("--log-file", type=Path, default=None,
                    help="append parent stdout/stderr to this local file")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--init-smoke", action="store_true",
                    help="initialize exactly workers independent workers and record their PIDs/thread caps")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log_handle
        sys.stderr = log_handle
    if args.replications < 1 or args.workers < 1 or args.chunk_size < 1:
        raise ValueError("replications, workers, and chunk-size must be positive")
    SEED_ROOT = int(args.seed_root)
    base_repo = args.base_repo.resolve()
    if not (base_repo / "scripts" / "run_revision_experiment.py").is_file():
        raise FileNotFoundError(f"base engine missing: {base_repo}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary").mkdir(exist_ok=True)
    preflight_record, design = preflight(base_repo)
    design.to_csv(out / "summary" / "oracle_tariff_design.csv", index=False)
    (out / "summary" / "preflight.json").write_text(json.dumps(preflight_record, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(json.dumps(preflight_record, indent=2))
        return
    if args.init_smoke:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                initargs=(str(base_repo), SEED_ROOT)) as pool:
            smoke_rows = list(pool.map(_worker_init_smoke_task, range(args.workers)))
        pids = sorted({int(row["pid"]) for row in smoke_rows})
        observed_thread_counts = [int(pool_info["num_threads"])
                                  for row in smoke_rows for pool_info in row["thread_pools"]
                                  if "num_threads" in pool_info]
        smoke = {
            "status": "PASS" if len(pids) == args.workers and
                     all(row["engine_loaded"] and row["support_module_loaded"] for row in smoke_rows) and
                     all(n <= 1 for n in observed_thread_counts) else "FAIL",
            "requested_workers": int(args.workers),
            "unique_worker_processes": len(pids),
            "worker_pids": pids,
            "rows": smoke_rows,
            "numerical_thread_caps": {key: os.environ.get(key) for key in NUMERICAL_THREAD_ENV},
        }
        smoke_path = out / "summary" / f"worker_init_smoke_{args.workers}.json"
        smoke_path.write_text(json.dumps(smoke, indent=2), encoding="utf-8")
        print(json.dumps(smoke, indent=2))
        if smoke["status"] != "PASS":
            raise SystemExit(3)
        return
    settings = read_settings(base_repo)
    manifest_path = out / "run_manifest.json"
    signature = {
        "experiment": "five_method_nonpoisson_gpd_misspecification_challenge",
        "status": "RUNNING",
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "replications_requested": int(args.replications),
        "workers": int(args.workers),
        "chunk_size_histories": int(args.chunk_size),
        "training_months": N_MONTHS,
        "seed_root": SEED_ROOT,
        "seed_namespace": SEED_NAMESPACE,
        "seed_mapping": "DGP replication r uses SeedSequence([seed_root,95021,r]) with PCG64; corrected TAIL select/bootstrap streams use engine.seed_rng(seed_root,2,0,r,20,1) and engine.seed_rng(seed_root,2,0,r,20,2)",
        "dgp": DGP,
        "methods": list(METHODS),
        "oracle_design_sha256": sha256(out / "summary" / "oracle_tariff_design.csv"),
        "preflight_status": preflight_record["status"],
        "base_repo": str(base_repo),
        "tariff_config_sha256": preflight_record["tariff_source_sha256"],
        "engine_sha256": preflight_record["engine_source_sha256"],
        "no_upper_fit_source": preflight_record["no_upper_fit_source"],
        "no_upper_fit_source_sha256": preflight_record["no_upper_fit_source_sha256"],
        "runner_code_sha256": sha256(HERE),
        "numerical_thread_caps": {key: os.environ.get(key) for key in NUMERICAL_THREAD_ENV},
        "estimator_mode": "full settings from frozen base revision.yaml; TAIL uses four-fold month-level held-out CRPS threshold selection and a 500-resample full-fit shape diagnostic; GEV uses NonnegativeGEV without artificial upper truncation; all decision functionals deterministic",
        "resume_requested": bool(args.resume),
        "chunk_manifest": [],
    }
    scientific_signature = {
        key: signature[key] for key in (
            "experiment", "replications_requested", "chunk_size_histories",
            "training_months", "seed_root", "seed_namespace", "dgp", "methods",
            "oracle_design_sha256", "preflight_status", "base_repo",
            "tariff_config_sha256", "engine_sha256", "no_upper_fit_source_sha256",
            "runner_code_sha256", "estimator_mode",
        )
    }
    scientific_signature_sha256 = canonical_hash(scientific_signature)
    signature["scientific_signature"] = scientific_signature
    signature["scientific_signature_sha256"] = scientific_signature_sha256
    prior_manifest = None
    if args.resume:
        if not manifest_path.is_file():
            raise RuntimeError("--resume requested but no prior run_manifest.json exists")
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_manifest.get("scientific_signature_sha256") != scientific_signature_sha256:
            raise RuntimeError("resume refused: scientific signature differs from prior run")
        if prior_manifest.get("status") == "INVALID":
            raise RuntimeError("resume refused: prior run manifest is marked INVALID")
        signature["started_local"] = prior_manifest.get("started_local", signature["started_local"])
        signature["worker_history"] = list(prior_manifest.get("worker_history", []))
    signature.setdefault("worker_history", []).append({
        "workers": int(args.workers), "started_local": signature["started_local"],
        "resume_requested": bool(args.resume),
    })
    manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    ranges = [(start, min(args.replications, start + args.chunk_size))
              for start in range(0, args.replications, args.chunk_size)]
    completed = [r for r in ranges if valid_chunk(out, *r, scientific_signature_sha256)] if args.resume else []
    pending = [r for r in ranges if r not in completed]
    signature["chunks_already_passed"] = len(completed)
    signature["chunks_planned"] = len(ranges)
    signature["chunks_pending_at_start"] = len(pending)
    manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    t0 = time.time()
    chunk_results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
        initargs=(str(base_repo), SEED_ROOT)) as pool:
        futures = {pool.submit(_chunk_task, start, stop): (start, stop)
                   for start, stop in pending}
        for future in as_completed(futures):
            start, stop = futures[future]
            payload = future.result()
            written = write_chunk(out, start, stop, payload, scientific_signature_sha256)
            chunk_results.append(written)
            done = len(completed) + len(chunk_results)
            signature["chunks_passed"] = done
            signature["chunk_manifest"] = chunk_results
            signature["elapsed_seconds"] = round(time.time() - t0, 3)
            manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
            if done % 4 == 0 or done == len(ranges):
                print(f"completed chunks: {done}/{len(ranges)}", flush=True)
    summary = summarize(out, args.replications)
    signature["status"] = "PASS" if summary["status"] == "PASS" else "FAIL"
    signature["completed_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    signature["elapsed_seconds"] = round(time.time() - t0, 3)
    signature["chunks_passed"] = len(ranges)
    signature["summary_validation"] = summary
    signature["chunk_manifest"] = [
        json.loads((out / "chunks" / "parts" / f"part_{s:05d}_{e-1:05d}.json").read_text(encoding="utf-8"))
        for s, e in ranges
    ]
    manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(json.dumps({"status": signature["status"], "out": str(out),
                      "elapsed_seconds": signature["elapsed_seconds"],
                      "summary": summary}, indent=2))
    if signature["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
