#!/usr/bin/env python3
"""Re-evaluate verified four-fold CV-CRPS fitted laws under high-p* tariffs.

This is a deterministic tariff re-analysis. It does not refit estimators or
generate new histories. Source fits must come from the PASS revised
10,000-history primary decision audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as student_t

PACKAGE = Path(__file__).resolve().parents[3]
REVISION = PACKAGE / "CSSC_REVIEW_REVISION_20261112"
BASE_REPO = PACKAGE / "exp1" / "code" / "legacy_engine"
SOURCE_ROOT = REVISION / "primary_cv_crps_decisions_20261115"
SOURCE_AUDIT_DIRS = list((SOURCE_ROOT / "support_policy_audit").glob("*/audit_manifest.json"))
if len(SOURCE_AUDIT_DIRS) != 1:
    raise RuntimeError("Expected exactly one revised primary decision manifest")
SOURCE_AUDIT = SOURCE_AUDIT_DIRS[0]
OUT = REVISION / "high_pstar_cv_crps_20261115"
CHUNK_OUT = OUT / "checkpoints"
SEED_ROOT = 20261001
N_REPS = 10000
SOURCE_CHUNK = 25
WORKERS = 5
ALPHA = 2.2
DEMAND_RATE = 36.0
CAPACITY_RATE = 22.5
DELTA = 0.4
UPPER_RATIO = 1.0
P_STARS = (0.90, 0.95, 0.99)
UTILIZATIONS = (0.39, 0.405, 0.43, 0.45, 0.50, 0.625, 0.70, 0.80, 0.84, 0.95)
METHODS = ("TAIL", "EVENT-EMP", "GEV", "KDE", "EMP")
MODES = ("actual_maximum_demand", "capacity", "contracted_maximum_demand")
TOL_CNY = 1e-5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_runtime() -> tuple[Any, Any]:
    sys.path.insert(0, str(BASE_REPO / "src"))
    source = BASE_REPO / "scripts" / "run_revision_experiment.py"
    spec = importlib.util.spec_from_file_location("high_pstar_source_engine", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source tariff engine: {source}")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    helper_path = PACKAGE / "exp1" / "code" / "revised" / "run_primary_no_nominal_ks.py"
    sys.path.insert(0, str(helper_path.parent))
    helper_spec = importlib.util.spec_from_file_location("high_pstar_support_audit", helper_path)
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError(f"Cannot import verified nonnegative GEV implementation: {helper_path}")
    helper = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_spec.name] = helper
    helper_spec.loader.exec_module(helper)
    return engine, helper


class FastDiscrete:
    """Discrete law adapter with exact O(log n) stop-loss evaluation."""

    def __init__(self, support: list[float], probabilities: list[float]):
        x = np.asarray(support, dtype=float)
        p = np.asarray(probabilities, dtype=float)
        order = np.argsort(x, kind="mergesort")
        self.x, self.p = x[order], p[order]
        total = float(self.p.sum())
        if (not np.isfinite(self.x).all() or not np.isfinite(self.p).all()
                or np.any(self.p < 0) or total <= 0 or abs(total - 1.0) > 1e-8):
            raise ValueError("Invalid discrete fitted law")
        self.p = self.p / total
        # Source fit laws are already unique and sorted; reject malformed records.
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("Discrete fitted law support is not unique and increasing")
        self.cum = np.cumsum(self.p)
        self.tail_p = np.r_[np.cumsum(self.p[::-1])[::-1], 0.0]
        self.tail_xp = np.r_[np.cumsum((self.x * self.p)[::-1])[::-1], 0.0]
        self._mean = float(np.dot(self.x, self.p))

    def mean(self) -> float:
        return self._mean

    def ppf(self, probability: float) -> float:
        i = int(np.searchsorted(self.cum, probability, side="left"))
        return float(self.x[min(i, len(self.x) - 1)])

    def stop_loss(self, threshold: float) -> float:
        i = int(np.searchsorted(self.x, max(0.0, float(threshold)), side="right"))
        return float(self.tail_xp[i] - max(0.0, float(threshold)) * self.tail_p[i])


def make_law(engine: Any, helper: Any, row: dict[str, Any], fast_discrete: bool = True) -> Any:
    law = row["law"]
    cls = law["class"]
    if cls == "TailProcessDistribution":
        return engine.TailProcessDistribution(
            law["threshold_kw"], law["xi"], law["sigma_kw"], law["baseline_monthly_maxima_kw"],
            law["count_model"], law["count_mean"], law["count_variance"], law["nb_size"], law["nb_probability"],
        )
    if cls == "NonnegativeGEV":
        d = helper.NonnegativeGEV(law["shape_c"], law["loc_kw"], law["scale_kw"])
        if "raw_negative_mass" in law and abs(d.f0 - law["raw_negative_mass"]) > 1e-10:
            raise ValueError("Stored GEV support-conditioning diagnostic mismatch")
        return d
    if cls == "NonnegativeGaussianKDE":
        d = engine.NonnegativeGaussianKDE(np.asarray(law["sample_kw"], dtype=float))
        if not math.isclose(d.bandwidth, float(law["bandwidth_kw"]), rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError("Stored KDE bandwidth does not match the source implementation")
        return d
    if cls == "DiscreteDistribution":
        if fast_discrete:
            return FastDiscrete(law["support_kw"], law["probabilities"])
        return engine.DiscreteDistribution(np.asarray(law["support_kw"], dtype=float),
                                           np.asarray(law["probabilities"], dtype=float))
    raise ValueError(f"Unsupported source law class {cls!r}")


def target_policy(engine: Any, p: float) -> Any:
    kappa = 1.0 / ((1.0 - p) * ALPHA)
    if not math.isclose(ALPHA * kappa, 1.0 / (1.0 - p), rel_tol=1e-14):
        raise AssertionError("p* parameterization failed")
    return engine.Policy(f"HIGH_PSTAR_{p:.2f}", DEMAND_RATE, CAPACITY_RATE, ALPHA, kappa, DELTA, UPPER_RATIO)


def design_settings() -> dict[str, Any]:
    return {
        "purpose": "high_pstar_tariff_stress_reanalysis_not_primary_tariff_result",
        "design_type": "fixed fitted laws and baseline DGP; vary only kappa to set p*=1-1/(kappa*alpha)",
        "alpha": ALPHA,
        "p_star_levels": list(P_STARS),
        "kappa_by_p_star": {f"{p:.2f}": 1.0 / ((1.0 - p) * ALPHA) for p in P_STARS},
        "demand_rate_cny_kw_month": DEMAND_RATE,
        "capacity_rate_cny_kva_month": CAPACITY_RATE,
        "delta_lower_contract_ratio": DELTA,
        "contract_upper_ratio": UPPER_RATIO,
        "utilization_grid": list(UTILIZATIONS),
        "replications": N_REPS,
        "seed_root_of_reused_histories": SEED_ROOT,
        "workers": WORKERS,
        "history_generation": "none; reuse verified primary 10k fits; no RNG draws in re-analysis",
        "methods": list(METHODS),
        "decision_tie_rule": "source engine lexicographic min by (expected cost, mode string)",
    }


def get_chunk_paths(audit: dict[str, Any]) -> list[dict[str, Any]]:
    if audit.get("status") != "PASS" or audit.get("range") != [0, N_REPS]:
        raise ValueError("Source audit is not the PASS 10,000-replication no-upper run")
    if int(audit.get("chunk_size", -1)) != SOURCE_CHUNK or audit.get("policies") != ["no_upper"]:
        raise ValueError("Source fits do not match the corrected no-upper 10k design")
    chunk_dir = SOURCE_ROOT / "support_policy_audit" / str(audit["signature"])[:16]
    rows = []
    expected = list(range(0, N_REPS, SOURCE_CHUNK))
    chunks = sorted(audit.get("chunks", []), key=lambda x: int(x.get("start", -1)))
    if [int(c.get("start", -1)) for c in chunks] != expected:
        raise ValueError("Source audit chunk coverage is incomplete")
    for c in chunks:
        if c.get("status") != "PASS" or int(c["end"]) != int(c["start"]) + SOURCE_CHUNK:
            raise ValueError(f"Source fit chunk is not PASS: {c.get('start')}")
        fit_path = chunk_dir / c["fits"]
        decision_path = chunk_dir / c["csv"]
        if not fit_path.is_file() or sha256_file(fit_path) != c.get("fits_sha256"):
            raise ValueError(f"Fit law hash mismatch: {fit_path}")
        if not decision_path.is_file() or sha256_file(decision_path) != c.get("csv_sha256"):
            raise ValueError(f"Decision source hash mismatch: {decision_path}")
        if int(c.get("fit_records", -1)) != SOURCE_CHUNK * len(METHODS):
            raise ValueError(f"Unexpected fit law count in chunk {c['start']}")
        rows.append({"start": int(c["start"]), "end": int(c["end"]), "fits": str(fit_path),
                     "fits_sha256": c["fits_sha256"], "csv": str(decision_path),
                     "csv_sha256": c["csv_sha256"]})
    return rows


def oracle_preflight(engine: Any) -> dict[str, Any]:
    with (BASE_REPO / "configs" / "revision.yaml").open("r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)["revision"]
    oracle = engine.OracleDistribution(engine.dgp_from_settings(settings))
    counts: dict[str, dict[str, int]] = {}
    statuses: dict[str, dict[str, int]] = {}
    cell_rows = []
    for p in P_STARS:
        policy = target_policy(engine, p)
        counts[f"{p:.2f}"] = {m: 0 for m in MODES}
        statuses[f"{p:.2f}"] = {"lower": 0, "interior": 0, "upper": 0}
        for u in UTILIZATIONS:
            S = oracle.mean() / u
            d = engine.optimized_decision(oracle, S, policy)
            mode = d["selected_mode"]
            counts[f"{p:.2f}"][mode] += 1
            D = float(d["contract_D_kw"])
            lower, upper = DELTA * S, UPPER_RATIO * S
            status = "lower" if math.isclose(D, lower, rel_tol=0, abs_tol=1e-7) else (
                "upper" if math.isclose(D, upper, rel_tol=0, abs_tol=1e-7) else "interior")
            if mode == engine.CONTRACT:
                statuses[f"{p:.2f}"][status] += 1
            cell_rows.append({"p_star": p, "kappa": policy.kappa, "utilization": u, "S_kva": S,
                              "oracle_mode": mode, "oracle_contract_D_kw": D,
                              "contract_optimizer_status": status,
                              "oracle_contract_value_cny": d["mode_costs"][engine.CONTRACT],
                              "oracle_actual_value_cny": d["mode_costs"][engine.ACTUAL],
                              "oracle_capacity_value_cny": d["mode_costs"][engine.CAPACITY],
                              "oracle_best_value_cny": d["best_value_cny"],
                              "oracle_margin_cny": d["margin_cny"]})
    for p in P_STARS:
        c = counts[f"{p:.2f}"]
        if not all(c[m] > 0 for m in MODES):
            raise ValueError(f"High-p* design invalid: {p} lacks at least one oracle mode: {c}")
        s = statuses[f"{p:.2f}"]
        if s["lower"] == 0 or s["interior"] == 0:
            raise ValueError(f"High-p* design invalid: {p} lacks lower/interior contract solutions: {s}")
    return {"status": "PASS", "oracle_mean_kw": oracle.mean(), "oracle_q_pstar_kw": {
                f"{p:.2f}": oracle.ppf(p) for p in P_STARS},
            "oracle_es_pstar_kw": {f"{p:.2f}": oracle.ppf(p) + oracle.stop_loss(oracle.ppf(p))/(1-p) for p in P_STARS},
            "oracle_mode_grid_counts": counts, "contract_solution_status_counts": statuses,
            "cell_rows": cell_rows}


def baseline_reconstruction(engine: Any, helper: Any, chunk: dict[str, Any]) -> dict[str, Any]:
    fit_path, decision_path = Path(chunk["fits"]), Path(chunk["csv"])
    laws: dict[tuple[int, str], dict[str, Any]] = {}
    with fit_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if int(row["replication_id"]) > 1:
                break
            laws[(int(row["replication_id"]), row["method_id"])] = row
    frame = pd.read_csv(decision_path)
    sample = frame[frame.replication_id.isin([0, 1])]
    checked = 0
    max_cost = max_D = 0.0
    mode_mismatches = []
    for rec in sample.to_dict(orient="records"):
        key = (int(rec["replication_id"]), rec["method_id"])
        law_row = laws.get(key)
        if law_row is None:
            raise ValueError(f"Missing source fitted law for baseline reconstruction: {key}")
        law = make_law(engine, helper, law_row, fast_discrete=False)
        policy = engine.Policy(str(rec["policy_id"]), float(DEMAND_RATE), float(CAPACITY_RATE),
                               float(rec["alpha"]), float(rec["kappa"]), DELTA, UPPER_RATIO)
        decision = engine.optimized_decision(law, float(rec["S_kva"]), policy)
        if decision["selected_mode"] != rec["selected_mode"]:
            mode_mismatches.append({"replication_id": key[0], "method": key[1], "cell_id": rec["cell_id"],
                                    "expected": rec["selected_mode"], "observed": decision["selected_mode"]})
        max_D = max(max_D, abs(float(decision["contract_D_kw"]) - float(rec["estimated_contract_D_kw"])))
        for mode, col in ((engine.ACTUAL, "estimated_mode_cost_actual"),
                          (engine.CAPACITY, "estimated_mode_cost_capacity"),
                          (engine.CONTRACT, "estimated_mode_cost_contract")):
            max_cost = max(max_cost, abs(float(decision["mode_costs"][mode]) - float(rec[col])))
        checked += 1
    if mode_mismatches or max_D > 1e-7 or max_cost > 1e-4:
        raise ValueError(f"Fitted law reconstruction failed: mismatches={len(mode_mismatches)}, max_cost={max_cost}, max_D={max_D}")
    return {"status": "PASS", "baseline_decision_rows_reconstructed": checked,
            "source_histories": [0, 1], "mode_mismatches": len(mode_mismatches),
            "max_cost_abs_error_cny": max_cost, "max_contract_D_abs_error_kw": max_D}


def _key(method: str, pidx: int, uidx: int) -> str:
    return f"{method}|{pidx}|{uidx}"


def _blank_accum() -> dict[str, Any]:
    return {"n": 0, "correct_n": 0, "mode_count": {m: 0 for m in MODES},
            "regret_sum": 0.0, "regret_sq_sum": 0.0, "mode_gap_sum": 0.0,
            "within_contract_sum": 0.0, "mean_error_sum": 0.0, "mean_error_sq_sum": 0.0,
            "q_error_sum": 0.0, "q_error_sq_sum": 0.0, "es_error_sum": 0.0,
            "es_error_sq_sum": 0.0, "lower_D_n": 0, "interior_D_n": 0, "upper_D_n": 0,
            "decomp_max_abs_error": 0.0}


def _calculate_rep(engine: Any, helper: Any, laws: dict[str, dict[str, Any]], oracle: Any,
                   oracle_mean: float, oracle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policy_cache = [target_policy(engine, p) for p in P_STARS]
    cell_truth = {(i, j): oracle_rows[i * len(UTILIZATIONS) + j]
                  for i in range(len(P_STARS)) for j in range(len(UTILIZATIONS))}
    out = []
    oracle_h_cache: dict[float, float] = {}
    for method in METHODS:
        d = make_law(engine, helper, laws[method], fast_discrete=True)
        mu_hat = float(d.mean())
        for pidx, (p, policy) in enumerate(zip(P_STARS, policy_cache, strict=True)):
            q_hat = float(d.ppf(p))
            h_q_hat = float(d.stop_loss(q_hat))
            es_hat = q_hat + h_q_hat / (1.0 - p)
            q_true = float(oracle.ppf(p))
            h_q_true = float(oracle.stop_loss(q_true))
            es_true = q_true + h_q_true / (1.0 - p)
            h_cache: dict[float, float] = {float(q_hat): h_q_hat}
            for uidx, u in enumerate(UTILIZATIONS):
                truth = cell_truth[(pidx, uidx)]
                S = float(truth["S_kva"])
                lo, hi = DELTA * S, UPPER_RATIO * S
                D_hat = float(np.clip(q_hat / ALPHA, lo, hi))
                threshold = float(ALPHA * D_hat)
                if threshold not in h_cache:
                    h_cache[threshold] = float(d.stop_loss(threshold))
                h_hat = h_cache[threshold]
                est_cost = {
                    engine.ACTUAL: DEMAND_RATE * mu_hat,
                    engine.CAPACITY: CAPACITY_RATE * S,
                    engine.CONTRACT: DEMAND_RATE * (D_hat + policy.kappa * h_hat),
                }
                selected = min(MODES, key=lambda m: (est_cost[m], m))
                truth_cost = {engine.ACTUAL: DEMAND_RATE * oracle_mean,
                              engine.CAPACITY: CAPACITY_RATE * S,
                              engine.CONTRACT: float(truth["oracle_contract_value_cny"])}
                if threshold not in oracle_h_cache:
                    oracle_h_cache[threshold] = float(oracle.stop_loss(threshold))
                h_true_estimated_action = oracle_h_cache[threshold]
                oracle_best = float(truth["oracle_best_value_cny"])
                selected_true = (CAPACITY_RATE * S if selected == engine.CAPACITY else
                                 DEMAND_RATE * oracle_mean if selected == engine.ACTUAL else
                                 DEMAND_RATE * (D_hat + policy.kappa * h_true_estimated_action))
                regret = selected_true - oracle_best
                mode_gap = float(truth_cost[selected] - oracle_best)
                within = (float(selected_true - truth_cost[engine.CONTRACT])
                          if selected == engine.CONTRACT else 0.0)
                decomp = regret - (mode_gap + within)
                if regret < -TOL_CNY or within < -TOL_CNY or abs(decomp) > TOL_CNY:
                    raise ArithmeticError(f"Regret decomposition/feasibility failed: {method} p={p} u={u} regret={regret}, within={within}, residual={decomp}")
                if math.isclose(D_hat, lo, rel_tol=0, abs_tol=1e-7): status = "lower"
                elif math.isclose(D_hat, hi, rel_tol=0, abs_tol=1e-7): status = "upper"
                else: status = "interior"
                out.append({"method_id": method, "p_index": pidx, "u_index": uidx,
                            "oracle_mode": truth["oracle_mode"], "selected_mode": selected,
                            "correct": int(selected == truth["oracle_mode"]), "regret": regret,
                            "mode_gap": mode_gap, "within": within,
                            "mu_error": mu_hat - oracle_mean, "q_error": q_hat - q_true,
                            "es_error": es_hat - es_true, "D_status": status,
                            "decomp_abs_error": abs(decomp)})
    return out


def process_chunk(task: dict[str, Any]) -> dict[str, Any]:
    outpath = Path(task["out"])
    regret_path = outpath.with_name(outpath.stem + "_regrets.npz")
    if outpath.is_file():
        try:
            old = json.loads(outpath.read_text(encoding="utf-8"))
            if (old.get("signature") == task["signature"] and old.get("status") == "PASS"
                    and regret_path.is_file()
                    and old.get("regrets_sha256") == sha256_file(regret_path)):
                return {"status": "SKIPPED_VALID", "start": task["start"], "end": task["end"], "out": str(outpath)}
        except Exception:
            pass
    engine, helper = load_runtime()
    with (BASE_REPO / "configs" / "revision.yaml").open("r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)["revision"]
    oracle = engine.OracleDistribution(engine.dgp_from_settings(settings))
    oracle_mean = float(oracle.mean())
    # Oracle rows are in p-major, utilization-minor order and were preflighted.
    oracle_rows = json.loads(Path(task["oracle_rows_path"]).read_text(encoding="utf-8"))
    oracle_rows = oracle_rows["cell_rows"]
    acc: dict[str, dict[str, Any]] = {}
    regret_cube = np.full((SOURCE_CHUNK, len(METHODS), len(P_STARS), len(UTILIZATIONS)), np.nan)
    reps = set()
    method_sets: dict[int, dict[str, dict[str, Any]]] = {}
    with Path(task["fits"]).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rep = int(row["replication_id"])
            if not int(task["start"]) <= rep < int(task["end"]):
                raise ValueError(f"replication id outside source chunk: {rep}")
            m = row["method_id"]
            if m not in METHODS or m in method_sets.setdefault(rep, {}):
                raise ValueError(f"invalid/duplicate method law in rep {rep}: {m}")
            method_sets[rep][m] = row
            if len(method_sets[rep]) == len(METHODS):
                result_rows = _calculate_rep(engine, helper, method_sets.pop(rep), oracle, oracle_mean, oracle_rows)
                reps.add(rep)
                for r in result_rows:
                    regret_cube[rep - int(task["start"]), METHODS.index(r["method_id"]),
                                r["p_index"], r["u_index"]] = r["regret"]
                    key = _key(r["method_id"], r["p_index"], r["u_index"])
                    a = acc.setdefault(key, _blank_accum())
                    a["n"] += 1; a["correct_n"] += int(r["correct"])
                    a["mode_count"][r["selected_mode"]] += 1
                    a["regret_sum"] += r["regret"]; a["regret_sq_sum"] += r["regret"] ** 2
                    a["mode_gap_sum"] += r["mode_gap"]; a["within_contract_sum"] += r["within"]
                    a["mean_error_sum"] += r["mu_error"]; a["mean_error_sq_sum"] += r["mu_error"] ** 2
                    a["q_error_sum"] += r["q_error"]; a["q_error_sq_sum"] += r["q_error"] ** 2
                    a["es_error_sum"] += r["es_error"]; a["es_error_sq_sum"] += r["es_error"] ** 2
                    a[f"{r['D_status']}_D_n"] += 1
                    a["decomp_max_abs_error"] = max(a["decomp_max_abs_error"], r["decomp_abs_error"])
    expected_reps = set(range(int(task["start"]), int(task["end"])))
    if reps != expected_reps or method_sets:
        raise ValueError(f"source chunk law coverage invalid: completed={len(reps)}, incomplete={len(method_sets)}")
    if not np.isfinite(regret_cube).all() or np.min(regret_cube) < -TOL_CNY:
        raise ValueError("High-p* regret cube is incomplete or infeasible")
    tmp_regret_path = regret_path.with_suffix(".npz.tmp")
    with tmp_regret_path.open("wb") as f:
        np.savez_compressed(f, regret_cny=regret_cube,
                            replication_id=np.arange(task["start"], task["end"], dtype=np.int32))
    os.replace(tmp_regret_path, regret_path)
    payload = {"status": "PASS", "signature": task["signature"], "source_fits_sha256": task["fits_sha256"],
               "start": task["start"], "end": task["end"], "replications": len(reps), "rows": acc,
               "regrets_file": regret_path.name, "regrets_sha256": sha256_file(regret_path)}
    atomic_json(outpath, payload)
    return {"status": "PASS", "start": task["start"], "end": task["end"], "out": str(outpath)}


def summarize(acc: dict[str, dict[str, Any]], oracle_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_rows = []
    confusion = []
    truth = {(i, j): oracle_rows[i * len(UTILIZATIONS) + j]
             for i in range(len(P_STARS)) for j in range(len(UTILIZATIONS))}
    for method in METHODS:
        for pidx, p in enumerate(P_STARS):
            for uidx, u in enumerate(UTILIZATIONS):
                a = acc[_key(method, pidx, uidx)]
                n = int(a["n"]); mean_reg = a["regret_sum"] / n
                var_reg = max(0.0, (a["regret_sq_sum"] - n * mean_reg * mean_reg) / max(1, n - 1))
                se_reg = math.sqrt(var_reg / n)
                tcrit = float(student_t.ppf(.975, n - 1)) if n > 1 else float("nan")
                lo, hi = mean_reg - tcrit * se_reg, mean_reg + tcrit * se_reg
                row = {"method_id": method, "p_star": p, "kappa_alpha": ALPHA * (1 / ((1-p) * ALPHA)),
                       "kappa": 1 / ((1-p) * ALPHA), "alpha": ALPHA, "utilization": u,
                       "oracle_mode": truth[(pidx, uidx)]["oracle_mode"], "histories": n,
                       "mode_accuracy": a["correct_n"] / n, "mean_regret_cny": mean_reg,
                       "mean_regret_95ci_low_cny": lo, "mean_regret_95ci_high_cny": hi,
                       "mean_mode_gap_cny": a["mode_gap_sum"] / n,
                       "mean_within_contract_cny": a["within_contract_sum"] / n,
                       "regret_decomposition_max_abs_error_cny": a["decomp_max_abs_error"],
                       "selected_actual_rate": a["mode_count"]["actual_maximum_demand"] / n,
                       "selected_contract_rate": a["mode_count"]["contracted_maximum_demand"] / n,
                       "selected_capacity_rate": a["mode_count"]["capacity"] / n,
                       "estimated_D_lower_bound_rate": a["lower_D_n"] / n,
                       "estimated_D_interior_rate": a["interior_D_n"] / n,
                       "estimated_D_upper_bound_rate": a["upper_D_n"] / n,
                       "mean_signed_mean_error_kw": a["mean_error_sum"] / n,
                       "mean_signed_q_pstar_error_kw": a["q_error_sum"] / n,
                       "mean_signed_es_pstar_error_kw": a["es_error_sum"] / n,
                       "oracle_contract_region_margin_cny": truth[(pidx, uidx)]["oracle_margin_cny"]}
                cell_rows.append(row)
                for selected, count in a["mode_count"].items():
                    confusion.append({"method_id": method, "p_star": p, "utilization": u,
                                      "oracle_mode": row["oracle_mode"], "selected_mode": selected,
                                      "count": count, "histories": n, "rate": count/n})
    cell_df = pd.DataFrame(cell_rows)
    confusion_df = pd.DataFrame(confusion)
    # Equal-weight the prespecified utilization cells within each exact oracle region;
    # this is a design-grid summary, not a population-weighted tariff estimate.
    agg_fields = ["mode_accuracy", "mean_regret_cny", "mean_mode_gap_cny", "mean_within_contract_cny",
                  "mean_signed_mean_error_kw", "mean_signed_q_pstar_error_kw", "mean_signed_es_pstar_error_kw",
                  "selected_actual_rate", "selected_contract_rate", "selected_capacity_rate"]
    region_df = (cell_df.groupby(["method_id", "p_star", "oracle_mode"], as_index=False)
                 .agg({**{x: "mean" for x in agg_fields}, "histories": "sum", "utilization": "count"})
                 .rename(columns={"utilization": "design_cells_equal_weighted", "histories": "history_cell_observations"}))
    return cell_df, confusion_df, region_df


def main() -> None:
    global WORKERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run the first verified 25-history source chunk only")
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        raise SystemExit("This task is capped at 5 workers")
    WORKERS = args.workers
    OUT.mkdir(parents=True, exist_ok=True); CHUNK_OUT.mkdir(parents=True, exist_ok=True)
    engine, helper = load_runtime()
    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8-sig"))
    if source_audit.get("status") != "PASS" or source_audit.get("range") != [0, N_REPS]:
        raise ValueError("Revised primary 10,000-history decision audit is incomplete")
    chunks = get_chunk_paths(source_audit)
    preflight = oracle_preflight(engine)
    preflight["law_reconstruction"] = baseline_reconstruction(engine, helper, chunks[0])
    preflight["source_fit_rows"] = len(chunks) * SOURCE_CHUNK * len(METHODS)
    preflight["oracle_mode_counts_at_selected_cells"] = {
        f"{p:.2f}": {m: sum(r["p_star"] == p and r["oracle_mode"] == m for r in preflight["cell_rows"]) for m in MODES}
        for p in P_STARS}
    # Do not launch any workers if the stress design ceases to span all three modes.
    if any(sum(preflight["oracle_mode_counts_at_selected_cells"][f"{p:.2f}"].values()) != len(UTILIZATIONS)
           or any(preflight["oracle_mode_counts_at_selected_cells"][f"{p:.2f}"][m] == 0 for m in MODES)
           for p in P_STARS):
        raise ValueError("Selected utilization grid failed three-mode oracle coverage; stopped before run")
    design = design_settings()
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "PyYAML")}
    source_payload = {"source_audit_sha256": sha256_file(SOURCE_AUDIT),
                      "source_signature": source_audit["signature"], "source_history_signature": source_audit["source_run_signature"],
                      "fit_chunks": [{"start": c["start"], "end": c["end"], "sha256": c["fits_sha256"]} for c in chunks],
                      "base_revision_yaml_sha256": sha256_file(BASE_REPO / "configs" / "revision.yaml"),
                      "source_engine_sha256": sha256_file(BASE_REPO / "scripts" / "run_revision_experiment.py"),
                      "support_audit_helper_sha256": sha256_file(PACKAGE / "exp1" / "code" / "revised" / "run_primary_no_nominal_ks.py"),
                      "script_sha256": sha256_file(Path(__file__).resolve())}
    run_sig = canonical_hash({"design": design, "source": source_payload, "versions": versions})
    oracle_path = OUT / "oracle_design_cells.json"
    atomic_json(oracle_path, {"status": "PASS", "cell_rows": preflight["cell_rows"]})
    manifest_path = OUT / "high_pstar_run_manifest.json"
    manifest = {"status": "RUNNING", "run_signature": run_sig, "design": design,
                "source": source_payload, "runtime_versions": versions,
                "preflight": {k: v for k, v in preflight.items() if k != "cell_rows"},
                "started_local": time.strftime("%Y-%m-%d %H:%M:%S %z"), "chunks": []}
    atomic_json(manifest_path, manifest)
    tasks = []
    for c in chunks:
        payload = {"run_signature": run_sig, "start": c["start"], "end": c["end"], "fits_sha256": c["fits_sha256"]}
        chunk_sig = canonical_hash(payload)
        tasks.append({"start": c["start"], "end": c["end"], "fits": c["fits"], "fits_sha256": c["fits_sha256"],
                      "signature": chunk_sig, "out": str(CHUNK_OUT / f"chunk_{c['start']:05d}_{c['end']:05d}.json"),
                      "oracle_rows_path": str(oracle_path)})
    if args.smoke:
        smoke_task = dict(tasks[0])
        smoke_task["out"] = str(CHUNK_OUT / "smoke_chunk_00000_00050.json")
        smoke_result = process_chunk(smoke_task)
        smoke_payload = json.loads(Path(smoke_task["out"]).read_text(encoding="utf-8"))
        smoke_checks = {"status": "PASS", "histories": smoke_payload.get("replications"),
                        "summary_key_count": len(smoke_payload.get("rows", {})),
                        "all_smoke_cells_have_25_histories": all(v.get("n") == SOURCE_CHUNK for v in smoke_payload.get("rows", {}).values()),
                        "worker_count": 1, "source_chunk": [0, SOURCE_CHUNK],
                        "three_mode_preflight_passed": True, "baseline_law_reconstruction": preflight["law_reconstruction"]}
        if smoke_result.get("status") not in ("PASS", "SKIPPED_VALID") or smoke_checks["histories"] != SOURCE_CHUNK or smoke_checks["summary_key_count"] != len(METHODS)*len(P_STARS)*len(UTILIZATIONS) or not smoke_checks["all_smoke_cells_have_25_histories"]:
            raise RuntimeError(f"High-p* smoke check failed: {smoke_checks}")
        atomic_json(OUT / "high_pstar_smoke_validation.json", smoke_checks)
        print(json.dumps(smoke_checks, indent=2), flush=True)
        return
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_chunk, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"high-p* chunks {i}/{len(tasks)} complete", flush=True)
                manifest["chunks_completed"] = i
                manifest["elapsed_seconds"] = round(time.time() - started, 2)
                atomic_json(manifest_path, manifest)
    if len(results) != len(tasks) or any(r["status"] not in ("PASS", "SKIPPED_VALID") for r in results):
        raise RuntimeError("High-p* chunk execution incomplete")
    merged: dict[str, dict[str, Any]] = {}
    chunk_records = []
    for t in tasks:
        p = Path(t["out"])
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("status") != "PASS" or d.get("signature") != t["signature"] or d.get("replications") != SOURCE_CHUNK:
            raise ValueError(f"Invalid high-p* checkpoint: {p}")
        regret_path = p.with_name(d["regrets_file"])
        if not regret_path.is_file() or sha256_file(regret_path) != d["regrets_sha256"]:
            raise ValueError(f"Invalid high-p* regret checkpoint: {regret_path}")
        chunk_records.append({"start": t["start"], "end": t["end"], "status": "PASS",
                              "source_fits_sha256": t["fits_sha256"], "output_sha256": sha256_file(p),
                              "output_bytes": p.stat().st_size,
                              "regrets_sha256": d["regrets_sha256"],
                              "regrets_bytes": regret_path.stat().st_size})
        for k, v in d["rows"].items():
            a = merged.setdefault(k, _blank_accum())
            for field in ("n", "correct_n", "regret_sum", "regret_sq_sum", "mode_gap_sum", "within_contract_sum",
                          "mean_error_sum", "mean_error_sq_sum", "q_error_sum", "q_error_sq_sum",
                          "es_error_sum", "es_error_sq_sum", "lower_D_n", "interior_D_n", "upper_D_n"):
                a[field] += v[field]
            a["decomp_max_abs_error"] = max(a["decomp_max_abs_error"], v["decomp_max_abs_error"])
            for m in MODES:
                a["mode_count"][m] += v["mode_count"][m]
    expected_keys = len(METHODS) * len(P_STARS) * len(UTILIZATIONS)
    if len(merged) != expected_keys or any(v["n"] != N_REPS for v in merged.values()):
        raise ValueError("High-p* aggregation has missing cells or history counts")
    oracle_rows = preflight["cell_rows"]
    cell_df, confusion_df, region_df = summarize(merged, oracle_rows)
    cell_df.to_csv(OUT / "high_pstar_cell_summary_10000.csv", index=False, float_format="%.12g")
    confusion_df.to_csv(OUT / "high_pstar_mode_confusion_10000.csv", index=False, float_format="%.12g")
    region_df.to_csv(OUT / "high_pstar_oracle_region_summary_10000.csv", index=False, float_format="%.12g")
    pd.DataFrame(oracle_rows).to_csv(OUT / "high_pstar_oracle_design_cells.csv", index=False, float_format="%.12g")
    checks = {
        "status": "PASS", "replications": N_REPS, "methods": list(METHODS),
        "p_star_levels": list(P_STARS), "utilization_cells": len(UTILIZATIONS),
        "decision_rows_reanalyzed": N_REPS * len(METHODS) * len(P_STARS) * len(UTILIZATIONS),
        "summary_cell_count": len(cell_df), "oracle_mode_grid_counts": preflight["oracle_mode_counts_at_selected_cells"],
        "oracle_modes_all_present_each_p_star": True,
        "contract_lower_and_interior_oracle_solutions_each_p_star": True,
        "all_histories_by_method_and_cell": bool((cell_df.histories == N_REPS).all()),
        "all_regret_decomposition_residuals_below_1e-5": bool((cell_df.regret_decomposition_max_abs_error_cny <= TOL_CNY).all()),
        "mean_regret_all_nonnegative_within_tolerance": bool((cell_df.mean_regret_cny >= -TOL_CNY).all()),
        "upper_contract_boundary_oracle_solutions_present": bool(any(r["contract_optimizer_status"] == "upper" and r["oracle_mode"] == "contracted_maximum_demand" for r in oracle_rows)),
        "source_no_upper_fitted_laws_used": True, "only_or_stale_outputs_used": False,
    }
    atomic_json(OUT / "high_pstar_checks.json", checks)
    manifest.update({"status": "PASS", "finished_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                     "elapsed_seconds": round(time.time() - started, 2), "chunks_completed": len(tasks),
                     "chunk_records": chunk_records, "outputs": []})
    for name in ("high_pstar_cell_summary_10000.csv", "high_pstar_mode_confusion_10000.csv",
                 "high_pstar_oracle_region_summary_10000.csv", "high_pstar_oracle_design_cells.csv",
                 "high_pstar_checks.json"):
        p = OUT / name
        manifest["outputs"].append({"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "elapsed_seconds": manifest["elapsed_seconds"],
                      "outputs": manifest["outputs"], "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
