from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import yaml

from run_revision_experiment import (
    ACTUAL,
    CAPACITY,
    CONTRACT,
    DGP,
    OracleDistribution,
    Policy,
    dgp_from_settings,
    distribution_builds,
    generate_event_history,
    load_settings,
    optimized_decision,
    policies_and_cells,
    seed_rng,
    true_cost_of_estimated_action,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "regime_shift_sensitivity.yaml"
BASE_CONFIG = ROOT / "configs" / "revision.yaml"
OUTPUT_ROOT = ROOT / "ablations" / "ASMBI_REGIME_SHIFT_V1"
TABLES = ROOT / "tables"


def load_design() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["regime_shift_sensitivity"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature() -> str:
    digest = hashlib.sha256()
    paths = [Path(__file__), CONFIG, BASE_CONFIG, ROOT / "scripts" / "run_revision_experiment.py"]
    paths.extend(sorted((ROOT / "src" / "extreme_demand").rglob("*.py")))
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def history_checksum(monthly: pd.DataFrame, events: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(monthly.sort_values("month_id").to_csv(index=False).encode("utf-8"))
    digest.update(events.sort_values(["month_id", "heat_id"]).to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def generate_piecewise_history(
    replication: int,
    scenario: dict[str, Any],
    target_dgp: DGP,
    design: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    early_months = int(design["early_months"])
    total_months = int(design["training_months"])
    late_months = total_months - early_months
    early_dgp = replace(target_dgp, **scenario.get("early_changes", {}))
    seed_root = int(design["seed_root"])
    early_monthly, early_events = generate_event_history(
        early_months,
        early_dgp,
        seed_rng(seed_root, 71, replication, 10),
    )
    late_monthly, late_events = generate_event_history(
        late_months,
        target_dgp,
        seed_rng(seed_root, 71, replication, 11),
    )
    checksum = history_checksum(late_monthly, late_events)
    late_monthly = late_monthly.copy()
    late_events = late_events.copy()
    late_monthly["month_id"] += early_months
    late_events["month_id"] += early_months
    monthly = pd.concat([early_monthly, late_monthly], ignore_index=True).sort_values("month_id")
    events = pd.concat([early_events, late_events], ignore_index=True).sort_values(["month_id", "heat_id"])
    return monthly.reset_index(drop=True), events.reset_index(drop=True), checksum


def evaluate_replication(
    replication: int,
    scenario: dict[str, Any],
    design: dict[str, Any],
    settings: dict[str, Any],
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_dgp = dgp_from_settings(settings)
    oracle = OracleDistribution(target_dgp)
    monthly, events, late_checksum = generate_piecewise_history(
        replication, scenario, target_dgp, design
    )
    physical_bound = 10.0 * oracle.ppf(0.999)
    builds = distribution_builds(
        monthly,
        events,
        settings,
        "frequency",
        int(design["seed_root"]),
        (72, replication, 20),
        physical_bound,
    )
    early_dgp = replace(target_dgp, **scenario.get("early_changes", {}))
    true_mean = oracle.mean()
    true_q99 = oracle.ppf(0.99)
    rows: list[dict[str, Any]] = []
    for method_id, information_set, model_class, distribution, diagnostics in builds:
        estimated_mean = distribution.mean()
        estimated_q99 = distribution.ppf(0.99)
        for cell in cells:
            policy: Policy = cell["policy"]
            oracle_decision = cell["oracle"]
            s_kva = float(cell["S_kva"])
            estimated = optimized_decision(distribution, s_kva, policy)
            oracle_costs = oracle_decision["mode_costs"]
            mode_errors = {
                name: abs(float(estimated["mode_costs"][name]) - float(oracle_costs[name]))
                for name in (CAPACITY, ACTUAL, CONTRACT)
            }
            d_mv = 2.0 * max(mode_errors.values()) / float(oracle_decision["margin_cny"])
            selected_true_cost = true_cost_of_estimated_action(
                estimated["selected_mode"],
                estimated["contract_D_kw"],
                oracle,
                s_kva,
                policy,
            )
            regret = max(0.0, selected_true_cost - float(oracle_decision["best_value_cny"]))
            oracle_threshold = policy.alpha * float(oracle_decision["contract_D_kw"])
            true_stoploss = oracle.stop_loss(oracle_threshold)
            estimated_stoploss = distribution.stop_loss(oracle_threshold)
            row = {
                "task_id": str(design["task_id"]),
                "scenario_id": str(scenario["id"]),
                "replication_id": replication,
                "early_months": int(design["early_months"]),
                "late_months": int(design["training_months"]) - int(design["early_months"]),
                "early_tail_rate": early_dgp.tail_rate,
                "target_tail_rate": target_dgp.tail_rate,
                "early_xi": early_dgp.xi,
                "target_xi": target_dgp.xi,
                "early_sigma_kw": early_dgp.sigma_kw,
                "target_sigma_kw": target_dgp.sigma_kw,
                "late_segment_sha256": late_checksum,
                "method_id": method_id,
                "information_set": information_set,
                "model_class": model_class,
                "fit_status": diagnostics.get("fit_status", "PASS"),
                "policy_id": policy.policy_id,
                "alpha": policy.alpha,
                "kappa": policy.kappa,
                "cell_id": cell["cell_id"],
                "utilization": cell["utilization"],
                "S_kva": s_kva,
                "oracle_mode": oracle_decision["selected_mode"],
                "selected_mode": estimated["selected_mode"],
                "action_correct": bool(estimated["selected_mode"] == oracle_decision["selected_mode"]),
                "oracle_margin_cny": oracle_decision["margin_cny"],
                "oracle_best_value_cny": oracle_decision["best_value_cny"],
                "estimated_best_value_cny": estimated["best_value_cny"],
                "maximum_mode_value_error_cny": max(mode_errors.values()),
                "d_mv": d_mv,
                "d_mv_below_one": bool(d_mv < 1.0),
                "true_mean_kw": true_mean,
                "estimated_mean_kw": estimated_mean,
                "mean_relative_error": (estimated_mean - true_mean) / true_mean,
                "true_q99_kw": true_q99,
                "estimated_q99_kw": estimated_q99,
                "q99_relative_error": (estimated_q99 - true_q99) / true_q99,
                "oracle_contract_D_kw": oracle_decision["contract_D_kw"],
                "estimated_contract_D_kw": estimated["contract_D_kw"],
                "contract_D_error_over_S": (
                    estimated["contract_D_kw"] - oracle_decision["contract_D_kw"]
                )
                / s_kva,
                "stoploss_threshold_kw": oracle_threshold,
                "true_stoploss_kw": true_stoploss,
                "estimated_stoploss_kw": estimated_stoploss,
                "stoploss_relative_error": (estimated_stoploss - true_stoploss) / true_stoploss,
                "regret_cny_per_month": regret,
                "regret_over_oracle_value": regret / float(oracle_decision["best_value_cny"]),
                "regret_over_capacity_fee": regret / (policy.capacity_rate * s_kva),
                "draw_count_for_final_functionals": 0,
                "final_functional_evaluation": "deterministic",
                "selected_threshold_quantile": diagnostics.get("selected_threshold_quantile", np.nan),
                "threshold_kw": diagnostics.get("threshold_kw", np.nan),
                "xi_hat": diagnostics.get("xi_hat", np.nan),
                "sigma_hat_kw": diagnostics.get("sigma_hat_kw", np.nan),
                "count_model": diagnostics.get("count_model", "not_applicable"),
                "fallback_reason": diagnostics.get("fallback_reason"),
                "refit_used_all_months": diagnostics.get("refit_used_all_months", False),
                "finite_mean_gate_pass": diagnostics.get("finite_mean_gate_pass", True),
                "finite_variance_flag": diagnostics.get("finite_variance_flag", True),
            }
            rows.append(row)
    return rows


def run_chunk(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(task["output"])
    scenario = task["scenario"]
    start = int(task["start"])
    end = int(task["end"])
    label = f"{scenario['id']}_{start:04d}_{end - 1:04d}"
    csv_path = output / "parts" / f"part_{label}.csv"
    meta_path = output / "parts" / f"part_{label}.json"
    if csv_path.exists() and meta_path.exists():
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        if record.get("signature") == task["signature"] and record.get("status") == "PASS":
            return {**record, "resumed": True}
    try:
        settings = task["settings"]
        target_dgp = dgp_from_settings(settings)
        oracle = OracleDistribution(target_dgp)
        cells = policies_and_cells(settings, oracle, False)
        rows: list[dict[str, Any]] = []
        for replication in range(start, end):
            rows.extend(
                evaluate_replication(
                    replication,
                    scenario,
                    task["design"],
                    settings,
                    cells,
                )
            )
        frame = pd.DataFrame(rows)
        atomic_csv(frame, csv_path)
        record = {
            "status": "PASS",
            "signature": task["signature"],
            "scenario_id": scenario["id"],
            "start": start,
            "end": end,
            "rows": int(len(frame)),
            "csv": str(csv_path.resolve()),
            "sha256": sha256(csv_path),
            "duration_seconds": time.perf_counter() - started,
            "resumed": False,
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = meta_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(temporary, meta_path)
        return record
    except Exception as exc:
        return {
            "status": "FAIL",
            "signature": task["signature"],
            "scenario_id": scenario["id"],
            "start": start,
            "end": end,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "duration_seconds": time.perf_counter() - started,
        }


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    indexes = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(frame: pd.DataFrame, design: dict[str, Any], output: Path) -> dict[str, Path]:
    method_summary = (
        frame.assign(
            absolute_q99_relative_error=frame["q99_relative_error"].abs(),
            # Successful GEV fits carry structured statuses such as
            # PASS_LMOM_FINITE_MEAN; only explicit FALLBACK statuses count.
            fallback=frame["fit_status"].fillna("").astype(str).str.startswith("FALLBACK").astype(float),
        )
        .groupby(["scenario_id", "method_id"], as_index=False)
        .agg(
            action_accuracy=("action_correct", "mean"),
            mean_regret_cny=("regret_cny_per_month", "mean"),
            median_regret_cny=("regret_cny_per_month", "median"),
            p95_regret_cny=("regret_cny_per_month", lambda values: float(np.quantile(values, 0.95))),
            mean_regret_over_oracle=("regret_over_oracle_value", "mean"),
            median_absolute_q99_error=("absolute_q99_relative_error", "median"),
            fallback_row_fraction=("fallback", "mean"),
        )
    )
    aggregate = (
        frame.assign(absolute_q99_relative_error=frame["q99_relative_error"].abs())
        .groupby(["scenario_id", "replication_id", "method_id"], as_index=False)
        .agg(
            mean_regret_cny=("regret_cny_per_month", "mean"),
            mean_regret_over_oracle=("regret_over_oracle_value", "mean"),
            mean_absolute_q99_error=("absolute_q99_relative_error", "mean"),
            action_accuracy=("action_correct", "mean"),
        )
    )
    rng = np.random.default_rng(int(design["seed_root"]) + 9900)
    contrast_rows: list[dict[str, Any]] = []
    metrics = ["mean_regret_cny", "mean_regret_over_oracle", "mean_absolute_q99_error", "action_accuracy"]
    for method_id, group in aggregate.groupby("method_id"):
        for metric in metrics:
            pivot = group.pivot(index="replication_id", columns="scenario_id", values=metric)
            for scenario_id in sorted(value for value in pivot.columns if value != "STATIONARY_TARGET"):
                difference = (pivot[scenario_id] - pivot["STATIONARY_TARGET"]).dropna().to_numpy(dtype=float)
                low, high = bootstrap_interval(
                    difference,
                    rng,
                    int(design["inference_bootstrap_replicates"]),
                )
                contrast_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "method_id": method_id,
                        "metric": metric,
                        "contrast": "shift_minus_stationary_target",
                        "paired_replications": int(difference.size),
                        "mean_difference": float(np.mean(difference)),
                        "median_difference": float(np.median(difference)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_replicates": int(design["inference_bootstrap_replicates"]),
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    method_path = output / "method_summary.csv"
    aggregate_path = output / "replication_method_aggregate.csv"
    contrast_path = output / "paired_shift_contrasts.csv"
    atomic_csv(method_summary, method_path)
    atomic_csv(aggregate, aggregate_path)
    atomic_csv(contrasts, contrast_path)
    TABLES.mkdir(parents=True, exist_ok=True)
    atomic_csv(method_summary, TABLES / "regime_shift_method_summary.csv")
    atomic_csv(contrasts, TABLES / "regime_shift_paired_contrasts.csv")
    return {"method_summary": method_path, "aggregate": aggregate_path, "contrasts": contrast_path}


def finalize(
    frame: pd.DataFrame,
    design: dict[str, Any],
    output: Path,
    mode: str,
    started: float,
    data_signature: str,
    summary_code_signature: str,
    summary_only_recomputed: bool = False,
) -> dict[str, Any]:
    """Write summaries and validate an existing regime-shift result table."""
    frame = frame.sort_values(["scenario_id", "replication_id", "method_id", "cell_id"]).reset_index(drop=True)
    result_path = output / "regime_shift_results.csv"
    summary_paths = summarize(frame, design, output)
    total = int(design["replications"]["full"])
    expected_rows = len(design["scenarios"]) * total * 5 * 9
    safe = frame[frame["d_mv_below_one"].astype(bool)]
    late_pairing = frame.groupby("replication_id")["late_segment_sha256"].nunique()
    checks = {
        "expected_row_count": len(frame) == expected_rows,
        "unique_keys": not frame.duplicated(["scenario_id", "replication_id", "method_id", "cell_id"]).any(),
        "all_four_scenarios_present": set(frame["scenario_id"]) == {str(row["id"]) for row in design["scenarios"]},
        "stationary_control_present": "STATIONARY_TARGET" in set(frame["scenario_id"]),
        "paired_late_segment_identical_across_scenarios": bool((late_pairing == 1).all()),
        "all_primary_metrics_finite": bool(
            np.isfinite(
                frame[["oracle_margin_cny", "d_mv", "mean_relative_error", "q99_relative_error", "stoploss_relative_error", "regret_cny_per_month"]].to_numpy(dtype=float)
            ).all()
        ),
        "d_mv_certificate_no_violations": bool(safe["action_correct"].astype(bool).all()),
        "deterministic_final_functionals": bool((frame["draw_count_for_final_functionals"] == 0).all()),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "task_id": str(design["task_id"]),
        "mode": mode,
        # `signature` remains the signature of the raw result-generating code.
        # The separate field records the code used to recompute summaries.
        "signature": data_signature,
        "summary_code_signature": summary_code_signature,
        "summary_only_recomputed": summary_only_recomputed,
        "replications_per_scenario": total,
        "scenarios": [str(row["id"]) for row in design["scenarios"]],
        "methods": sorted(frame["method_id"].unique()),
        "cells": int(frame["cell_id"].nunique()),
        "rows": int(len(frame)),
        "duration_seconds": time.perf_counter() - started,
        "checks": checks,
        "result_sha256": sha256(result_path),
        "summary_sha256": {name: sha256(path) for name, path in summary_paths.items()},
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(f"regime-shift verification failed: {checks}")
    return payload


def run(mode: str) -> dict[str, Any]:
    design = load_design()
    settings = load_settings()
    run_signature = signature()

    if mode == "resummarize":
        output = OUTPUT_ROOT / "full"
        result_path = output / "regime_shift_results.csv"
        if not result_path.exists():
            raise RuntimeError(f"existing regime-shift result table not found: {result_path}")
        part_meta = sorted((output / "parts").glob("*.json"))
        generation_signatures = {
            str(json.loads(path.read_text(encoding="utf-8")).get("signature"))
            for path in part_meta
            if json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"
        }
        if len(generation_signatures) != 1:
            raise RuntimeError(f"raw regime-shift parts do not have one generation signature: {generation_signatures}")
        frame = pd.read_csv(result_path)
        return finalize(
            frame,
            design,
            output,
            mode="full",
            started=time.perf_counter(),
            data_signature=next(iter(generation_signatures)),
            summary_code_signature=run_signature,
            summary_only_recomputed=True,
        )

    output = OUTPUT_ROOT / mode
    output.mkdir(parents=True, exist_ok=True)
    if mode == "full":
        smoke_path = OUTPUT_ROOT / "smoke" / "summary.json"
        if not smoke_path.exists():
            raise RuntimeError("regime-shift smoke run is required before full run")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "PASS" or smoke.get("signature") != run_signature:
            raise RuntimeError("regime-shift smoke run does not match current signature")

    total = int(design["replications"][mode])
    chunk = int(design["chunk_replications"][mode])
    tasks = []
    for scenario in design["scenarios"]:
        for start in range(0, total, chunk):
            tasks.append(
                {
                    "output": str(output.resolve()),
                    "design": design,
                    "settings": settings,
                    "signature": run_signature,
                    "scenario": scenario,
                    "start": start,
                    "end": min(start + chunk, total),
                }
            )
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(design["workers"])) as executor:
        futures = [executor.submit(run_chunk, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[regime:{mode}] {record['scenario_id']} {record['start']}:{record['end']} "
                f"{record['status']} {record['duration_seconds']:.1f}s",
                flush=True,
            )
    failures = [record for record in records if record["status"] != "PASS"]
    if failures:
        payload = {"status": "FAIL", "signature": run_signature, "failures": failures}
        (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise RuntimeError(f"regime-shift worker failure: {failures[0]}")
    records.sort(key=lambda row: (row["scenario_id"], row["start"]))
    frame = pd.concat([pd.read_csv(record["csv"]) for record in records], ignore_index=True)
    result_path = output / "regime_shift_results.csv"
    atomic_csv(frame, result_path)
    return finalize(
        frame,
        design,
        output,
        mode=mode,
        started=started,
        data_signature=run_signature,
        summary_code_signature=run_signature,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "resummarize"], required=True)
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
