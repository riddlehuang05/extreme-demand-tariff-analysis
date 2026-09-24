"""Resumable paired outer Monte Carlo for the finite-history comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from track_a.decision import (
    EstimatedDecision,
    decision_from_draws,
    patent_history_decision,
    tai_margin_decision,
)
from track_a.estimators import (
    FitResult,
    fit_empirical,
    fit_gev,
    fit_kde,
    fit_point,
    fit_process_tail,
)
from track_a.io import read_json, sha256_file, write_csv, write_json
from track_a.metrics import paired_result_row
from track_a.policy import PolicyConfig, capacity_fee
from track_a.rng import make_seed_streams
from track_a.simulation.precision import precision_rows


CORE_METHODS = (
    "POINT-3M",
    "PATENT-HIST-3M",
    "TAI-MARGIN",
    "KDE-MC-JOINT",
    "EMP-JOINT",
    "B2-CONTRACT",
    "TAIL-JOINT",
    "TAIL-NODECLUSTER",
    "GEV-JOINT",
)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path, compression: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression=compression)
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "rows": len(frame), "sha256": sha256_file(path)}


def _valid_batch(done_path: Path, run_signature: str) -> dict[str, Any] | None:
    if not done_path.exists():
        return None
    record = read_json(done_path)
    if record.get("status") != "PASS" or record.get("run_signature") != run_signature:
        return None
    for artifact in record.get("artifacts", []):
        path = Path(artifact["path"])
        if not path.exists() or sha256_file(path) != artifact["sha256"]:
            return None
    return record


def _history(
    monthly: pd.DataFrame,
    clusters: pd.DataFrame,
    selected_pool_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = {int(pool_id): index for index, pool_id in enumerate(selected_pool_ids)}
    history = monthly.set_index("pool_month_id").loc[selected_pool_ids].reset_index().copy()
    history["month_id"] = history["pool_month_id"].map(order).astype(int)
    history = history.sort_values("month_id")
    event_history = clusters[clusters["pool_month_id"].isin(selected_pool_ids)].copy()
    event_history["month_id"] = event_history["pool_month_id"].map(order).astype(int)
    event_history = event_history.sort_values(["month_id", "heat_id"])
    return history, event_history


def _fit_probabilistic_methods(
    history: pd.DataFrame,
    event_history: pd.DataFrame,
    n_draws: int,
    settings: dict[str, Any],
    estimator_settings: dict[str, Any],
    mode: str,
    method_bundles: dict[str, Any],
) -> list[FitResult]:
    maxima = history["M_fixed15_kw"].to_numpy(dtype=float)
    tail_settings = estimator_settings["tail"]
    fits = [
        fit_point(maxima, n_draws),
        fit_empirical(maxima, n_draws, method_bundles["EMP-JOINT"].generators["method_mc"]),
        fit_kde(maxima, n_draws, method_bundles["KDE-MC-JOINT"].generators["method_mc"]),
    ]
    tail = fit_process_tail(
        history,
        event_history,
        n_draws,
        method_bundles["TAIL-JOINT"].generators["method_mc"],
        method_bundles["TAIL-JOINT"].generators["bootstrap"],
        [float(value) for value in tail_settings["threshold_quantiles"]],
        int(tail_settings["minimum_exceedances_main"]),
        int(tail_settings["minimum_exceedances_fallback"]),
        int(tail_settings["bootstrap_replicates"][mode]),
        float(tail_settings["holdout_fraction"]),
        float(tail_settings["shape_ci_width_gate"]),
        float(tail_settings["physical_poc_bound_kw"]),
        float(tail_settings["truncation_fraction_gate"]),
        int(tail_settings["validation_draws"][mode]),
        float(tail_settings["stability_penalty_weight"]),
        decluster=True,
    )
    fits.append(tail)
    fits.append(
        fit_process_tail(
            history,
            event_history,
            n_draws,
            method_bundles["TAIL-NODECLUSTER"].generators["method_mc"],
            method_bundles["TAIL-NODECLUSTER"].generators["bootstrap"],
            [float(value) for value in tail_settings["threshold_quantiles"]],
            int(tail_settings["minimum_exceedances_main"]),
            int(tail_settings["minimum_exceedances_fallback"]),
            int(tail_settings["bootstrap_replicates"][mode]),
            float(tail_settings["holdout_fraction"]),
            float(tail_settings["shape_ci_width_gate"]),
            float(tail_settings["physical_poc_bound_kw"]),
            float(tail_settings["truncation_fraction_gate"]),
            int(tail_settings["validation_draws"][mode]),
            float(tail_settings["stability_penalty_weight"]),
            decluster=False,
        )
    )
    gev_settings = estimator_settings["gev"]
    fits.append(
        fit_gev(
            maxima,
            n_draws,
            method_bundles["GEV-JOINT"].generators["method_mc"],
            float(gev_settings["physical_poc_bound_kw"]),
            float(gev_settings["truncation_fraction_gate"]),
        )
    )
    fits.append(
        FitResult(
            "B2-CONTRACT",
            tail.draws_kw,
            tail.fit_status,
            tail.threshold,
            tail.xi,
            tail.sigma,
            tail.cluster_rate,
            tail.diagnostics,
        )
    )
    return fits


def _decision_row(
    decision: EstimatedDecision,
    oracle_margin: float,
    keys: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **keys,
        "method_id": decision.method_id,
        "selected_mode": decision.selected_mode,
        "D_kw": decision.D_kw,
        "estimated_cost": decision.estimated_cost,
        "oracle_margin": oracle_margin,
        "tie_set": json.dumps(decision.tie_set),
        "mode_costs": json.dumps(decision.mode_costs, sort_keys=True),
        "decision_diagnostics": json.dumps(diagnostics or {}, sort_keys=True),
    }


def _batch_paths(output: Path, physical_id: str, start: int, end: int) -> dict[str, Path]:
    name = f"part_{start:05d}_{end - 1:05d}"
    return {
        "fits": output / "fits.parquet" / physical_id / f"{name}.parquet",
        "decisions": output / "decisions.parquet" / physical_id / f"{name}.parquet",
        "paired": output / "paired_results.parquet" / physical_id / f"{name}.parquet",
        "seeds": output / "seed_registry.parquet" / physical_id / f"{name}.parquet",
        "done": output / "parts" / physical_id / f"{name}.json",
    }


def run_tail_outer(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(task["output_root"])
    physical_id = str(task["physical_scenario_id"])
    mode = str(task["mode"])
    experiment = task["experiment_settings"]
    estimators = task["estimator_settings"]
    cfg = PolicyConfig.from_yaml(task["policy_config"])
    monthly_path = Path(task["gate3_root"]) / "physical_truth_months.parquet"
    cluster_path = Path(task["gate3_root"]) / "heat_clusters.parquet"
    monthly_train = pd.read_parquet(
        monthly_path,
        filters=[("physical_scenario_id", "=", physical_id), ("pool_role", "=", "train")],
    ).sort_values("pool_month_id")
    monthly_test = pd.read_parquet(
        monthly_path,
        filters=[("physical_scenario_id", "=", physical_id), ("pool_role", "=", "test")],
    ).sort_values("pool_month_id")
    clusters_train = pd.read_parquet(
        cluster_path,
        filters=[("physical_scenario_id", "=", physical_id), ("pool_role", "=", "train")],
    )
    truth = monthly_test["M_fixed15_kw"].to_numpy(dtype=float)
    training_lengths = [int(value) for value in experiment["training_months"][mode]]
    maximum_history = max(training_lengths)
    if len(monthly_train) < maximum_history:
        if mode == "smoke":
            maximum_history = len(monthly_train)
            training_lengths = [maximum_history]
        else:
            raise ValueError("training pool is shorter than the registered maximum history")
    scenario_rows = task["scenario_rows"]
    scenario_by_id = {str(row["scenario_id"]): row for row in scenario_rows}
    oracle: dict[str, EstimatedDecision] = {}
    oracle_margin: dict[str, float] = {}
    capacity_monthly: dict[str, float] = {}
    tie_fraction = float(experiment["tie_tolerance_fraction_quarterly_capacity_fee"])
    for scenario_id, row in scenario_by_id.items():
        S_kva = float(row["S_kva"])
        tie_epsilon = tie_fraction * 3.0 * float(capacity_fee(S_kva, cfg))
        oracle[scenario_id] = decision_from_draws("DIST-ORACLE", truth, S_kva, cfg, tie_epsilon)
        ordered = sorted(oracle[scenario_id].mode_costs.values())
        oracle_margin[scenario_id] = float(ordered[1] - ordered[0])
        capacity_monthly[scenario_id] = float(capacity_fee(S_kva, cfg))

    initial = int(experiment["outer_replications_initial"][mode])
    batch_size = int(experiment["chunk_replications"][mode])
    maximum = int(experiment["outer_replications_max"][mode])
    compression = str(experiment["parquet_compression"])
    accumulated_parts: list[pd.DataFrame] = []
    resumed_batches = 0
    completed_replications = 0
    precision = []
    precision_passed = False
    failed_fit_rows = 0

    for batch_start in range(0, maximum, batch_size):
        batch_end = min(batch_start + batch_size, maximum)
        paths = _batch_paths(output, physical_id, batch_start, batch_end)
        valid = _valid_batch(paths["done"], str(task["run_signature"]))
        if valid is not None:
            resumed_batches += 1
            failed_fit_rows += int(valid.get("failed_fit_rows", 0))
            accumulated_parts.append(pd.read_parquet(paths["paired"]))
            completed_replications = batch_end
        else:
            fit_rows: list[dict[str, Any]] = []
            decision_rows: list[dict[str, Any]] = []
            paired_rows: list[dict[str, Any]] = []
            seed_rows: list[dict[str, Any]] = []
            for replication in range(batch_start, batch_end):
                bundle = make_seed_streams(
                    int(experiment["seed_root"]),
                    [int(experiment["seed_namespace_prefix"]), int(task["tail_rank"]), replication],
                    f"E4_{physical_id}_r{replication}",
                )
                seed_rows.extend(
                    {
                        **row,
                        "physical_scenario_id": physical_id,
                        "replication_id": replication,
                        "run_signature": str(task["run_signature"]),
                        "config_hash": str(task["config_hash"]),
                    }
                    for row in bundle.registry_rows
                )
                selected = bundle.generators["training_draws"].choice(
                    monthly_train["pool_month_id"].to_numpy(dtype=int),
                    size=maximum_history,
                    replace=False,
                )
                for training_months in training_lengths:
                    history_ids = selected[:training_months]
                    history, event_history = _history(monthly_train, clusters_train, history_ids)
                    method_bundles = {}
                    method_namespace = {"POINT-3M": f"{experiment['seed_namespace_prefix']}.{task['tail_rank']}.{replication}.deterministic"}
                    for method_index, method_id in enumerate(
                        ("EMP-JOINT", "KDE-MC-JOINT", "TAIL-JOINT", "TAIL-NODECLUSTER", "GEV-JOINT"),
                        start=1,
                    ):
                        method_bundle = make_seed_streams(
                            int(experiment["seed_root"]),
                            [
                                int(experiment["seed_namespace_prefix"]),
                                int(task["tail_rank"]),
                                replication,
                                training_months,
                                method_index,
                            ],
                            f"E4_{physical_id}_r{replication}_n{training_months}_{method_id}",
                        )
                        method_bundles[method_id] = method_bundle
                        method_namespace[method_id] = (
                            f"{experiment['seed_namespace_prefix']}.{task['tail_rank']}."
                            f"{replication}.{training_months}.{method_index}"
                        )
                        seed_rows.extend(
                            {
                                **row,
                                "physical_scenario_id": physical_id,
                                "replication_id": replication,
                                "training_months": training_months,
                                "method_id": method_id,
                                "run_signature": str(task["run_signature"]),
                                "config_hash": str(task["config_hash"]),
                            }
                            for row in method_bundle.registry_rows
                        )
                    method_namespace["B2-CONTRACT"] = method_namespace["TAIL-JOINT"]
                    fits = _fit_probabilistic_methods(
                        history,
                        event_history,
                        int(experiment["decision_draws"][mode]),
                        experiment,
                        estimators,
                        mode,
                        method_bundles,
                    )
                    for scenario_id, scenario in scenario_by_id.items():
                        S_kva = float(scenario["S_kva"])
                        tie_epsilon = tie_fraction * 3.0 * capacity_monthly[scenario_id]
                        common = {
                            "replication_id": replication,
                            "scenario_id": scenario_id,
                            "physical_scenario_id": physical_id,
                            "training_months": training_months,
                            "utilization_target": float(scenario["utilization_target"]),
                            "S_kva": S_kva,
                            "capacity_fee_monthly": capacity_monthly[scenario_id],
                            "tail_scenario": physical_id,
                            "training_pool_ids": json.dumps([int(value) for value in history_ids]),
                            "seed_root": int(experiment["seed_root"]),
                            "seed_ancestry": f"{experiment['seed_namespace_prefix']}.{task['tail_rank']}.{replication}",
                            "config_hash": str(task["config_hash"]),
                            "run_signature": str(task["run_signature"]),
                        }
                        for fit in fits:
                            method_common = {
                                **common,
                                "seed_ancestry": method_namespace[fit.method_id],
                            }
                            fit_rows.append(fit.as_row(**method_common))
                            decision = decision_from_draws(
                                fit.method_id,
                                fit.draws_kw,
                                S_kva,
                                cfg,
                                tie_epsilon,
                                force_contract=fit.method_id == "B2-CONTRACT",
                            )
                            decision_rows.append(
                                _decision_row(decision, oracle_margin[scenario_id], method_common)
                            )
                            paired_rows.append(
                                paired_result_row(
                                    decision,
                                    oracle[scenario_id],
                                    truth,
                                    S_kva,
                                    cfg,
                                    oracle_margin=oracle_margin[scenario_id],
                                    **method_common,
                                )
                            )
                        patent = patent_history_decision(
                            history["M_fixed15_kw"].to_numpy(dtype=float),
                            S_kva,
                            cfg,
                            estimators["patent_history"]["candidate_summaries"],
                            tie_epsilon,
                        )
                        tai, tai_diagnostics = tai_margin_decision(
                            history["M_fixed15_kw"].to_numpy(dtype=float),
                            S_kva,
                            cfg,
                            estimators["tai_margin"]["beta_grid"],
                            float(estimators["tai_margin"]["residual_quantile"]),
                            int(estimators["tai_margin"]["minimum_inner_train_months"]),
                            tie_epsilon,
                        )
                        for decision, diagnostics in ((patent, {}), (tai, tai_diagnostics)):
                            deterministic_common = {
                                **common,
                                "seed_ancestry": (
                                    f"{experiment['seed_namespace_prefix']}.{task['tail_rank']}."
                                    f"{replication}.deterministic"
                                ),
                            }
                            fit_rows.append(
                                {
                                    **deterministic_common,
                                    "method_id": decision.method_id,
                                    "fit_status": "PASS",
                                    "threshold": None,
                                    "xi": None,
                                    "sigma": None,
                                    "cluster_rate": None,
                                    "draw_count": 0,
                                    "draw_mean_kw": None,
                                    "draw_q95_kw": None,
                                    "draw_q99_kw": None,
                                    "diagnostics": json.dumps(diagnostics, sort_keys=True),
                                }
                            )
                            decision_rows.append(
                                _decision_row(decision, oracle_margin[scenario_id], deterministic_common, diagnostics)
                            )
                            paired_rows.append(
                                paired_result_row(
                                    decision,
                                    oracle[scenario_id],
                                    truth,
                                    S_kva,
                                    cfg,
                                    oracle_margin=oracle_margin[scenario_id],
                                    **deterministic_common,
                                )
                            )
            frames = {
                "fits": pd.DataFrame(fit_rows),
                "decisions": pd.DataFrame(decision_rows),
                "paired": pd.DataFrame(paired_rows),
                "seeds": pd.DataFrame(seed_rows),
            }
            artifacts = [
                _write_parquet_atomic(frames[name], paths[name], compression)
                for name in ("fits", "decisions", "paired", "seeds")
            ]
            batch_failed_fits = sum(
                str(value).startswith("FAIL") for value in frames["fits"]["fit_status"]
            )
            failed_fit_rows += batch_failed_fits
            write_json(
                paths["done"],
                {
                    "status": "PASS",
                    "physical_scenario_id": physical_id,
                    "replication_start": batch_start,
                    "replication_end": batch_end,
                    "run_signature": str(task["run_signature"]),
                    "failed_fit_rows": batch_failed_fits,
                    "artifacts": artifacts,
                },
            )
            accumulated_parts.append(frames["paired"])
            completed_replications = batch_end

        if completed_replications >= initial:
            accumulated = pd.concat(accumulated_parts, ignore_index=True)
            precision = precision_rows(
                accumulated,
                capacity_monthly,
                [str(value) for value in experiment["comparison_baselines"]],
                [str(value) for value in experiment["precision_methods"]],
                float(experiment["confidence_level"]),
                float(experiment["mode_error_ci_halfwidth"]),
                float(experiment["regret_gain_relative_halfwidth"]),
                float(experiment["regret_gain_floor_fraction_monthly_capacity_fee"]),
            )
            write_csv(
                output / "precision" / f"{physical_id}.csv",
                precision,
                list(precision[0]),
            )
            precision_passed = all(bool(row["passed"]) for row in precision)
            if mode == "smoke" or precision_passed:
                break

    if failed_fit_rows:
        status = "FAIL_ESTIMATOR_GATE"
    else:
        status = "PASS" if mode == "smoke" or precision_passed else "FAIL_PRECISION_MAX"
    summary = {
        "status": status,
        "physical_scenario_id": physical_id,
        "completed_replications": completed_replications,
        "resumed_batches": resumed_batches,
        "precision_passed": precision_passed,
        "failed_precision_rows": sum(not bool(row["passed"]) for row in precision),
        "failed_fit_rows": failed_fit_rows,
        "run_signature": str(task["run_signature"]),
        "duration_seconds": time.perf_counter() - started,
        "method_ids": list(CORE_METHODS),
    }
    write_json(output / "tail_summaries" / f"{physical_id}.json", summary)
    return summary
