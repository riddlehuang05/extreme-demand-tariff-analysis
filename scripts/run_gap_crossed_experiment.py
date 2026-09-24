from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

import run_revision_experiment as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "residual_experiments.yaml"
OUTPUT = ROOT / "outputs" / "gap_crossed"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature() -> str:
    digest = hashlib.sha256()
    paths = [Path(__file__), CONFIG, base.CONFIG_PATH, base.DESIGN_PATH, Path(base.__file__)]
    paths.extend(sorted((ROOT / "vendor").rglob("*.py")))
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load() -> tuple[dict[str, Any], dict[str, Any]]:
    revision = base.load_settings()
    residual = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["residual_experiments"]
    if not residual["crossed_tariff"].get("reuse_full_seed_root", False):
        revision["seed_root"] = int(residual["seed_root"])
    return revision, residual


def utilization_grid(settings: dict[str, Any]) -> list[float]:
    start = float(settings["utilization_start"])
    stop = float(settings["utilization_stop"])
    step = float(settings["utilization_step"])
    count = int(round((stop - start) / step)) + 1
    return [round(start + index * step, 10) for index in range(count)]


def build_cells(revision: dict[str, Any], crossed: dict[str, Any]) -> list[dict[str, Any]]:
    oracle = base.OracleDistribution(base.dgp_from_settings(revision, float(revision["dgp"]["poisson_tail_rate_per_month"])))
    tariff = revision["tariff"]
    cells: list[dict[str, Any]] = []
    for alpha in [float(value) for value in crossed["alphas"]]:
        policy = base.Policy(
            f"CROSS_ALPHA{int(round(alpha * 100)):03d}",
            float(tariff["demand_rate"]),
            float(tariff["capacity_rate"]),
            alpha,
            float(crossed["kappa"]),
            float(tariff["delta"]),
            float(tariff["contract_upper_ratio"]),
        )
        for utilization in utilization_grid(crossed):
            s_kva = oracle.mean() / utilization
            decision = base.optimized_decision(oracle, s_kva, policy)
            cells.append(
                {
                    "policy": policy,
                    "cell_id": f"A{int(round(alpha * 100)):03d}_U{int(round(utilization * 1000)):03d}",
                    "utilization": utilization,
                    "S_kva": float(s_kva),
                    "oracle": decision,
                }
            )
    return cells


def write_design(cells: list[dict[str, Any]]) -> tuple[Path, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in cells:
        policy = cell["policy"]
        oracle = cell["oracle"]
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "policy_id": policy.policy_id,
                "alpha": policy.alpha,
                "kappa": policy.kappa,
                "utilization": cell["utilization"],
                "S_kva": cell["S_kva"],
                "oracle_mode": oracle["selected_mode"],
                "oracle_margin_cny": oracle["margin_cny"],
                "oracle_best_value_cny": oracle["best_value_cny"],
                "oracle_contract_D_kw": oracle["contract_D_kw"],
            }
        )
    design = pd.DataFrame(rows)
    design_path = OUTPUT / "crossed_tariff_design.csv"
    design.to_csv(design_path, index=False)

    pair_rows = []
    modes = sorted(design["oracle_mode"].unique())
    for left_index, left_mode in enumerate(modes):
        for right_mode in modes[left_index + 1 :]:
            left = design[design["oracle_mode"] == left_mode]
            right = design[design["oracle_mode"] == right_mode]
            candidates = []
            for lrow in left.itertuples(index=False):
                for rrow in right.itertuples(index=False):
                    denominator = max(float(lrow.oracle_margin_cny), float(rrow.oracle_margin_cny), 1e-12)
                    candidates.append(
                        {
                            "mode_left": left_mode,
                            "cell_left": lrow.cell_id,
                            "margin_left_cny": float(lrow.oracle_margin_cny),
                            "mode_right": right_mode,
                            "cell_right": rrow.cell_id,
                            "margin_right_cny": float(rrow.oracle_margin_cny),
                            "absolute_margin_gap_cny": abs(float(lrow.oracle_margin_cny) - float(rrow.oracle_margin_cny)),
                            "relative_margin_gap": abs(float(lrow.oracle_margin_cny) - float(rrow.oracle_margin_cny)) / denominator,
                        }
                    )
            selected = sorted(candidates, key=lambda row: (row["relative_margin_gap"], row["absolute_margin_gap_cny"]))[:10]
            for rank, row in enumerate(selected, start=1):
                pair_rows.append({"pair_rank_within_mode_pair": rank, **row})
    pairs_path = OUTPUT / "nearest_gap_cell_pairs.csv"
    pd.DataFrame(pair_rows).to_csv(pairs_path, index=False)
    return design_path, pairs_path


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def run_chunk(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    start = int(task["start"])
    end = int(task["end"])
    part = Path(task["output"]) / "parts" / f"part_{start:05d}_{end - 1:05d}.csv"
    meta = part.with_suffix(".json")
    if part.exists() and meta.exists():
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record.get("signature") == task["signature"] and record.get("status") == "PASS":
            return {**record, "resumed": True}
    try:
        revision = task["revision"]
        crossed = task["crossed"]
        cells = build_cells(revision, crossed)
        tail_rate = float(revision["dgp"]["poisson_tail_rate_per_month"])
        rows: list[dict[str, Any]] = []
        for replication in range(start, end):
            rows.extend(base.run_replication(replication, tail_rate, 0, revision, "full", cells))
        frame = pd.DataFrame(rows)
        atomic_csv(frame, part)
        record = {
            "status": "PASS",
            "signature": task["signature"],
            "start": start,
            "end": end,
            "rows": int(len(frame)),
            "csv": str(part.resolve()),
            "sha256": sha256(part),
            "duration_seconds": time.perf_counter() - started,
            "resumed": False,
        }
        meta.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    except Exception as exc:
        return {
            "status": "FAIL",
            "signature": task["signature"],
            "start": start,
            "end": end,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "duration_seconds": time.perf_counter() - started,
        }


def run() -> dict[str, Any]:
    revision, residual = load()
    crossed = residual["crossed_tariff"]
    cells = build_cells(revision, crossed)
    design_path, pairs_path = write_design(cells)
    run_signature = signature()
    total = int(crossed["replications"])
    chunk = int(crossed["chunk_replications"])
    tasks = [
        {
            "output": str(OUTPUT.resolve()),
            "revision": revision,
            "crossed": crossed,
            "signature": run_signature,
            "start": start,
            "end": min(start + chunk, total),
        }
        for start in range(0, total, chunk)
    ]
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=int(crossed["workers"])) as executor:
        futures = [executor.submit(run_chunk, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"[gap-crossed] {record['start']}:{record['end']} {record['status']} {record['duration_seconds']:.1f}s", flush=True)
    failures = [record for record in records if record["status"] != "PASS"]
    if failures:
        payload = {"status": "FAIL", "signature": run_signature, "failures": failures}
        (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise RuntimeError(f"worker failure: {failures[0]}")

    records.sort(key=lambda record: record["start"])
    frame = pd.concat([pd.read_csv(record["csv"]) for record in records], ignore_index=True)
    result_path = OUTPUT / "gap_crossed_results.csv"
    atomic_csv(frame, result_path)
    expected_rows = total * 5 * len(cells)
    safe = frame[frame["d_mv_below_one"].astype(bool)]
    metrics = ["oracle_margin_cny", "d_mv", "mean_relative_error", "q99_relative_error", "stoploss_relative_error", "regret_cny_per_month"]
    checks = {
        "expected_row_count": len(frame) == expected_rows,
        "unique_keys": not frame.duplicated(["replication_id", "method_id", "cell_id"]).any(),
        "all_primary_metrics_finite": bool(np.isfinite(frame[metrics].to_numpy(dtype=float)).all()),
        "d_mv_certificate_no_violations": bool(safe["action_correct"].astype(bool).all()),
        "deterministic_final_functionals": bool((frame["draw_count_for_final_functionals"] == 0).all()),
        "all_oracle_modes_present": len(set(frame["oracle_mode"])) == 3,
    }
    summary_table = frame.groupby(["alpha", "utilization", "cell_id", "oracle_mode", "method_id"], as_index=False).agg(
        action_accuracy=("action_correct", "mean"),
        mean_regret_cny=("regret_cny_per_month", "mean"),
        mean_regret_over_oracle=("regret_over_oracle_value", "mean"),
        mean_regret_over_capacity=("regret_over_capacity_fee", "mean"),
        mean_d_mv=("d_mv", "mean"),
        oracle_margin_cny=("oracle_margin_cny", "first"),
    )
    summary_path = OUTPUT / "gap_crossed_cell_summary.csv"
    atomic_csv(summary_table, summary_path)
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "signature": run_signature,
        "base_runner_signature": base.signature(),
        "replications": total,
        "cells": len(cells),
        "rows": int(len(frame)),
        "alphas": [float(value) for value in crossed["alphas"]],
        "utilizations": utilization_grid(crossed),
        "checks": checks,
        "duration_seconds": time.perf_counter() - started,
        "result_sha256": sha256(result_path),
        "design_sha256": sha256(design_path),
        "gap_pairs_sha256": sha256(pairs_path),
        "cell_summary_sha256": sha256(summary_path),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(f"gap-crossed verification failed: {checks}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-only", action="store_true")
    args = parser.parse_args()
    revision, residual = load()
    cells = build_cells(revision, residual["crossed_tariff"])
    design_path, pairs_path = write_design(cells)
    if args.design_only:
        print(json.dumps({"cells": len(cells), "design": str(design_path), "gap_pairs": str(pairs_path)}, indent=2))
        return
    run()


if __name__ == "__main__":
    main()
