"""Resumable, chunked Parquet truth-library construction for Gate 3."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from track_a.generator.eaf_b6 import GeneratorConfig, simulate_month
from track_a.io import read_json, sha256_file, write_json
from track_a.rng import make_seed_streams


PHYSICAL_MONTH_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("physical_scenario_id", pa.string()),
        ("month_id", pa.int64()),
        ("pool_role", pa.string()),
        ("pool_month_id", pa.int64()),
        ("M_fixed15_kw", pa.float64()),
        ("M_sliding15_kw", pa.float64()),
        ("energy_kwh", pa.float64()),
        ("cluster_count", pa.int64()),
        ("rare_heat_count", pa.int64()),
        ("q999_violation", pa.bool_()),
        ("max_eaf_mw", pa.float64()),
        ("max_poc_mw", pa.float64()),
        ("heat_count", pa.int64()),
        ("seed_root", pa.int64()),
        ("seed_namespace", pa.string()),
        ("seed_ancestry", pa.string()),
        ("config_hash", pa.string()),
        ("run_signature", pa.string()),
        ("run_mode", pa.string()),
    ]
)

HEAT_CLUSTER_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("physical_scenario_id", pa.string()),
        ("month_id", pa.int64()),
        ("pool_role", pa.string()),
        ("pool_month_id", pa.int64()),
        ("heat_id", pa.int64()),
        ("start_time", pa.int64()),
        ("start_minute", pa.int64()),
        ("duration_min", pa.int64()),
        ("cluster_max_kw", pa.float64()),
        ("cluster_max_sliding_kw", pa.float64()),
        ("rare_event_flag", pa.bool_()),
        ("rare_cluster_id", pa.int64()),
        ("rare_event_family", pa.string()),
        ("rare_severity", pa.float64()),
        ("rare_energy_cap_applied", pa.bool_()),
        ("seed_root", pa.int64()),
        ("seed_namespace", pa.string()),
        ("seed_ancestry", pa.string()),
        ("config_hash", pa.string()),
        ("run_signature", pa.string()),
        ("run_mode", pa.string()),
    ]
)

SEED_REGISTRY_SCHEMA = pa.schema(
    [
        ("physical_scenario_id", pa.string()),
        ("pool_role", pa.string()),
        ("pool_month_id", pa.int64()),
        ("context_id", pa.string()),
        ("stream_name", pa.string()),
        ("seed_root", pa.int64()),
        ("namespace", pa.string()),
        ("parent_entropy", pa.string()),
        ("spawn_key", pa.string()),
        ("seed_namespace", pa.string()),
        ("seed_ancestry", pa.string()),
        ("config_hash", pa.string()),
        ("run_signature", pa.string()),
        ("run_mode", pa.string()),
    ]
)


def load_truth_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)["truth_library"]
    scenarios = sorted(spec["tail_scenarios"], key=lambda row: int(row["rank"]))
    if len(scenarios) != 7 or [int(row["rank"]) for row in scenarios] != list(range(7)):
        raise ValueError("truth library requires exactly seven consecutively ranked tail scenarios")
    previous = None
    for row in scenarios:
        vector = (
            float(row["cluster_rate_per_month"]),
            float(row["headroom_fraction"]),
            float(row["consecutive_probability"]),
        )
        if previous is not None and any(current < prior for current, prior in zip(vector, previous, strict=True)):
            raise ValueError("tail scenario parameters must be componentwise non-decreasing")
        previous = vector
    if len(spec["utilization_grid"]) != 7:
        raise ValueError("truth library requires seven utilization targets")
    for mode in ("smoke", "full"):
        if int(spec["training_pool_months"][mode]) <= 0 or int(spec["test_pool_months"][mode]) <= 0:
            raise ValueError("training and test pools must both contain positive month counts")
    return spec


def build_scenario_registry(
    spec: dict[str, Any], config_hash: str, run_signature: str, mode: str
) -> list[dict[str, Any]]:
    rows = []
    for tail in sorted(spec["tail_scenarios"], key=lambda row: int(row["rank"])):
        for utilization in spec["utilization_grid"]:
            utilization_value = float(utilization)
            rows.append(
                {
                    "scenario_id": f"E3_{tail['id']}_U{utilization_value:.3f}",
                    "physical_scenario_id": str(tail["id"]),
                    "policy_id": "YUNNAN_2026_110KV",
                    "dgp_id": f"B6_{tail['id']}_BOUNDED_BETA",
                    "utilization_target": utilization_value,
                    "tail_scenario": str(tail["id"]),
                    "tail_rank": int(tail["rank"]),
                    "cluster_rate_per_month": float(tail["cluster_rate_per_month"]),
                    "headroom_fraction": float(tail["headroom_fraction"]),
                    "consecutive_probability": float(tail["consecutive_probability"]),
                    "rare_family": str(tail["family"]),
                    "background_id": "BACKGROUND_20MW_DIURNAL4",
                    "flexibility_id": "NO_FLEX",
                    "seed_root": int(spec["seed_root"]),
                    "seed_ancestry": "common physical namespace by month; tail scenario recorded separately",
                    "config_hash": config_hash,
                    "run_signature": run_signature,
                    "run_mode": mode,
                }
            )
    return rows


def write_parquet_atomic(
    rows: list[dict[str, Any]],
    path: str | Path,
    compression: str,
    schema: pa.Schema | None = None,
) -> tuple[int, str]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, temporary, compression=compression)
    os.replace(temporary, destination)
    return table.num_rows, sha256_file(destination)


def _valid_done_record(done_path: Path, run_signature: str) -> dict[str, Any] | None:
    if not done_path.exists():
        return None
    record = read_json(done_path)
    if record.get("run_signature") != run_signature or record.get("status") != "PASS":
        return None
    for artifact in record.get("artifacts", []):
        path = Path(artifact["path"])
        if not path.exists() or sha256_file(path) != artifact["sha256"]:
            return None
    return record


def _cluster_rows(month: Any, physical_scenario_id: str, common: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    billing = 15
    horizon = month.poc_mw.size
    for heat in month.heats:
        heat_end = min(heat.start_minute + heat.duration_min, horizon)
        fixed_first = heat.start_minute // billing
        fixed_last = max(fixed_first, (heat_end - 1) // billing)
        sliding_first = max(0, heat.start_minute - billing + 1)
        sliding_last = min(heat_end - 1, month.sliding15_mw.size - 1)
        rows.append(
            {
                "scenario_id": physical_scenario_id,
                "physical_scenario_id": physical_scenario_id,
                "month_id": month.month_id,
                "heat_id": heat.heat_id,
                "start_time": heat.start_minute,
                "start_minute": heat.start_minute,
                "duration_min": heat.duration_min,
                "cluster_max_kw": float(
                    month.fixed15_mw[fixed_first : fixed_last + 1].max() * 1000.0
                ),
                "cluster_max_sliding_kw": float(
                    month.sliding15_mw[sliding_first : sliding_last + 1].max() * 1000.0
                ),
                "rare_event_flag": heat.rare_event_flag,
                "rare_cluster_id": heat.rare_cluster_id,
                "rare_event_family": heat.rare_family,
                "rare_severity": heat.rare_severity,
                "rare_energy_cap_applied": heat.rare_energy_cap_applied,
                **common,
            }
        )
    return rows


def simulate_truth_chunk(task: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(task["output_root"])
    tail = task["tail"]
    physical_id = str(tail["id"])
    start = int(task["month_start"])
    end = int(task["month_end"])
    part_id = f"part_{start:06d}_{end - 1:06d}"
    done_path = output_root / "parts" / physical_id / f"{part_id}.json"
    valid = _valid_done_record(done_path, str(task["run_signature"]))
    if valid is not None:
        return {**valid, "resumed": True}

    base_cfg = GeneratorConfig.from_yaml(task["generator_config"], task["assumptions_config"])
    cfg = replace(
        base_cfg,
        rare_cluster_rate=float(tail["cluster_rate_per_month"]),
        rare_headroom_fraction=float(tail["headroom_fraction"]),
        consecutive_heat_probability=float(tail["consecutive_probability"]),
        rare_truth_family=str(tail["family"]),
    )
    monthly_rows: list[dict[str, Any]] = []
    heat_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for month_id in range(start, end):
        namespace = [int(task["seed_namespace_prefix"]), month_id]
        bundle = make_seed_streams(int(task["seed_root"]), namespace, f"truth_month_{month_id}")
        training_pool_months = int(task["training_pool_months"])
        pool_role = "train" if month_id < training_pool_months else "test"
        pool_month_id = month_id if pool_role == "train" else month_id - training_pool_months
        common = {
            "pool_role": pool_role,
            "pool_month_id": pool_month_id,
            "seed_root": int(task["seed_root"]),
            "seed_namespace": ".".join(str(value) for value in namespace),
            "seed_ancestry": "common_random_numbers_across_tail_scenarios",
            "config_hash": str(task["config_hash"]),
            "run_signature": str(task["run_signature"]),
            "run_mode": str(task["mode"]),
        }
        month = simulate_month(month_id, cfg, bundle.generators)
        monthly_rows.append(
            {
                "scenario_id": physical_id,
                "physical_scenario_id": physical_id,
                "month_id": month_id,
                "M_fixed15_kw": float(month.fixed15_mw.max() * 1000.0),
                "M_sliding15_kw": float(month.sliding15_mw.max() * 1000.0),
                "energy_kwh": float(month.poc_mw.sum() * 1000.0 / 60.0),
                "cluster_count": month.rare_cluster_count,
                "rare_heat_count": sum(heat.rare_event_flag for heat in month.heats),
                "q999_violation": False,
                "max_eaf_mw": float(month.eaf_mw.max()),
                "max_poc_mw": float(month.poc_mw.max()),
                "heat_count": len(month.heats),
                **common,
            }
        )
        heat_rows.extend(_cluster_rows(month, physical_id, common))
        for row in bundle.registry_rows:
            seed_rows.append({"physical_scenario_id": physical_id, **row, **common})

    compression = str(task["compression"])
    artifacts = []
    for dataset_name, rows, schema in (
        ("physical_truth_months.parquet", monthly_rows, PHYSICAL_MONTH_SCHEMA),
        ("heat_clusters.parquet", heat_rows, HEAT_CLUSTER_SCHEMA),
        ("seed_registry.parquet", seed_rows, SEED_REGISTRY_SCHEMA),
    ):
        path = output_root / dataset_name / physical_id / f"{part_id}.parquet"
        row_count, digest = write_parquet_atomic(rows, path, compression, schema)
        artifacts.append({"path": str(path.resolve()), "rows": row_count, "sha256": digest})
    record = {
        "status": "PASS",
        "physical_scenario_id": physical_id,
        "month_start": start,
        "month_end": end,
        "months": end - start,
        "heat_rows": len(heat_rows),
        "seed_rows": len(seed_rows),
        "run_signature": str(task["run_signature"]),
        "artifacts": artifacts,
        "resumed": False,
    }
    write_json(done_path, record)
    return record
