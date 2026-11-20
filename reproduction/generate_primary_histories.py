#!/usr/bin/env python3
"""Generate the deterministic primary synthetic histories from the paper design.

The published analysis used seed_root=20261001 and namespace
(2, 0, replication_id, 10). This script writes the monthly maxima and event
clusters consumed by the revised primary runner without distributing the raw
history archive.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_engine(path: Path):
    sys.path.insert(0, str(path.parent.parent / "src"))
    spec = importlib.util.spec_from_file_location("cssc_generation_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=10000)
    parser.add_argument("--seed-root", type=int, default=20261001)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    engine = load_engine(root / "exp1/code/legacy_engine/scripts/run_revision_experiment.py")
    settings = engine.load_settings()
    settings["training_months"] = 24
    dgp = engine.dgp_from_settings(settings, 30.0)
    output = args.output
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    monthly_path = raw / "monthly_maxima_10000.parquet"
    event_path = raw / "event_clusters_10000.parquet"
    monthly_writer = event_writer = None
    monthly_rows = event_rows = 0
    try:
        for start in range(0, args.replications, 50):
            monthly_parts, event_parts = [], []
            for replication in range(start, min(start + 50, args.replications)):
                rng = engine.seed_rng(args.seed_root, 2, 0, replication, 10)
                monthly, events = engine.generate_event_history(24, dgp, rng)
                monthly.insert(0, "replication_id", replication)
                events.insert(0, "replication_id", replication)
                monthly["seed_root"] = args.seed_root
                events["seed_root"] = args.seed_root
                monthly_parts.append(monthly)
                event_parts.append(events)
            month_frame = pd.concat(monthly_parts, ignore_index=True)
            event_frame = pd.concat(event_parts, ignore_index=True)
            month_table = pa.Table.from_pandas(month_frame, preserve_index=False)
            event_table = pa.Table.from_pandas(event_frame, preserve_index=False)
            if monthly_writer is None:
                monthly_writer = pq.ParquetWriter(monthly_path, month_table.schema)
                event_writer = pq.ParquetWriter(event_path, event_table.schema)
            monthly_writer.write_table(month_table)
            event_writer.write_table(event_table)
            monthly_rows += len(month_frame)
            event_rows += len(event_frame)
    finally:
        if monthly_writer is not None:
            monthly_writer.close()
        if event_writer is not None:
            event_writer.close()
    config = root / "exp1/code/legacy_engine/configs/revision.yaml"
    input_payload = {
        "generator": "base_engine.generate_event_history",
        "generator_script_sha256": sha256(root / "exp1/code/legacy_engine/scripts/run_revision_experiment.py"),
        "revision_yaml_sha256": sha256(config),
        "seed_root": args.seed_root,
        "replications": args.replications,
        "training_months": 24,
        "rng_namespace": [2, 0, "replication_id", 10],
        "dgp_settings": settings["dgp"],
        "tail_rate_per_month": 30.0,
    }
    input_signature = hashlib.sha256(json.dumps(
        input_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    manifest = {
        "status": "PASS",
        "seed_root": args.seed_root,
        "replications": args.replications,
        "training_months": 24,
        "tail_rate_per_month": 30.0,
        "rng_namespace": [2, 0, "replication_id", 10],
        "monthly_maxima_rows": monthly_rows,
        "event_cluster_rows": event_rows,
        "config_sha256": sha256(config),
        "input_signature_payload": input_payload,
        "input_signature": input_signature,
        "outputs": {
            "monthly_maxima": "raw/monthly_maxima_10000.parquet",
            "event_clusters": "raw/event_clusters_10000.parquet",
            "monthly_maxima_sha256": sha256(monthly_path),
            "event_clusters_sha256": sha256(event_path),
        },
    }
    (output / "offline_history_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
