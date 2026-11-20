"""Compute the preregistered four-fold CRPS selector on saved primary histories."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd

from primary_cv_crps import select_threshold


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_batch(raw: str, start: int, end: int, out: str) -> dict:
    root, destination = Path(raw), Path(out)
    ids = list(range(start, end))
    monthly = pd.read_parquet(root / "monthly_maxima_10000.parquet",
                              filters=[("replication_id", "in", ids)])
    events = pd.read_parquet(root / "event_clusters_10000.parquet",
                             filters=[("replication_id", "in", ids)])
    if set(monthly.replication_id.unique()) != set(ids) or set(events.replication_id.unique()) != set(ids):
        raise AssertionError(f"missing input history in {start}:{end}")
    choices, folds = [], []
    for rep in ids:
        m = monthly.loc[monthly.replication_id.eq(rep)]
        e = events.loc[events.replication_id.eq(rep)]
        try:
            chosen, rows = select_threshold(m, e)
            status = "PASS"
        except ValueError as exc:
            chosen, rows, status = float("nan"), [], str(exc)
        choices.append({"replication_id": rep, "selected_threshold_quantile": chosen,
                        "status": status})
        folds.extend({"replication_id": rep, **row.__dict__} for row in rows)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"cv_{start:05d}_{end-1:05d}"
    choice_path = destination / f"{stem}_choices.csv"
    fold_path = destination / f"{stem}_folds.csv"
    for path, rows in ((choice_path, choices), (fold_path, folds)):
        temp = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(temp, index=False)
        os.replace(temp, path)
    meta = {"start": start, "end": end, "choices": len(choices),
            "folds": len(folds), "choice_sha256": digest(choice_path),
            "fold_sha256": digest(fold_path), "status": "PASS"}
    (destination / f"{stem}_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 10000 or not 1 <= args.workers <= 15:
        parser.error("expected 0 <= start < end <= 10000 and 1..15 workers")
    if args.chunk_size < 1:
        parser.error("chunk size must be positive")
    raw = args.history / "raw"
    manifest = json.loads((args.history / "offline_history_manifest.json").read_text(encoding="utf-8"))
    for label, name in (("monthly_maxima", "monthly_maxima_10000.parquet"),
                        ("event_clusters", "event_clusters_10000.parquet")):
        if digest(raw / name) != manifest["outputs"][label + "_sha256"]:
            raise AssertionError(f"history hash mismatch: {name}")
    started = time.perf_counter()
    chunks = [(lo, min(lo + args.chunk_size, args.end))
              for lo in range(args.start, args.end, args.chunk_size)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_batch, str(raw), lo, hi, str(args.output)): (lo, hi)
                   for lo, hi in chunks}
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            print(f"[CV PASS] {item['start']}:{item['end']}", flush=True)
    result = {"status": "PASS", "start": args.start, "end": args.end,
              "workers": args.workers, "chunk_size": args.chunk_size,
              "history_manifest_sha256": digest(args.history / "offline_history_manifest.json"),
              "duration_seconds": time.perf_counter() - started,
              "chunks": sorted(results, key=lambda row: row["start"])}
    (args.output / "run_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "histories": args.end - args.start,
                      "folds": sum(row["folds"] for row in results),
                      "duration_seconds": result["duration_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
