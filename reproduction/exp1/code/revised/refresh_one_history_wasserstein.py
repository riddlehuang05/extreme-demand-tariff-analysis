"""Recompute stored-law numerical stability outputs for corrected history 8185."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-fits", type=Path, required=True)
    parser.add_argument("--new-fits", type=Path, required=True)
    parser.add_argument("--corrected-decisions", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--law-script", type=Path, required=True)
    parser.add_argument("--base-repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    old = [json.loads(x) for x in args.old_fits.read_text(encoding="utf-8").splitlines() if x]
    new = [json.loads(x) for x in args.new_fits.read_text(encoding="utf-8").splitlines() if x]
    old = [x for x in old if int(x["replication_id"]) == 8185]
    new_tail = [x for x in new if int(x["replication_id"]) == 8185 and x["method_id"] == "TAIL"]
    if len(old) != 5 or len(new_tail) != 1:
        raise AssertionError("expected five archived laws and one corrected TAIL law")
    corrected = {x["method_id"]: x for x in new}
    records = [corrected[x["method_id"]] if x["method_id"] == "TAIL" else x for x in old]
    fits_path = args.work / "rep_08185_08185_fits.jsonl"
    fits_path.write_text("".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in records),
                         encoding="utf-8")
    decisions = pd.read_csv(args.corrected_decisions)
    if len(decisions) != 45 or decisions.replication_id.nunique() != 1:
        raise AssertionError("corrected one-history decision table is incomplete")
    csv_path = args.work / "rep_08185_08185.csv"
    decisions.to_csv(csv_path, index=False)
    primary_path = args.work / "primary_08185.parquet"
    decisions.to_parquet(primary_path, index=False)
    meta_path = args.work / "rep_08185_08185_meta.json"
    meta_path.write_text(json.dumps({"status": "PASS", "start": 8185, "end": 8186,
                                     "fits_sha256": sha(fits_path), "csv_sha256": sha(csv_path)}), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("cssc_corrected_primitive", args.audit_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import numerical stability engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.init_worker(str(args.base_repo), str(args.law_script),
                       128, 512, 64)
    record = module.process_chunk((str(fits_path), str(meta_path), str(csv_path), str(primary_path),
                                   str(args.work), 8185, 8186, sha(fits_path), sha(csv_path)))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
