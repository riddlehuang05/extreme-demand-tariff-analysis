"""Merge a verified one-history selector correction into the primary archive."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archived-raw", type=Path, required=True)
    parser.add_argument("--original-check", type=Path, required=True)
    parser.add_argument("--corrected-check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    archived = pd.read_csv(args.archived_raw)
    original = pd.read_csv(args.original_check)
    corrected = pd.read_csv(args.corrected_check)
    keys = ["replication_id", "method_id", "cell_id"]
    for frame in (archived, original, corrected):
        if frame.duplicated(keys).any():
            raise AssertionError("duplicate primary decision keys")
    if set(archived.columns) != set(original.columns) or set(archived.columns) != set(corrected.columns):
        raise AssertionError("decision schemas differ")
    if len(archived) != 450000 or set(archived.replication_id.unique()) != set(range(10000)):
        raise AssertionError("archived primary domain is incomplete")
    original_rows = archived[archived.replication_id.eq(8185)].sort_values(keys).reset_index(drop=True)
    original_check = original.sort_values(keys).reset_index(drop=True)
    if not original_rows[keys].equals(original_check[keys]):
        raise AssertionError("original check keys differ")
    for col in archived.columns:
        a, b = original_rows[col], original_check[col]
        if pd.api.types.is_numeric_dtype(a):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-11, atol=1e-8, equal_nan=True):
                raise AssertionError(f"old selector refit differs in {col}")
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            raise AssertionError(f"old selector refit differs in {col}")
    old_other = original[original.method_id.ne("TAIL")].sort_values(keys).reset_index(drop=True)
    new_other = corrected[corrected.method_id.ne("TAIL")].sort_values(keys).reset_index(drop=True)
    for col in archived.columns:
        a, b = old_other[col], new_other[col]
        if pd.api.types.is_numeric_dtype(a):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-11, atol=1e-8, equal_nan=True):
                raise AssertionError(f"unaffected estimator differs in {col}")
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            raise AssertionError(f"unaffected estimator differs in {col}")
    replacement = corrected[corrected.method_id.eq("TAIL")]
    if len(replacement) != 9 or replacement.replication_id.nunique() != 1:
        raise AssertionError("expected one corrected TAIL history and nine cells")
    mask = archived.replication_id.eq(8185) & archived.method_id.eq("TAIL")
    if not archived.loc[mask, keys].sort_values(keys).reset_index(drop=True).equals(
        replacement[keys].sort_values(keys).reset_index(drop=True)
    ):
        raise AssertionError("corrected decision keys differ")
    archived = archived.set_index(keys)
    replacement = replacement.set_index(keys)
    archived.loc[replacement.index, :] = replacement[archived.columns].to_numpy()
    merged = archived.reset_index()
    if len(merged) != 450000 or merged.duplicated(keys).any():
        raise AssertionError("corrected primary domain is incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=6, mtime=0) as compressed:
            compressed.write(merged.to_csv(index=False).encode("utf-8"))
    report = {
        "status": "PASS",
        "archived_raw_sha256": digest(args.archived_raw),
        "original_check_sha256": digest(args.original_check),
        "corrected_check_sha256": digest(args.corrected_check),
        "merged_raw_sha256": digest(args.output),
        "histories": 10000,
        "decision_rows": 450000,
        "corrected_history": 8185,
        "corrected_method": "TAIL",
        "corrected_cells": 9,
        "old_threshold": float(original.loc[original.method_id.eq("TAIL"), "selected_threshold_quantile"].iloc[0]),
        "new_threshold": float(corrected.loc[corrected.method_id.eq("TAIL"), "selected_threshold_quantile"].iloc[0]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
