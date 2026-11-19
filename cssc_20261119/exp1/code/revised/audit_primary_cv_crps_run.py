#!/usr/bin/env python3
"""Verify the complete saved four-fold CV-CRPS selection run."""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["start"] == 0 and manifest["end"] == 10000
    assert len(manifest["chunks"]) == 100
    choices, folds = [], []
    for i, chunk in enumerate(manifest["chunks"]):
        start, end = 100 * i, 100 * (i + 1)
        assert (chunk["start"], chunk["end"], chunk["choices"], chunk["folds"], chunk["status"]) == (start, end, 100, 1600, "PASS")
        stem = f"cv_{start:05d}_{end-1:05d}"
        for suffix, expected_hash, sink in (("choices", "choice_sha256", choices), ("folds", "fold_sha256", folds)):
            path = args.run / f"{stem}_{suffix}.csv"
            assert digest(path) == chunk[expected_hash]
            sink.append(pd.read_csv(path))
    choice = pd.concat(choices, ignore_index=True)
    fold = pd.concat(folds, ignore_index=True)
    assert len(choice) == 10000 and len(fold) == 160000
    assert choice.replication_id.tolist() == list(range(10000))
    assert not choice.replication_id.duplicated().any()
    assert choice.status.eq("PASS").all()
    assert not fold.duplicated(["replication_id", "quantile", "fold"]).any()
    assert set(fold["quantile"].unique()) == {0.9, 0.925, 0.95, 0.975}
    assert set(fold["fold"].unique()) == {0, 1, 2, 3}
    assert fold.groupby("replication_id").size().eq(16).all()
    assert fold["fit_months"].eq(18).all() and fold["validation_months"].eq(6).all()
    valid = fold[fold.status.eq("PASS")]
    candidate = valid.groupby(["replication_id", "quantile"]).agg(mean_crps=("crps", "mean"), valid_folds=("fold", "nunique")).reset_index()
    candidate = candidate[candidate.valid_folds.eq(4)].sort_values(["replication_id", "mean_crps", "quantile"])
    picked = candidate.drop_duplicates("replication_id")[["replication_id", "quantile"]]
    joined = choice.merge(picked, on="replication_id", how="left", validate="one_to_one")
    assert joined["quantile"].notna().all()
    assert np.allclose(joined["selected_threshold_quantile"], joined["quantile"], rtol=0, atol=1e-12)
    output = {
        "status": "PASS", "history_count": int(len(choice)), "candidate_fold_rows": int(len(fold)),
        "complete_valid_candidate_rows": int(len(candidate)),
        "selected_quantile_counts": {str(q): int(n) for q, n in choice.selected_threshold_quantile.value_counts().sort_index().items()},
        "fold_status_counts": {str(status): int(n) for status, n in fold.status.value_counts().items()},
        "selection_disagreements": 0, "run_manifest_sha256": digest(args.run / "run_manifest.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
