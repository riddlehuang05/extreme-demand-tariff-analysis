"""History-paired high-p* comparisons on the four common contract cells."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "CSSC_REVIEW_REVISION_20261112" / "high_pstar_cv_crps_20261115"
METHODS = ("TAIL", "EVENT-EMP", "GEV", "KDE", "EMP")
P_STARS = (0.90, 0.95, 0.99)
PAIRS = (("GEV", "KDE"), ("TAIL", "EVENT-EMP"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = RUN / "high_pstar_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = sorted(manifest["chunk_records"], key=lambda item: item["start"])
    if manifest.get("status") != "PASS" or len(records) != 400:
        raise ValueError("Complete high-p* run required")
    oracle_path = RUN / "high_pstar_oracle_design_cells.csv"
    oracle = pd.read_csv(oracle_path)
    utilizations = sorted(oracle.utilization.unique())
    masks = []
    for p in P_STARS:
        rows = oracle[np.isclose(oracle.p_star, p) & oracle.oracle_mode.eq("contracted_maximum_demand")]
        masks.append(set(rows.utilization))
    common = sorted(set.intersection(*masks))
    if common != [0.43, 0.45, 0.5, 0.625]:
        raise ValueError(f"Unexpected common contract cells: {common}")
    indices = [utilizations.index(u) for u in common]
    chunks = []
    for expected, record in enumerate(records):
        start, end = 25 * expected, 25 * (expected + 1)
        if record["start"] != start or record["end"] != end or record["status"] != "PASS":
            raise ValueError("Checkpoint coverage mismatch")
        path = RUN / "checkpoints" / f"chunk_{start:05d}_{end:05d}_regrets.npz"
        if sha256(path) != record["regrets_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        with np.load(path) as archive:
            ids = archive["replication_id"]
            cube = archive["regret_cny"]
        if cube.shape != (25, 5, 3, 10) or not np.array_equal(ids, np.arange(start, end)):
            raise ValueError("Checkpoint shape or history IDs mismatch")
        chunks.append(cube)
    regrets = np.concatenate(chunks)
    rows = []
    for pi, p in enumerate(P_STARS):
        oracle_p = oracle[np.isclose(oracle.p_star, p)].set_index("utilization")
        denominators = oracle_p.loc[common, "oracle_best_value_cny"].to_numpy(float)
        if not np.isfinite(denominators).all() or (denominators <= 0).any():
            raise ValueError("Invalid oracle cost denominator")
        for first, second in PAIRS:
            diff_cell = regrets[:, METHODS.index(first), pi, :][:, indices] - regrets[:, METHODS.index(second), pi, :][:, indices]
            difference = diff_cell.mean(axis=1)
            norm_difference = 100 * (diff_cell / denominators[None, :]).mean(axis=1)
            row = {"p_star": p, "method_a": first, "method_b": second,
                   "oracle_region": "common_contract_cells", "cells": len(common),
                   "utilizations": ",".join(str(u) for u in common), "paired_histories": len(difference)}
            for key, values in (("difference_cny", difference),
                                ("normalized_difference_percentage_points", norm_difference)):
                mean = float(values.mean())
                half = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
                row[key + "_mean"] = mean
                row[key + "_95ci_low"] = mean - half
                row[key + "_95ci_high"] = mean + half
            rows.append(row)
    output = RUN / "high_pstar_common_contract_cells.csv"
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.12g")
    audit = {"status": "PASS", "paired_histories": 10000, "common_utilizations": common,
             "source_manifest_sha256": sha256(manifest_path),
             "source_oracle_sha256": sha256(oracle_path), "output_sha256": sha256(output)}
    (RUN / "high_pstar_common_contract_cells_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
