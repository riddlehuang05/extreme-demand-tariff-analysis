#!/usr/bin/env python3
"""Audit saved primary decision chunks against four-fold selector choices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


METHODS = {"TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-manifest", type=Path, required=True)
    parser.add_argument("--selector-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    work = args.decision_manifest.parent
    manifest = json.loads(args.decision_manifest.read_text(encoding="utf-8"))
    selector = json.loads((args.selector_run / "full_audit.json").read_text(encoding="utf-8"))
    assert selector["status"] == "PASS" and selector["history_count"] == 10000
    choices = pd.concat([
        pd.read_csv(path)
        for path in sorted(args.selector_run.glob("cv_*_choices.csv"))
        if (lambda pieces: len(pieces) == 4 and int(pieces[2]) - int(pieces[1]) == 99)(path.stem.split("_"))
    ], ignore_index=True).set_index("replication_id")
    assert len(choices) == 10000 and choices.index.is_unique
    assert choices.status.eq("PASS").all()
    seen: set[int] = set()
    fit_status: dict[str, int] = {}
    rows_total = fits_total = 0
    for chunk in sorted(manifest["chunks"], key=lambda x: x["start"]):
        start, end = int(chunk["start"]), int(chunk["end"])
        assert chunk["status"] == "PASS" and chunk["signature"] == manifest["signature"]
        assert end - start == 25 and not (seen & set(range(start, end)))
        csv_path, fit_path = work / chunk["csv"], work / chunk["fits"]
        assert digest(csv_path) == chunk["csv_sha256"]
        assert digest(fit_path) == chunk["fits_sha256"]
        data = pd.read_csv(csv_path)
        assert len(data) == chunk["decision_rows"] == 45 * 25
        assert set(data.replication_id.unique()) == set(range(start, end))
        assert set(data.method_id.unique()) == METHODS
        assert data.cell_id.nunique() == 9 and data.support_policy.eq("no_upper").all()
        assert not data.duplicated(["replication_id", "method_id", "cell_id"]).any()
        assert data.groupby(["replication_id", "method_id"]).size().eq(9).all()
        fit_count = tail_count = 0
        with fit_path.open(encoding="utf-8") as stream:
            for line in stream:
                fit = json.loads(line)
                rep = int(fit["replication_id"])
                assert start <= rep < end and fit["method_id"] in METHODS
                assert fit["support_policy"] == "no_upper"
                fit_count += 1
                if fit["method_id"] == "TAIL":
                    tail_count += 1
                    diag = fit["fit_diagnostics"]
                    candidate = fit["candidate_diagnostics"]
                    assert diag["selection_method"] == "four_fold_month_level_cv_crps"
                    assert diag["shape_bootstrap_replicates"] == 500
                    assert (diag["selection_fit_months"], diag["selection_validation_months"],
                            diag["selection_folds"], diag["selection_total_held_out_months"]) == (18, 6, 4, 24)
                    assert len(candidate["fold_scores"]) == 16
                    assert abs(float(candidate["selected_quantile"])
                               - float(choices.loc[rep, "selected_threshold_quantile"])) < 1e-12
                    status = str(diag["fit_status"])
                    fit_status[status] = fit_status.get(status, 0) + 1
        assert fit_count == chunk["fit_records"] == 5 * 25 and tail_count == 25
        seen.update(range(start, end))
        rows_total += len(data)
        fits_total += fit_count
    if args.require_complete:
        assert manifest["status"] == "PASS" and len(manifest["chunks"]) == 400
        assert seen == set(range(10000)) and rows_total == 450000 and fits_total == 50000
        merged = Path(manifest["merged_decisions"])
        assert merged.is_file() and digest(merged) == manifest["merged_decisions_sha256"]
    output = {"status": "PASS" if args.require_complete else "PARTIAL_PASS",
              "decision_manifest_status": manifest["status"], "verified_chunks": len(manifest["chunks"]),
              "verified_histories": len(seen), "verified_decision_rows": rows_total,
              "verified_fit_records": fits_total, "tail_fit_status_counts": fit_status,
              "decision_manifest_sha256": digest(args.decision_manifest)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
