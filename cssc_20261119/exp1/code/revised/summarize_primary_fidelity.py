#!/usr/bin/env python3
"""Summarize primary functional errors at the independent-history level."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=10000)
    args = parser.parse_args()
    cols = ["replication_id", "method_id", "cell_id", "q99_relative_error",
            "mean_relative_error", "stoploss_relative_error"]
    data = pd.read_csv(args.decisions, usecols=cols)
    assert len(data) == args.histories * 5 * 9
    assert set(data.method_id) == set(METHODS)
    assert not data.duplicated(["replication_id", "method_id", "cell_id"]).any()
    per = data.groupby(["replication_id", "method_id"], sort=True)
    assert per.size().eq(9).all()
    q99 = per.q99_relative_error.agg(["min", "max"])
    mean = per.mean_relative_error.agg(["min", "max"])
    assert np.allclose(q99["min"], q99["max"], rtol=0, atol=1e-12)
    assert np.allclose(mean["min"], mean["max"], rtol=0, atol=1e-12)
    history = pd.DataFrame({
        "q99_abs": q99["min"].abs(), "mean_abs": mean["min"].abs(),
        "stoploss_abs": per.stoploss_relative_error.apply(lambda x: float(x.abs().mean())),
    }).reset_index()
    assert len(history) == args.histories * 5
    rows = []
    for method in METHODS:
        block = history[history.method_id.eq(method)]
        assert len(block) == args.histories and np.isfinite(block[["q99_abs", "mean_abs", "stoploss_abs"]]).all().all()
        row = {"method_id": method, "histories": args.histories}
        for metric in ("q99_abs", "mean_abs", "stoploss_abs"):
            row[f"{metric}_median_percent"] = 100 * float(block[metric].median())
            row[f"{metric}_mean_percent"] = 100 * float(block[metric].mean())
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
