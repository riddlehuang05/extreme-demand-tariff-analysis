"""Check K-BLOCK window continuity against the recorded timestamps."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT
INPUT = DATA / "inputs" / "korea_gate6_data_full" / "fixed15.parquet"
FITS = DATA / "outputs" / "external_rolling_cv" / "fit_diagnostics.csv"
OUTPUT = ROOT / "reports" / "K_BLOCK_CONTINUITY_AUDIT_20260924.json"
BLOCK_LENGTH = 96


def main() -> None:
    fixed = pd.read_parquet(INPUT)
    fits = pd.read_csv(FITS)
    fits = fits.loc[fits["method_id"] == "K-BLOCK"]
    units = []
    for row in fits.itertuples(index=False):
        source = fixed.loc[
            (fixed["factory"] == row.factory)
            & (fixed["variant"] == row.variant)
            & (fixed["measurement"] == row.measurement)
        ]
        series = source.set_index("window_timestamp")["load_kw"].sort_index()
        train_months = json.loads(row.train_months)
        series = series[series.index.to_period("M").astype(str).isin(train_months)]
        old_count = 0
        valid_count = 0
        gap_count = 0
        for _, group in series.groupby(series.index.to_period("M")):
            complete = group.rolling(BLOCK_LENGTH, min_periods=BLOCK_LENGTH).count().eq(BLOCK_LENGTH)
            adjacent = group.index.to_series().diff().eq(pd.Timedelta(minutes=15))
            consecutive = adjacent.rolling(BLOCK_LENGTH - 1).sum().eq(BLOCK_LENGTH - 1)
            old_count += int(complete.sum())
            valid_count += int((complete & consecutive).sum())
            gap_count += int((complete & ~consecutive).sum())
        recorded_count = json.loads(row.diagnostics)["moving_blocks_available"]
        units.append({
            "factory": row.factory,
            "origin_id": row.origin_id,
            "valid_blocks": valid_count,
            "nonconsecutive_blocks": gap_count,
            "previous_blocks": old_count,
            "recorded_blocks": recorded_count,
            "recorded_count_matches": old_count == recorded_count,
        })
    report = {
        "source": "external_rolling_cv fit_diagnostics.csv and fixed15.parquet",
        "script": "check_kblock_continuity.py",
        "unit_count": len(units),
        "valid_blocks_total": sum(unit["valid_blocks"] for unit in units),
        "nonconsecutive_blocks_total": sum(unit["nonconsecutive_blocks"] for unit in units),
        "recorded_counts_match": all(unit["recorded_count_matches"] for unit in units),
        "scores_unchanged_if_recomputed": all(unit["nonconsecutive_blocks"] == 0 for unit in units),
        "units": units,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "units"}, indent=2))


if __name__ == "__main__":
    main()
