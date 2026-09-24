from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "gap_crossed"
RESULTS = OUTPUT / "gap_crossed_results.csv"
PAIRS = OUTPUT / "nearest_gap_cell_pairs.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    frame = pd.read_csv(RESULTS)
    if not PAIRS.exists():
        raise FileNotFoundError(f"nearest-gap pair table not found: {PAIRS}")
    pairs = pd.read_csv(PAIRS)
    if "comparison_type" not in pairs.columns:
        pairs.insert(0, "comparison_type", "nearest_gap_descriptive")
    if "one_to_one_matching" not in pairs.columns:
        pairs.insert(1, "one_to_one_matching", False)
    if "selection_rule" not in pairs.columns:
        pairs.insert(2, "selection_rule", "top-10 nearest cross-mode pairs per unordered mode pair; cells may repeat")
    metrics = {
        "action_accuracy": ("action_correct", "mean"),
        "absolute_regret_mean_cny": ("regret_cny_per_month", "mean"),
        "absolute_regret_p95_cny": ("regret_cny_per_month", lambda values: float(np.quantile(values, 0.95))),
        "regret_over_oracle_mean": ("regret_over_oracle_value", "mean"),
        "regret_over_capacity_mean": ("regret_over_capacity_fee", "mean"),
        "d_mv_mean": ("d_mv", "mean"),
    }
    cell = frame.groupby(["cell_id", "method_id"], as_index=False).agg(**metrics)
    design = frame.groupby("cell_id", as_index=False).agg(
        alpha=("alpha", "first"),
        utilization=("utilization", "first"),
        oracle_mode=("oracle_mode", "first"),
        oracle_margin_cny=("oracle_margin_cny", "first"),
    )
    cell = cell.merge(design, on="cell_id", how="left")

    rows = []
    for pair in pairs.itertuples(index=False):
        for method in sorted(frame["method_id"].unique()):
            left = cell[(cell["cell_id"] == pair.cell_left) & (cell["method_id"] == method)].iloc[0]
            right = cell[(cell["cell_id"] == pair.cell_right) & (cell["method_id"] == method)].iloc[0]
            rows.append(
                {
                    "mode_left": pair.mode_left,
                    "mode_right": pair.mode_right,
                    "pair_rank_within_mode_pair": pair.pair_rank_within_mode_pair,
                    "method_id": method,
                    "cell_left": pair.cell_left,
                    "cell_right": pair.cell_right,
                    "relative_margin_gap": pair.relative_margin_gap,
                    **{f"left_{name}": left[name] for name in metrics},
                    **{f"right_{name}": right[name] for name in metrics},
                }
            )
    matched = pd.DataFrame(rows)
    matched_path = OUTPUT / "nearest_gap_method_summary.csv"
    matched.insert(0, "comparison_type", "nearest_gap_descriptive")
    matched.insert(1, "one_to_one_matching", False)
    matched.to_csv(matched_path, index=False)

    regional = frame.groupby(["alpha", "oracle_mode", "method_id"], as_index=False).agg(
        cells=("cell_id", "nunique"),
        action_accuracy=("action_correct", "mean"),
        absolute_regret_mean_cny=("regret_cny_per_month", "mean"),
        regret_over_oracle_mean=("regret_over_oracle_value", "mean"),
        regret_over_capacity_mean=("regret_over_capacity_fee", "mean"),
        d_mv_mean=("d_mv", "mean"),
    )
    regional_path = OUTPUT / "crossed_region_method_summary.csv"
    regional.to_csv(regional_path, index=False)

    checks = {
        "all_nearest_gap_pairs_have_five_methods": bool((matched.groupby(["mode_left", "mode_right", "pair_rank_within_mode_pair"])["method_id"].nunique() == 5).all()),
        "absolute_and_normalized_regret_present": all(column in matched.columns for column in ["left_absolute_regret_mean_cny", "left_regret_over_oracle_mean", "right_absolute_regret_mean_cny", "right_regret_over_oracle_mean"]),
        "all_three_oracle_modes_present": frame["oracle_mode"].nunique() == 3,
        "all_four_alphas_present": frame["alpha"].nunique() == 4,
        "nearest_gap_is_descriptive_not_one_to_one": bool((pairs["one_to_one_matching"] == False).all()),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "result_sha256": sha256(RESULTS),
        "nearest_gap_rows": int(len(matched)),
        "regional_rows": int(len(regional)),
        "nearest_gap_sha256": sha256(matched_path),
        "regional_sha256": sha256(regional_path),
        "matching_label": "nearest-gap descriptive comparisons",
        "matching_is_one_to_one": False,
    }
    report = OUTPUT / "gap_crossed_summary_audit.json"
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(f"summary audit failed: {checks}")


if __name__ == "__main__":
    main()
