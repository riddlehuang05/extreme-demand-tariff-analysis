"""Capacity-region leave-one-cell-out sensitivity using the supplied primary results.

This is a post hoc summary analysis only: it does not refit predictive
models, regenerate histories, or alter the formal primary experiment.
The resampling unit is the replication/history. For each omission, the
remaining two capacity cells are averaged within history before method
contrasts are bootstrapped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = (
    ARCHIVE_ROOT
    / "outputs"
    / "full"
    / "revision_results.csv"
)
REGIONAL_PATH = (
    ARCHIVE_ROOT
    / "tables"
    / "regional_paired_contrasts.csv"
)
OUTPUT_PATH = (
    ARCHIVE_ROOT
    / "tables"
    / "capacity_leave_one_cell_out.csv"
)

CAPACITY_CELLS = [
    {"cell_id": "BASE_CAPACITY_NEAR", "alpha": 1.05, "utilization": 0.630},
    {"cell_id": "STRESS_CAPACITY_NEAR", "alpha": 1.20, "utilization": 0.635},
    {"cell_id": "STRESS_CAPACITY_CLEAR", "alpha": 1.20, "utilization": 0.700},
]
CONTRASTS = [("TAIL", "KDE"), ("GEV", "KDE")]
EXPECTED_HISTORY_COUNT = 1_000
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 2026092401


def main() -> None:
    frame = pd.read_csv(RAW_PATH)
    required = {
        "replication_id",
        "cell_id",
        "oracle_mode",
        "method_id",
        "regret_cny_per_month",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Primary raw data missing columns: {sorted(missing)}")

    cell_ids = [cell["cell_id"] for cell in CAPACITY_CELLS]
    capacity = frame.loc[
        frame["oracle_mode"].eq("capacity") & frame["cell_id"].isin(cell_ids),
        ["replication_id", "cell_id", "method_id", "regret_cny_per_month"],
    ].copy()
    if capacity.duplicated(["replication_id", "cell_id", "method_id"]).any():
        raise ValueError("Duplicate replication × cell × method rows found")

    expected_rows = EXPECTED_HISTORY_COUNT * len(CAPACITY_CELLS) * 5
    if len(capacity) != expected_rows:
        raise ValueError(f"Expected {expected_rows} capacity rows; found {len(capacity)}")
    if set(capacity["method_id"].unique()) != {"TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"}:
        raise ValueError("Unexpected method set in the selected capacity cells")

    wide = capacity.pivot(
        index=["replication_id", "cell_id"],
        columns="method_id",
        values="regret_cny_per_month",
    ).sort_index()
    histories = sorted(capacity["replication_id"].unique())
    if len(histories) != EXPECTED_HISTORY_COUNT:
        raise ValueError(f"Expected 1,000 histories; found {len(histories)}")

    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = bootstrap_rng.integers(
        0, EXPECTED_HISTORY_COUNT, size=(BOOTSTRAP_REPLICATES, EXPECTED_HISTORY_COUNT)
    )
    rows: list[dict[str, object]] = []

    for method_a, method_b in CONTRASTS:
        cell_differences: dict[str, pd.Series] = {}
        for cell_id in cell_ids:
            block = wide.xs(cell_id, level="cell_id")
            cell_differences[cell_id] = (block[method_a] - block[method_b]).reindex(histories)

        full_history_difference = pd.concat(cell_differences, axis=1).mean(axis=1)
        formal = pd.read_csv(REGIONAL_PATH)
        formal_row = formal.loc[
            formal["metric"].eq("regret_cny_per_month")
            & formal["oracle_mode"].eq("capacity")
            & formal["contrast"].eq(f"{method_a}-minus-{method_b}")
        ]
        if len(formal_row) != 1:
            raise ValueError(f"Could not uniquely locate formal regional contrast {method_a}-{method_b}")
        if not np.isclose(
            full_history_difference.mean(),
            float(formal_row.iloc[0]["estimate"]),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"Full three-cell mean does not match formal summary for {method_a}-{method_b}")

        for omitted in CAPACITY_CELLS:
            omitted_id = omitted["cell_id"]
            retained_ids = [cell_id for cell_id in cell_ids if cell_id != omitted_id]
            history_difference = pd.concat(
                [cell_differences[cell_id] for cell_id in retained_ids], axis=1
            ).mean(axis=1)
            values = history_difference.to_numpy(dtype=float)
            bootstrap_means = values[bootstrap_indices].mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

            rows.append(
                {
                    "contrast": f"{method_a}-minus-{method_b}",
                    "omitted_cell_id": omitted_id,
                    "omitted_alpha": omitted["alpha"],
                    "omitted_utilization": omitted["utilization"],
                    "retained_cell_ids": ";".join(retained_ids),
                    "n_histories": EXPECTED_HISTORY_COUNT,
                    "n_retained_cells": len(retained_ids),
                    "mean_difference_cny_per_month": float(values.mean()),
                    "ci95_low_cny_per_month": float(ci_low),
                    "ci95_high_cny_per_month": float(ci_high),
                    "mean_difference_thousand_cny_per_month": float(values.mean() / 1000.0),
                    "ci95_low_thousand_cny_per_month": float(ci_low / 1000.0),
                    "ci95_high_thousand_cny_per_month": float(ci_high / 1000.0),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "resampling_unit": "replication_id (history)",
                    "full_three_cell_mean_cny_per_month": float(full_history_difference.mean()),
                }
            )

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, float_format="%.12g", encoding="utf-8")
    print(f"Validated full three-cell means against {REGIONAL_PATH.name}.")
    print(f"Wrote {len(result)} history-paired leave-one-cell-out contrasts to {OUTPUT_PATH}.")
    print(
        result[
            [
                "contrast",
                "omitted_cell_id",
                "mean_difference_thousand_cny_per_month",
                "ci95_low_thousand_cny_per_month",
                "ci95_high_thousand_cny_per_month",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
