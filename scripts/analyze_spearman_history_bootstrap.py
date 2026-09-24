"""History-paired intervals for secondary within-method correlations."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT
INPUT = DATA / "outputs" / "full" / "revision_results.csv"
OUTPUT = DATA / "tables" / "diagnostic_spearman_history_bootstrap.csv"
SEED = 2026092402
REPLICATES = 5000
DIAGNOSTICS = (
    "abs_q99_relative_error",
    "abs_mean_relative_error",
    "abs_stoploss_relative_error",
    "d_mv",
)


def main() -> None:
    frame = pd.read_csv(
        INPUT,
        usecols=[
            "replication_id", "method_id", "cell_id", "regret_cny_per_month",
            "q99_relative_error", "mean_relative_error", "stoploss_relative_error", "d_mv",
        ],
    )
    for source, target in (
        ("q99_relative_error", "abs_q99_relative_error"),
        ("mean_relative_error", "abs_mean_relative_error"),
        ("stoploss_relative_error", "abs_stoploss_relative_error"),
    ):
        frame[target] = frame[source].abs()
    units = (
        frame.groupby(["replication_id", "method_id"], as_index=False)
        .agg(regret=("regret_cny_per_month", "mean"), **{
            name: (name, "mean") for name in DIAGNOSTICS
        })
    )
    histories = np.sort(units["replication_id"].unique())
    methods = sorted(units["method_id"].unique())
    if len(histories) != 1000 or any(len(units.loc[units.method_id == method]) != 1000 for method in methods):
        raise ValueError("expected 1,000 complete histories per method")
    rng = np.random.default_rng(SEED)
    indexes = rng.integers(0, len(histories), size=(REPLICATES, len(histories)))
    rows = []
    for method in methods:
        group = units.loc[units.method_id == method].set_index("replication_id").loc[histories]
        y = group["regret"].to_numpy(dtype=float)
        for diagnostic in DIAGNOSTICS:
            x = group[diagnostic].to_numpy(dtype=float)
            estimate = float(stats.spearmanr(x, y).statistic)
            draws = np.array([
                stats.spearmanr(x[index], y[index]).statistic for index in indexes
            ], dtype=float)
            if not np.isfinite(draws).all():
                raise ValueError(f"nonfinite bootstrap association: {method}, {diagnostic}")
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append({
                "method_id": method,
                "diagnostic": diagnostic,
                "spearman_rho": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "histories": len(histories),
                "bootstrap_replicates": REPLICATES,
                "bootstrap_seed": SEED,
                "resampling_unit": "replication_id (history)",
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False, float_format="%.12g")
    print(f"Wrote {len(result)} method-diagnostic intervals to {OUTPUT}")


if __name__ == "__main__":
    main()
