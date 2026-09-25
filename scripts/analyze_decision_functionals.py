"""Functional fidelity and history-paired differences in regret associations."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "extensions"
SEED = 2026092501
RESAMPLES = 5000
ERRORS = ["q99_relative_error", "mean_relative_error", "stoploss_relative_error"]


def rank_correlation(x, y):
    x = rankdata(x, axis=-1)
    y = rankdata(y, axis=-1)
    x -= x.mean(axis=-1, keepdims=True)
    y -= y.mean(axis=-1, keepdims=True)
    return (x * y).sum(axis=-1) / np.sqrt((x * x).sum(axis=-1) * (y * y).sum(axis=-1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(ROOT / "data/derived/primary_rows.csv.gz")
    if rows.duplicated(["replication_id", "method_id", "cell_id"]).any():
        raise ValueError("Duplicate primary decisions")
    if not (rows.groupby(["replication_id", "method_id"]).size() == 9).all():
        raise ValueError("Each fitted history must have nine tariff cells")
    rows[ERRORS] = rows[ERRORS].abs()
    units = rows.groupby(["replication_id", "method_id"], as_index=False)[
        ERRORS + ["regret_cny_per_month"]
    ].mean()
    units.to_csv(OUT / "functional_history_summary.csv", index=False)
    summaries, comparisons = [], []
    indices = np.random.default_rng(SEED).integers(0, 1000, (RESAMPLES, 1000))
    for method, group in units.groupby("method_id", sort=True):
        group = group.sort_values("replication_id")
        if len(group) != 1000:
            raise ValueError("Expected 1,000 histories per method")
        y = group.regret_cny_per_month.to_numpy()
        x = group[ERRORS].to_numpy().T
        point = rank_correlation(x, y)
        boot = []
        for index in np.array_split(indices, 20):
            boot.append(np.stack([rank_correlation(values[index], y[index]) for values in x], axis=1))
        boot = np.concatenate(boot)
        for j, error in enumerate(ERRORS):
            summaries.append(dict(method_id=method, diagnostic=error,
                                  median_percent=100 * np.median(x[j]),
                                  mean_percent=100 * np.mean(x[j]), histories=1000))
        for j in (1, 2):
            difference = boot[:, j] - boot[:, 0]
            low, high = np.quantile(difference, [0.025, 0.975])
            comparisons.append(dict(method_id=method, diagnostic=ERRORS[j],
                                    rho_functional=point[j], rho_q99=point[0],
                                    rho_difference=point[j] - point[0], ci95_low=low,
                                    ci95_high=high, histories=1000, bootstrap_replicates=RESAMPLES,
                                    bootstrap_seed=SEED))
    fidelity = pd.DataFrame(summaries)
    fidelity.to_csv(OUT / "functional_fidelity.csv", index=False)
    table_path = ROOT / "tables/method_fidelity_summary.csv"
    table = pd.read_csv(table_path)
    stoploss = fidelity[fidelity.diagnostic == "stoploss_relative_error"].set_index("method_id")
    table["median_abs_stoploss_error"] = table.method_id.map(stoploss.median_percent) / 100
    table["mean_abs_stoploss_error"] = table.method_id.map(stoploss.mean_percent) / 100
    table.to_csv(table_path, index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "functional_correlation_contrasts.csv", index=False)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
