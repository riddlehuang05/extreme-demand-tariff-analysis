from pathlib import Path
import json
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def main():
    RESULTS.mkdir(exist_ok=True)

    q = pd.read_csv(DATA / "q99_errors.csv")
    q_summary = (
        q.groupby("method")["absolute_relative_q99_error_pct"]
        .agg(mean="mean", median="median")
        .reset_index()
    )

    regret = pd.read_csv(DATA / "regret_differences.csv")
    regret_summary = (
        regret.groupby(["oracle_region", "baseline_method"])["tail_minus_baseline_regret_cny_per_month"]
        .mean()
        .reset_index()
    )

    diag = pd.read_csv(DATA / "diagnostic_units.csv")
    diag_cols = {
        "abs_q99_relative_error": "q0.99 error",
        "abs_mean_relative_error": "mean error",
        "abs_stoploss_relative_error": "stop-loss error",
        "mode_value_gap_ratio": "mode-value / gap",
    }
    rows = []
    for col, label in diag_cols.items():
        rho = spearmanr(diag[col], diag["regret_cny_per_month"], nan_policy="omit").statistic
        rows.append({"diagnostic": label, "spearman_with_regret": float(rho)})
    diag_summary = pd.DataFrame(rows)

    q_summary.to_csv(RESULTS / "q99_error_summary.csv", index=False)
    regret_summary.to_csv(RESULTS / "paired_regret_summary.csv", index=False)
    diag_summary.to_csv(RESULTS / "diagnostic_regret_summary.csv", index=False)

    checks = {
        "q99_rows": int(len(q)),
        "paired_regret_rows": int(len(regret)),
        "diagnostic_unit_rows": int(len(diag)),
        "tail_minus_kde_contract_mean": float(
            regret.loc[(regret.oracle_region == "Contract-demand") & (regret.baseline_method == "KDE"),
                       "tail_minus_baseline_regret_cny_per_month"].mean()
        ),
        "tail_minus_kde_capacity_mean": float(
            regret.loc[(regret.oracle_region == "Capacity") & (regret.baseline_method == "KDE"),
                       "tail_minus_baseline_regret_cny_per_month"].mean()
        ),
    }
    (RESULTS / "release_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    print("Public analysis completed.")
    print(q_summary.to_string(index=False))
    print("\nMean paired regret differences (TAIL - baseline, CNY/month):")
    print(regret_summary.to_string(index=False))
    print("\nSpearman association with full-action regret:")
    print(diag_summary.to_string(index=False))


if __name__ == "__main__":
    main()
