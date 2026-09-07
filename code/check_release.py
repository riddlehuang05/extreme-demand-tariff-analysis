from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def close(x, y, tol=1e-6):
    return abs(float(x) - float(y)) <= tol

def auc_from_ranks(scores, positive):
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def main():
    q = pd.read_csv(DATA / "q99_errors.csv")
    r = pd.read_csv(DATA / "regret_differences.csv")
    d = pd.read_csv(DATA / "diagnostic_units.csv")
    rows = pd.read_csv(DATA / "mechanism_decision_rows_5400.csv")

    checks = []
    checks.append(("600 q0.99 error rows", len(q) == 600))
    checks.append(("1200 paired regret rows", len(r) == 1200))
    checks.append(("600 diagnostic units", len(d) == 600))
    checks.append(("5,400 decision rows", len(rows) == 5400))

    contract = r.loc[(r.oracle_region == "Contract-demand") & (r.baseline_method == "KDE"),
                     "tail_minus_baseline_regret_cny_per_month"].mean()
    capacity = r.loc[(r.oracle_region == "Capacity") & (r.baseline_method == "KDE"),
                     "tail_minus_baseline_regret_cny_per_month"].mean()
    checks.append(("contract TAIL-KDE mean difference", close(contract, -4373.847048, 1e-3)))
    checks.append(("capacity TAIL-KDE mean difference", close(capacity, 6300.653078, 1e-3)))

    rhos = {
        "q99": spearmanr(d.abs_q99_relative_error, d.regret_cny_per_month).statistic,
        "mean": spearmanr(d.abs_mean_relative_error, d.regret_cny_per_month).statistic,
        "stoploss": spearmanr(d.abs_stoploss_relative_error, d.regret_cny_per_month).statistic,
        "mode": spearmanr(d.mode_value_gap_ratio, d.regret_cny_per_month).statistic,
    }
    checks.append(("mean error has strongest regret association", max(rhos, key=rhos.get) == "mean"))

    positive = ~rows["tariff_mode_correct"].astype(bool)
    aucs = {
        "q99": auc_from_ranks(rows.abs_q99_relative_error, positive),
        "mean": auc_from_ranks(rows.abs_mean_relative_error, positive),
        "stoploss": auc_from_ranks(rows.abs_stoploss_relative_error, positive),
        "mode": auc_from_ranks(rows.mode_value_gap_ratio, positive),
    }
    # Article Table 3 mode-error AUC point values (Mann-Whitney, 5,400 decision rows).
    checks.append(("q0.99-error AUC 0.515", close(aucs["q99"], 0.515226, 1e-3)))
    checks.append(("mean-error AUC 0.553", close(aucs["mean"], 0.552972, 1e-3)))
    checks.append(("stop-loss-error AUC 0.520", close(aucs["stoploss"], 0.520051, 1e-3)))
    checks.append(("mode-value/gap AUC 0.689", close(aucs["mode"], 0.688978, 1e-3)))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if failed:
        raise SystemExit("Release check failed: " + ", ".join(failed))

if __name__ == "__main__":
    main()
