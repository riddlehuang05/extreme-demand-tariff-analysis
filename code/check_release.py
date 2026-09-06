from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def close(x, y, tol=1e-6):
    return abs(float(x) - float(y)) <= tol

def main():
    q = pd.read_csv(DATA / "q99_errors.csv")
    r = pd.read_csv(DATA / "regret_differences.csv")
    d = pd.read_csv(DATA / "diagnostic_units.csv")

    checks = []
    checks.append(("600 q0.99 error rows", len(q) == 600))
    checks.append(("1200 paired regret rows", len(r) == 1200))
    checks.append(("600 diagnostic units", len(d) == 600))

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
        "mode": spearmanr(d.mode_value_margin_ratio, d.regret_cny_per_month).statistic,
    }
    checks.append(("mean error has strongest regret association", max(rhos, key=rhos.get) == "mean"))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if failed:
        raise SystemExit("Release check failed: " + ", ".join(failed))

if __name__ == "__main__":
    main()
