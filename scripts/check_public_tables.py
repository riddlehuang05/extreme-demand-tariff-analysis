"""Check the staged aggregate tables without requiring private inputs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
EXPECTED = {
    "method_fidelity_summary.csv",
    "regional_paired_contrasts.csv",
    "capacity_leave_one_cell_out.csv",
    "cell_decision_summary.csv",
    "diagnostic_spearman_history_bootstrap.csv",
}


def main() -> None:
    found = {path.name for path in TABLES.glob("*.csv")}
    missing = EXPECTED - found
    if missing:
        raise SystemExit(f"missing staged tables: {sorted(missing)}")
    for name in sorted(EXPECTED):
        frame = pd.read_csv(TABLES / name)
        if frame.empty:
            raise SystemExit(f"empty table: {name}")
        print(f"{name}: {len(frame)} rows, {len(frame.columns)} columns")
    print("public aggregate-table check: PASS")


if __name__ == "__main__":
    main()
