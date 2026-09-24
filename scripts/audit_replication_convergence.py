from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "residual_experiments.yaml"
SOURCE = ROOT / "outputs" / "full" / "revision_results.csv"
DETAIL = ROOT / "reports" / "replication_convergence_detail.csv"
AUDIT = ROOT / "reports" / "replication_convergence_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mcse(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def add_metric(
    rows: list[dict],
    prefix: int,
    method_id: str,
    cell_id: str,
    metric: str,
    values: np.ndarray,
) -> None:
    estimate = float(np.mean(values))
    standard_error = mcse(values)
    rows.append(
        {
            "prefix_replications": prefix,
            "method_id": method_id,
            "cell_id": cell_id,
            "metric": metric,
            "estimate": estimate,
            "mcse": standard_error,
            "normal_ci95_low": estimate - 1.96 * standard_error,
            "normal_ci95_high": estimate + 1.96 * standard_error,
            "replications": int(values.size),
        }
    )


def main() -> None:
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["residual_experiments"]
    prefixes = [int(value) for value in settings["convergence_prefixes"]]
    frame = pd.read_csv(SOURCE)
    required = {
        "replication_id",
        "method_id",
        "cell_id",
        "action_correct",
        "regret_cny_per_month",
        "regret_over_oracle_value",
        "regret_over_capacity_fee",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing columns: {missing}")
    methods = sorted(frame["method_id"].unique())
    cells = sorted(frame["cell_id"].unique())
    max_replication = int(frame["replication_id"].max()) + 1
    if max_replication < max(prefixes):
        raise RuntimeError(f"only {max_replication} replications are available")

    rows: list[dict] = []
    for prefix in prefixes:
        subset = frame[frame["replication_id"] < prefix]
        for (method_id, cell_id), group in subset.groupby(["method_id", "cell_id"], sort=True):
            if len(group) != prefix:
                raise RuntimeError(f"incomplete prefix={prefix}, method={method_id}, cell={cell_id}")
            add_metric(rows, prefix, method_id, cell_id, "action_accuracy", group["action_correct"].astype(float).to_numpy())
            for metric in ("regret_cny_per_month", "regret_over_oracle_value", "regret_over_capacity_fee"):
                add_metric(rows, prefix, method_id, cell_id, metric, group[metric].to_numpy(dtype=float))

        paired = subset[subset["method_id"].isin(["TAIL", "EVENT-EMP"])].pivot(
            index=["replication_id", "cell_id"], columns="method_id"
        )
        for cell_id in cells:
            cell = paired.xs(cell_id, level="cell_id")
            for metric in ("action_correct", "regret_cny_per_month", "regret_over_oracle_value", "regret_over_capacity_fee"):
                values = cell[(metric, "TAIL")].astype(float).to_numpy() - cell[(metric, "EVENT-EMP")].astype(float).to_numpy()
                name = "paired_accuracy_TAIL_minus_EVENT_EMP" if metric == "action_correct" else f"paired_{metric}_TAIL_minus_EVENT_EMP"
                add_metric(rows, prefix, "TAIL-minus-EVENT-EMP", cell_id, name, values)

    detail = pd.DataFrame(rows).sort_values(["metric", "method_id", "cell_id", "prefix_replications"])
    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL, index=False)

    final_n = prefixes[-1]
    prior_n = prefixes[-2]
    final = detail[detail["prefix_replications"] == final_n].set_index(["method_id", "cell_id", "metric"])
    prior = detail[detail["prefix_replications"] == prior_n].set_index(["method_id", "cell_id", "metric"])
    comparison = final[["estimate", "mcse"]].join(prior[["estimate"]], rsuffix="_prior")
    comparison["absolute_drift"] = (comparison["estimate"] - comparison["estimate_prior"]).abs()
    comparison["tolerance"] = np.maximum(
        2.0 * comparison["mcse"].fillna(0.0),
        0.001 * np.maximum(comparison["estimate"].abs(), 1.0),
    )
    comparison["stable"] = comparison["absolute_drift"] <= comparison["tolerance"]
    unstable = comparison[~comparison["stable"]].reset_index()

    checks = {
        "source_has_1000_replications": max_replication == 1000,
        "all_prefixes_complete": len(detail) == len(prefixes) * len(cells) * (len(methods) * 4 + 4),
        "all_estimates_finite": bool(np.isfinite(detail[["estimate", "mcse"]].to_numpy(dtype=float)).all()),
        "final_800_to_1000_stability": unstable.empty,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "config_sha256": sha256(CONFIG),
        "prefixes": prefixes,
        "methods": methods,
        "cells": cells,
        "detail_rows": int(len(detail)),
        "checks": checks,
        "maximum_absolute_drift_800_to_1000": float(comparison["absolute_drift"].max()),
        "maximum_drift_over_tolerance": float((comparison["absolute_drift"] / comparison["tolerance"].replace(0.0, np.nan)).max()),
        "unstable_series_count": int(len(unstable)),
        "unstable_series": unstable.to_dict(orient="records"),
        "detail_sha256": sha256(DETAIL),
    }
    AUDIT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError("replication convergence audit failed; inspect the JSON report")


if __name__ == "__main__":
    main()
