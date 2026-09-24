from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NATURAL = ROOT / "outputs" / "external_rolling_cv"
ALL_OBS = ROOT / "outputs" / "external_rolling_cv_all_observations"
OUTPUT = ROOT / "outputs" / "external_variant_sensitivity"
TABLES = ROOT / "tables"
TASK_ID = "EXTERNAL_ALL_OBSERVATIONS_PAIRED_SENSITIVITY_V1"
METRICS = [
    "crps",
    "qwcrps_090",
    "q95_pinball",
    "q99_pinball",
    "q95_exceedance_brier",
    "q99_exceedance_brier",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    natural_path = NATURAL / "prediction_scores.csv"
    all_path = ALL_OBS / "prediction_scores.csv"
    natural = pd.read_csv(natural_path)
    all_observations = pd.read_csv(all_path)
    keys = ["factory", "origin_id", "method_id"]
    merged = natural[keys + METRICS].merge(
        all_observations[keys + METRICS],
        on=keys,
        how="inner",
        suffixes=("_natural_tail", "_all_observations"),
        validate="one_to_one",
    )
    for metric in METRICS:
        merged[f"{metric}_delta_all_minus_natural"] = (
            merged[f"{metric}_all_observations"] - merged[f"{metric}_natural_tail"]
        )

    rng = np.random.default_rng(20260917)
    rows = []
    for method_id, group in merged.groupby("method_id"):
        for metric in METRICS:
            column = f"{metric}_delta_all_minus_natural"
            factory_delta = group.groupby("factory")[column].mean().to_numpy(dtype=float)
            indexes = rng.integers(0, factory_delta.size, size=(5000, factory_delta.size))
            bootstrap = factory_delta[indexes].mean(axis=1)
            rows.append(
                {
                    "method_id": method_id,
                    "metric": metric,
                    "contrast": "all_observations_minus_natural_tail",
                    "positive_means_higher_loss_under_all_observations": True,
                    "factory_mean_difference": float(factory_delta.mean()),
                    "factory_cluster_ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "factory_cluster_ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "factories": int(factory_delta.size),
                    "origins_per_factory": int(group["origin_id"].nunique()),
                    "bootstrap_replicates": 5000,
                }
            )
    paired = pd.DataFrame(rows).sort_values(["metric", "method_id"]).reset_index(drop=True)

    natural_fits = pd.read_csv(NATURAL / "fit_diagnostics.csv", usecols=keys + ["fit_status"])
    all_fits = pd.read_csv(ALL_OBS / "fit_diagnostics.csv", usecols=keys + ["fit_status"])
    transitions = natural_fits.merge(
        all_fits,
        on=keys,
        how="inner",
        suffixes=("_natural_tail", "_all_observations"),
        validate="one_to_one",
    )
    transitions = (
        transitions.groupby(
            ["method_id", "fit_status_natural_tail", "fit_status_all_observations"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "factory_origin_units"})
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    merged_path = OUTPUT / "paired_factory_origin_scores.csv"
    paired_path = OUTPUT / "paired_factory_cluster_sensitivity.csv"
    transitions_path = OUTPUT / "fit_status_transitions.csv"
    atomic_csv(merged, merged_path)
    atomic_csv(paired, paired_path)
    atomic_csv(transitions, transitions_path)
    atomic_csv(paired, TABLES / "external_all_observations_paired_sensitivity.csv")
    atomic_csv(transitions, TABLES / "external_all_observations_fit_status_transitions.csv")

    checks = {
        "all_224_rows_paired": len(merged) == 224,
        "seven_methods_present": merged["method_id"].nunique() == 7,
        "eight_factories_present": merged["factory"].nunique() == 8,
        "four_origins_present": merged["origin_id"].nunique() == 4,
        "all_differences_finite": bool(
            np.isfinite(
                merged[[f"{metric}_delta_all_minus_natural" for metric in METRICS]].to_numpy(dtype=float)
            ).all()
        ),
        "all_fit_status_rows_paired": int(transitions["factory_origin_units"].sum()) == 224,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "task_id": TASK_ID,
        "checks": checks,
        "natural_scores_sha256": sha256(natural_path),
        "all_observations_scores_sha256": sha256(all_path),
        "paired_scores_sha256": sha256(merged_path),
        "sensitivity_sha256": sha256(paired_path),
        "transitions_sha256": sha256(transitions_path),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(f"external variant sensitivity failed: {checks}")


if __name__ == "__main__":
    main()
