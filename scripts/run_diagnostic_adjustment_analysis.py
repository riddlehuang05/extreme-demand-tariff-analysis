from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "full" / "revision_results.csv"
OUTPUT = ROOT / "outputs" / "diagnostic_adjustment"
TABLES = ROOT / "tables"
TASK_ID = "DIAGNOSTIC_GAP_GEE_GROUPCV_V1"

DIAGNOSTICS = {
    "absolute_q99_relative_error": "q99_relative_error",
    "absolute_mean_relative_error": "mean_relative_error",
    "absolute_stoploss_relative_error": "stoploss_relative_error",
    "d_mv": "d_mv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def prepare_frame() -> pd.DataFrame:
    frame = pd.read_csv(INPUT)
    frame["mode_error"] = (~frame["action_correct"].astype(bool)).astype(int)
    frame["relative_gap"] = frame["oracle_margin_cny"] / frame["oracle_best_value_cny"]
    for name, column in DIAGNOSTICS.items():
        values = frame[column].abs() if name != "d_mv" else frame[column]
        frame[name] = values.astype(float)
    return frame


def gap_strata(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = (
        frame.groupby(["cell_id", "oracle_mode"], as_index=False)
        .agg(relative_gap=("relative_gap", "first"), mode_error_prevalence=("mode_error", "mean"))
        .sort_values("relative_gap")
        .reset_index(drop=True)
    )
    cells["gap_stratum"] = pd.qcut(
        cells["relative_gap"].rank(method="first"),
        q=3,
        labels=["small", "medium", "large"],
    ).astype(str)
    mapping = dict(zip(cells["cell_id"], cells["gap_stratum"]))
    work = frame.assign(gap_stratum=frame["cell_id"].map(mapping))
    by_cell_rows: list[dict[str, Any]] = []
    for cell_id, group in work.groupby("cell_id"):
        labels = group["mode_error"].to_numpy(dtype=int)
        if np.unique(labels).size < 2:
            continue
        for diagnostic in DIAGNOSTICS:
            by_cell_rows.append(
                {
                    "cell_id": cell_id,
                    "gap_stratum": str(group["gap_stratum"].iloc[0]),
                    "relative_gap": float(group["relative_gap"].iloc[0]),
                    "diagnostic": diagnostic,
                    "auc_within_cell": float(roc_auc_score(labels, group[diagnostic])),
                    "mode_error_prevalence": float(labels.mean()),
                }
            )
    by_cell = pd.DataFrame(by_cell_rows)
    rows: list[dict[str, Any]] = []
    for (stratum, diagnostic), group in by_cell.groupby(["gap_stratum", "diagnostic"]):
        source = work[work["gap_stratum"] == stratum]
        rows.append(
            {
                "gap_stratum": stratum,
                "diagnostic": diagnostic,
                "cells": int(group["cell_id"].nunique()),
                "relative_gap_min": float(group["relative_gap"].min()),
                "relative_gap_max": float(group["relative_gap"].max()),
                "macro_auc_equal_cell": float(group["auc_within_cell"].mean()),
                "min_cell_auc": float(group["auc_within_cell"].min()),
                "max_cell_auc": float(group["auc_within_cell"].max()),
                "pooled_within_stratum_auc": float(
                    roc_auc_score(source["mode_error"], source[diagnostic])
                ),
                "mode_error_prevalence": float(source["mode_error"].mean()),
            }
        )
    return cells, pd.DataFrame(rows).sort_values(["gap_stratum", "diagnostic"])


def gee_models(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for diagnostic in DIAGNOSTICS:
        model = smf.gee(
            f"mode_error ~ {diagnostic} + C(cell_id) + C(method_id)",
            groups="replication_id",
            data=frame,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        result = model.fit(maxiter=200)
        coefficient = float(result.params[diagnostic])
        standard_error = float(result.bse[diagnostic])
        low, high = (float(value) for value in result.conf_int().loc[diagnostic])
        rows.append(
            {
                "diagnostic": diagnostic,
                "scale": "raw_diagnostic_value",
                "coefficient": coefficient,
                "robust_standard_error": standard_error,
                "z_value": float(result.tvalues[diagnostic]),
                "p_value": float(result.pvalues[diagnostic]),
                "odds_ratio_full_range": float(np.exp(coefficient)),
                "odds_ratio_ci95_low": float(np.exp(low)),
                "odds_ratio_ci95_high": float(np.exp(high)),
                "working_correlation": float(result.cov_struct.dep_params),
                "clusters": int(frame["replication_id"].nunique()),
                "rows": int(len(frame)),
                "converged": bool(result.converged),
            }
        )
    return pd.DataFrame(rows)


def group_cross_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical = ["cell_id", "method_id"]
    groups = frame["replication_id"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    fold_rows: list[dict[str, Any]] = []
    for diagnostic in DIAGNOSTICS:
        numeric = [f"{diagnostic}_fold_percentile"]
        for fold, (train, test) in enumerate(splitter.split(frame, frame["mode_error"], groups), start=1):
            # Fit the empirical CDF on training rows only, then apply that
            # mapping to the held-out replication groups. This removes the
            # previous global-percentile preprocessing leakage.
            train_values = np.sort(frame.iloc[train][diagnostic].to_numpy(dtype=float))
            train_percentile = np.searchsorted(
                train_values,
                frame.iloc[train][diagnostic].to_numpy(dtype=float),
                side="right",
            ) / float(train_values.size)
            test_percentile = np.searchsorted(
                train_values,
                frame.iloc[test][diagnostic].to_numpy(dtype=float),
                side="right",
            ) / float(train_values.size)
            train_frame = frame.iloc[train].copy()
            test_frame = frame.iloc[test].copy()
            train_frame[numeric[0]] = np.clip(train_percentile, 0.0, 1.0)
            test_frame[numeric[0]] = np.clip(test_percentile, 0.0, 1.0)
            for model_name, columns in (("fixed_effects_only", categorical), ("plus_diagnostic", categorical + numeric)):
                transformer = ColumnTransformer(
                    [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
                    remainder="passthrough" if numeric[0] in columns else "drop",
                )
                model = Pipeline(
                    [
                        ("features", transformer),
                        (
                            "logistic",
                            LogisticRegression(
                                solver="lbfgs",
                                max_iter=2000,
                                random_state=20260916,
                            ),
                        ),
                    ]
                )
                selected = categorical + numeric if model_name == "plus_diagnostic" else categorical
                model.fit(train_frame[selected], train_frame["mode_error"])
                probability = model.predict_proba(test_frame[selected])[:, 1]
                fold_rows.append(
                    {
                        "diagnostic": diagnostic,
                        "fold": fold,
                        "model": model_name,
                        "preprocessing": "training_fold_empirical_percentile" if model_name == "plus_diagnostic" else "none",
                        "test_replications": int(test_frame["replication_id"].nunique()),
                        "test_rows": int(len(test)),
                        "roc_auc": float(roc_auc_score(test_frame["mode_error"], probability)),
                        "log_loss": float(log_loss(test_frame["mode_error"], probability)),
                    }
                )
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby(["diagnostic", "model"], as_index=False)
        .agg(mean_roc_auc=("roc_auc", "mean"), sd_roc_auc=("roc_auc", "std"), mean_log_loss=("log_loss", "mean"), sd_log_loss=("log_loss", "std"))
    )
    pivot_auc = summary.pivot(index="diagnostic", columns="model", values="mean_roc_auc")
    pivot_loss = summary.pivot(index="diagnostic", columns="model", values="mean_log_loss")
    increments = pd.DataFrame(
        {
            "diagnostic": pivot_auc.index,
            "auc_increment_over_fixed_effects": pivot_auc["plus_diagnostic"] - pivot_auc["fixed_effects_only"],
            "log_loss_reduction_over_fixed_effects": pivot_loss["fixed_effects_only"] - pivot_loss["plus_diagnostic"],
        }
    ).reset_index(drop=True)
    summary = summary.merge(increments, on="diagnostic", how="left")
    return folds, summary


def main() -> None:
    frame = prepare_frame()
    cells, strata = gap_strata(frame)
    gee = gee_models(frame)
    cv_folds, cv_summary = group_cross_validation(frame)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    paths = {
        "gap_mapping": OUTPUT / "gap_strata_mapping.csv",
        "gap_auc": OUTPUT / "diagnostic_auc_by_gap_stratum.csv",
        "gee": OUTPUT / "diagnostic_gee.csv",
        "cv_folds": OUTPUT / "diagnostic_group_cv_folds.csv",
        "cv_summary": OUTPUT / "diagnostic_group_cv_summary.csv",
    }
    atomic_csv(cells, paths["gap_mapping"])
    atomic_csv(strata, paths["gap_auc"])
    atomic_csv(gee, paths["gee"])
    atomic_csv(cv_folds, paths["cv_folds"])
    atomic_csv(cv_summary, paths["cv_summary"])
    atomic_csv(strata, TABLES / "diagnostic_auc_by_gap_stratum.csv")
    atomic_csv(gee, TABLES / "diagnostic_gee.csv")
    atomic_csv(cv_summary, TABLES / "diagnostic_group_cv_summary.csv")

    checks = {
        "four_diagnostics_present": set(strata["diagnostic"]) == set(DIAGNOSTICS),
        "three_gap_strata_present": set(strata["gap_stratum"]) == {"small", "medium", "large"},
        "all_nine_cells_mapped": len(cells) == 9,
        "gee_converged": bool(gee["converged"].all()),
        "gee_uses_1000_clusters": bool((gee["clusters"] == 1000).all()),
        "group_cv_five_folds": bool((cv_folds.groupby(["diagnostic", "model"])["fold"].nunique() == 5).all()),
        "group_cv_preprocessing_is_fold_safe": bool(
            set(cv_folds.loc[cv_folds["model"] == "plus_diagnostic", "preprocessing"])
            == {"training_fold_empirical_percentile"}
        ),
        "all_outputs_finite": bool(
            np.isfinite(
                pd.concat(
                    [
                        strata.select_dtypes(include=[np.number]).stack(),
                        gee.select_dtypes(include=[np.number]).stack(),
                        cv_summary.select_dtypes(include=[np.number]).stack(),
                    ]
                ).to_numpy(dtype=float)
            ).all()
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "task_id": TASK_ID,
        "input_rows": int(len(frame)),
        "replication_clusters": int(frame["replication_id"].nunique()),
        "diagnostics": list(DIAGNOSTICS),
        "gee_scale": "raw_diagnostic_value",
        "cv_preprocessing": "empirical percentile mapping fitted within each training fold and applied to its held-out groups",
        "checks": checks,
        "input_sha256": sha256(INPUT),
        "output_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(f"diagnostic adjustment analysis failed: {checks}")


if __name__ == "__main__":
    main()
