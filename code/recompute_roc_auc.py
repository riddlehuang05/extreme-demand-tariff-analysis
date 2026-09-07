"""Recompute the tariff-mode-error ROC curves and AUC values reported in the article.

The article reports four endpoint-specific diagnostics evaluated at the
5,400 decision-row level of the nine-cell controlled mechanism experiment
(Table 3 and Figure 3a of the main text).  Each row is one
replication x method x tariff-cell evaluation.  The outcome is tariff-mode
error: the row is positive when the selected tariff mode differs from the
oracle mode.

This script reads data/mechanism_decision_rows_5400.csv and recomputes
  * the ROC curve coordinates for each diagnostic, and
  * the area under the curve (Mann-Whitney U statistic), with pointwise
    95% replication-cluster bootstrap intervals over the 200 replications.

The point AUC values are deterministic rank statistics and reproduce the
article's Table 3 column "mode-error AUC" exactly (q0.99 error 0.515, mean
error 0.553, stop-loss error 0.520, mode-value/gap 0.689).

The bootstrap resamples the 200 replication clusters with replacement,
retaining every row of each sampled replication as many times as that
replication was drawn (the multiplicity of each draw is preserved), and
computes percentile intervals from 10,000 resamples with seed 2026082301.
This matches the archived interval source data/diagnostic_summary.csv,
which is the canonical record of the intervals reported in the article.

ROC coordinates are written to a local results/ directory; they are
descriptive and no new experiment is run.

Requires: numpy, pandas, scipy (see requirements.txt).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DIAGNOSTICS = {
    "q0.99 error": "abs_q99_relative_error",
    "mean error": "abs_mean_relative_error",
    "stop-loss error": "abs_stoploss_relative_error",
    "mode-value / gap": "mode_value_gap_ratio",
}

BOOTSTRAP_REPLICATES = 10_000
SEED = 2026082301


def auc_from_ranks(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney U / (n_pos n_neg) AUC with average ranks."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_coordinates(scores: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """ROC coordinates (fpr, tpr) for a score that is larger under error.

    The curve is built by lowering the threshold over the distinct observed
    score values from high to low, so both the false-positive and the
    true-positive rate are non-decreasing along the returned path, which
    begins at (0, 0) and ends at (1, 1).
    """
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return np.empty((0, 2), dtype=float)
    thresholds = np.unique(scores)  # ascending
    # Walk thresholds from highest to lowest: the classified-positive set
    # {score >= t} grows as t falls, so TPR and FPR are non-decreasing.
    pts = [(0.0, 0.0)]
    for t in thresholds[::-1]:
        pred_pos = scores >= t
        tpr = int((pred_pos & positive).sum()) / n_pos
        fpr = int((pred_pos & ~positive).sum()) / n_neg
        pts.append((fpr, tpr))
    # The lowest threshold classifies everything positive, so the last point
    # is (1, 1); drop duplicate coordinates from tied thresholds if present.
    arr = np.asarray(pts, dtype=float)
    keep = np.ones(len(arr), dtype=bool)
    keep[1:] = ~(
        (arr[1:, 0] == arr[:-1, 0]) & (arr[1:, 1] == arr[:-1, 1])
    )
    return arr[keep]


def _cluster_bootstrap_interval(
    scores: np.ndarray, positive: np.ndarray, reps: np.ndarray
) -> tuple[float, float, float]:
    """Percentile 95% interval of the AUC by replication-cluster bootstrap.

    Each resample draws 200 replication ids with replacement and retains all
    rows of each sampled replication, duplicating a replication's rows as
    many times as that replication was drawn.
    """
    cluster_ids = np.unique(reps)
    rows_by_cluster = {r: np.flatnonzero(reps == r) for r in cluster_ids}
    rng = np.random.default_rng(SEED)
    boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for b in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([rows_by_cluster[r] for r in chosen])
        boot[b] = auc_from_ranks(scores[idx], positive[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(np.mean(boot)), float(lo), float(hi)


def main() -> None:
    rows = pd.read_csv(DATA / "mechanism_decision_rows_5400.csv")
    if len(rows) != 5_400:
        raise SystemExit(f"Expected 5,400 decision rows, found {len(rows)}")
    positive = ~rows["tariff_mode_correct"].astype(bool)
    reps = rows["replication_id"].to_numpy()

    records = []
    roc_frames = []
    for label, col in DIAGNOSTICS.items():
        scores = rows[col].to_numpy(dtype=float)
        if scores.size != len(rows):
            raise SystemExit(f"Diagnostic column {col} has unexpected length")
        auc_pt = auc_from_ranks(scores, positive)
        _, lo, hi = _cluster_bootstrap_interval(scores, positive, reps)

        records.append({
            "diagnostic": label,
            "mode_error_auc": auc_pt,
            "mode_error_auc_ci_lower": lo,
            "mode_error_auc_ci_upper": hi,
            "n_rows": int(len(rows)),
            "n_positive": int(positive.sum()),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "replication_id",
        })

        roc = roc_coordinates(scores, positive)
        if len(roc) > 1 and not (
            np.all(np.diff(roc[:, 0]) >= -1e-12)
            and np.all(np.diff(roc[:, 1]) >= -1e-12)
        ):
            raise SystemExit(f"ROC coordinates not monotone for {label}")
        roc_frames.append(pd.DataFrame({
            "diagnostic": label,
            "false_positive_rate": roc[:, 0],
            "true_positive_rate": roc[:, 1],
        }))

    summary = pd.DataFrame(records)
    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "diagnostic_auc_recomputed.csv", index=False)
    pd.concat(roc_frames, ignore_index=True).to_csv(
        RESULTS / "roc_curves_recomputed.csv", index=False
    )

    print("Recomputed tariff-mode-error AUC (5,400 decision rows):")
    print(summary[["diagnostic", "mode_error_auc", "mode_error_auc_ci_lower",
                   "mode_error_auc_ci_upper"]].to_string(index=False))
    print("\nArticle Table 3 point-AUC reference values: 0.515 (q0.99), 0.553 "
          "(mean), 0.520 (stop-loss), 0.689 (mode-value/gap).")
    print("\nBootstrap intervals resample the 200 replication clusters with\n"
          "replacement (10,000 resamples, seed 2026082301), retaining each\n"
          "sampled replication as many times as drawn; the archived intervals\n"
          "in data/diagnostic_summary.csv are the canonical record.")
    print(f"\nWrote results/diagnostic_auc_recomputed.csv and "
          f"results/roc_curves_recomputed.csv")


if __name__ == "__main__":
    main()
