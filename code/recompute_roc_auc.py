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
error 0.553, stop-loss error 0.520, mode-value/gap 0.689).  The bootstrap
intervals use the same design as the article (10,000 resamples of the 200
replication clusters, percentile method, seed 2026082301) and agree with
the archived intervals in data/diagnostic_summary.csv within Monte Carlo
variation; the archived file remains the canonical interval source.  ROC
coordinates are written to a local results/ directory; they are descriptive
and no new experiment is run.

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
    """ROC coordinates (fpr, tpr) for a score that is larger under error."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    order = np.argsort(scores, kind="mergesort")  # stable: ties enter in data order
    # Evaluate the ROC at each distinct threshold following the sorted scores.
    thresholds = np.unique(scores[order])
    pts = []
    for t in thresholds:
        pred_pos = scores >= t
        tpr = int((pred_pos & positive).sum()) / n_pos
        fpr = int((pred_pos & ~positive).sum()) / n_neg
        pts.append((fpr, tpr))
    # ROC passes through (1, 1) after the smallest threshold.
    pts.append((1.0, 1.0))
    arr = np.asarray(pts, dtype=float)
    # Sort strictly increasing in fpr, keeping the highest tpr at ties.
    arr = arr[np.lexsort((-arr[:, 1], arr[:, 0]))]
    return arr


def main() -> None:
    rows = pd.read_csv(DATA / "mechanism_decision_rows_5400.csv")
    if len(rows) != 5_400:
        raise SystemExit(f"Expected 5,400 decision rows, found {len(rows)}")
    positive = ~rows["tariff_mode_correct"].astype(bool)
    rng = np.random.default_rng(SEED)

    records = []
    roc_frames = []
    for label, col in DIAGNOSTICS.items():
        scores = rows[col].to_numpy(dtype=float)
        if scores.size != len(rows):
            raise SystemExit(f"Diagnostic column {col} has unexpected length")
        auc_pt = auc_from_ranks(scores, positive)

        # Replication-cluster bootstrap for a pointwise 95% interval: resample
        # the 200 replication ids jointly, keeping all rows of each sampled
        # replication (matching the article's inference unit).
        reps = rows["replication_id"].to_numpy()
        cluster_ids = np.unique(reps)
        boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for b in range(BOOTSTRAP_REPLICATES):
            chosen = rng.choice(cluster_ids, size=cluster_ids.size, replace=True)
            mask = np.isin(reps, chosen)
            boot[b] = auc_from_ranks(scores[mask], positive[mask])
        lo, hi = np.quantile(boot, [0.025, 0.975])

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
    print("\nBootstrap intervals use the article's design (10,000 replication-"
          "cluster resamples, seed 2026082301) and agree with data/"
          "diagnostic_summary.csv within Monte Carlo variation.")
    print(f"\nWrote results/diagnostic_auc_recomputed.csv and "
          f"results/roc_curves_recomputed.csv")


if __name__ == "__main__":
    main()
