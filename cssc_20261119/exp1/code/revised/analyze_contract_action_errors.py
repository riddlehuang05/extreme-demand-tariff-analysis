"""Quantile and contract-action errors on the revised primary decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (ROOT / "CSSC_REVIEW_REVISION_20261112" / "primary_cv_crps_analysis_20261115"
          / "analysis" / "primary_decisions_enriched_10000.csv.gz")
OUT = ROOT / "CSSC_REVIEW_REVISION_20261112" / "contract_action_errors_20261119"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    names = ["replication_id", "method_id", "cell_id", "oracle_mode", "S_kva",
             "oracle_contract_D_kw", "estimated_contract_D_kw",
             "oracle_contract_quantile_p_star_kw", "estimated_contract_quantile_p_star_kw",
             "contract_quantile_probability_p_star", "regret_cny_per_month",
             "mode_selection_regret_cny", "within_contract_regret_cny"]
    frame = pd.read_csv(SOURCE, usecols=names)
    if len(frame) != 450000 or frame.replication_id.nunique() != 10000:
        raise ValueError("Complete revised primary decisions required")
    frame = frame[frame.oracle_mode.eq("contracted_maximum_demand")].copy()
    if len(frame) != 150000 or frame.cell_id.nunique() != 3:
        raise ValueError("Expected three 10,000-history contract cells for five methods")
    frame["oracle_contract_status"] = np.where(
        np.isclose(frame.oracle_contract_D_kw, 0.4 * frame.S_kva, atol=1e-5),
        "lower", np.where(np.isclose(frame.oracle_contract_D_kw, frame.S_kva, atol=1e-5),
                          "upper", "interior"))
    frame["q_p_abs_error_kw"] = (frame.estimated_contract_quantile_p_star_kw
                                  - frame.oracle_contract_quantile_p_star_kw).abs()
    frame["q_p_abs_relative_error"] = (frame.q_p_abs_error_kw
                                          / frame.oracle_contract_quantile_p_star_kw)
    frame["D_abs_error_kw"] = (frame.estimated_contract_D_kw
                                - frame.oracle_contract_D_kw).abs()
    frame["D_abs_over_S"] = frame.D_abs_error_kw / frame.S_kva
    if not np.isfinite(frame[["q_p_abs_relative_error", "D_abs_over_S"]].to_numpy()).all():
        raise ValueError("Nonfinite contract diagnostic")
    rows = []
    for (method, cell, status), group in frame.groupby(
            ["method_id", "cell_id", "oracle_contract_status"], sort=True):
        if len(group) != 10000:
            raise ValueError("Incomplete history-cell diagnostic")
        rows.append({"method_id": method, "cell_id": cell,
                     "oracle_contract_status": status, "histories": len(group),
                     "p_star": float(group.contract_quantile_probability_p_star.iloc[0]),
                     "mean_abs_q_p_error_kw": float(group.q_p_abs_error_kw.mean()),
                     "mean_abs_q_p_relative_percent": float(100 * group.q_p_abs_relative_error.mean()),
                     "mean_abs_D_error_kw": float(group.D_abs_error_kw.mean()),
                     "mean_abs_D_over_S_percent": float(100 * group.D_abs_over_S.mean()),
                     "spearman_q_p_error_vs_regret": float(spearmanr(group.q_p_abs_error_kw, group.regret_cny_per_month).statistic),
                     "spearman_D_error_vs_regret": float(spearmanr(group.D_abs_error_kw, group.regret_cny_per_month).statistic),
                     "mean_mode_selection_regret_cny": float(group.mode_selection_regret_cny.mean()),
                     "mean_within_contract_regret_cny": float(group.within_contract_regret_cny.mean())})
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "contract_action_error_by_method_cell.csv"
    result.to_csv(output, index=False, float_format="%.12g")
    audit = {"status": "PASS", "source_sha256": sha256(SOURCE),
             "rows": len(frame), "histories": 10000,
             "cell_statuses": result[["cell_id", "oracle_contract_status"]].drop_duplicates().to_dict("records"),
             "output_sha256": sha256(output)}
    (OUT / "run_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
