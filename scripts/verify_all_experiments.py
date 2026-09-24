from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "all_experiments_verification.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    core = read(ROOT / "reports" / "verification.json")
    convergence = read(ROOT / "reports" / "replication_convergence_audit.json")
    gap = read(ROOT / "outputs" / "gap_crossed" / "summary.json")
    gap_summary = read(ROOT / "outputs" / "gap_crossed" / "gap_crossed_summary_audit.json")
    external = read(ROOT / "outputs" / "external_rolling_cv" / "summary.json")
    external_all = read(ROOT / "outputs" / "external_rolling_cv_all_observations" / "summary.json")
    external_sensitivity = read(ROOT / "outputs" / "external_variant_sensitivity" / "summary.json")
    diagnostics = read(ROOT / "outputs" / "diagnostic_adjustment" / "summary.json")
    regime = read(ROOT / "ablations" / "ASMBI_REGIME_SHIFT_V1" / "full" / "summary.json")

    gap_results = ROOT / "outputs" / "gap_crossed" / "gap_crossed_results.csv"
    external_scores = ROOT / "outputs" / "external_rolling_cv" / "prediction_scores.csv"
    external_fits = ROOT / "outputs" / "external_rolling_cv" / "fit_diagnostics.csv"
    external_candidates = ROOT / "outputs" / "external_rolling_cv" / "candidate_rolling_cv.csv"
    external_contrasts = ROOT / "outputs" / "external_rolling_cv" / "factory_cluster_contrasts.csv"
    external_all_scores = ROOT / "outputs" / "external_rolling_cv_all_observations" / "prediction_scores.csv"
    external_all_fits = ROOT / "outputs" / "external_rolling_cv_all_observations" / "fit_diagnostics.csv"
    external_all_candidates = ROOT / "outputs" / "external_rolling_cv_all_observations" / "candidate_rolling_cv.csv"
    external_all_contrasts = ROOT / "outputs" / "external_rolling_cv_all_observations" / "factory_cluster_contrasts.csv"
    regime_results = ROOT / "ablations" / "ASMBI_REGIME_SHIFT_V1" / "full" / "regime_shift_results.csv"

    gap_frame = pd.read_csv(gap_results, usecols=["replication_id", "method_id", "cell_id", "action_correct", "d_mv_below_one"])
    scores = pd.read_csv(external_scores)
    candidates = pd.read_csv(external_candidates)
    all_scores = pd.read_csv(external_all_scores)
    all_candidates = pd.read_csv(external_all_candidates)
    regime_frame = pd.read_csv(
        regime_results,
        usecols=["scenario_id", "replication_id", "method_id", "cell_id", "action_correct", "d_mv_below_one", "late_segment_sha256"],
    )
    checks = {
        "core_revision_pass": core["status"] == "PASS",
        "original_sources_unchanged": bool(core["checks"]["original_sources_unchanged"]),
        "replication_convergence_pass": convergence["status"] == "PASS",
        "gap_crossed_pass": gap["status"] == "PASS",
        "gap_crossed_summary_pass": gap_summary["status"] == "PASS",
        "external_rolling_pass": external["status"] == "PASS",
        "external_all_observations_pass": external_all["status"] == "PASS",
        "external_variant_sensitivity_pass": external_sensitivity["status"] == "PASS",
        "diagnostic_adjustment_pass": diagnostics["status"] == "PASS",
        "regime_shift_sensitivity_pass": regime["status"] == "PASS",
        "gap_rows_380000": len(gap_frame) == 380000,
        "gap_unique_keys": not gap_frame.duplicated(["replication_id", "method_id", "cell_id"]).any(),
        "gap_d_mv_certificate": bool(gap_frame.loc[gap_frame["d_mv_below_one"].astype(bool), "action_correct"].astype(bool).all()),
        "external_rows_224": len(scores) == 224,
        "external_seven_methods": scores["method_id"].nunique() == 7,
        "external_no_test_month_leakage": not bool((candidates["validation_month"].astype(str) == candidates["test_month"].astype(str)).any()),
        "external_all_rows_224": len(all_scores) == 224,
        "external_all_seven_methods": all_scores["method_id"].nunique() == 7,
        "external_all_no_test_month_leakage": not bool((all_candidates["validation_month"].astype(str) == all_candidates["test_month"].astype(str)).any()),
        "diagnostic_uses_1000_clusters": diagnostics["replication_clusters"] == 1000,
        "regime_rows_36000": len(regime_frame) == 36000,
        "regime_unique_keys": not regime_frame.duplicated(["scenario_id", "replication_id", "method_id", "cell_id"]).any(),
        "regime_paired_late_segment": bool((regime_frame.groupby("replication_id")["late_segment_sha256"].nunique() == 1).all()),
        "regime_d_mv_certificate": bool(regime_frame.loc[regime_frame["d_mv_below_one"].astype(bool), "action_correct"].astype(bool).all()),
        "gap_hash_matches_summary": sha256(gap_results) == gap["result_sha256"],
        "external_score_hash_matches_summary": sha256(external_scores) == external["scores_sha256"],
        "external_fit_hash_matches_summary": sha256(external_fits) == external["fits_sha256"],
        "external_candidate_hash_matches_summary": sha256(external_candidates) == external["candidates_sha256"],
        "external_contrast_hash_matches_summary": sha256(external_contrasts) == external["contrasts_sha256"],
        "external_all_score_hash_matches_summary": sha256(external_all_scores) == external_all["scores_sha256"],
        "external_all_fit_hash_matches_summary": sha256(external_all_fits) == external_all["fits_sha256"],
        "external_all_candidate_hash_matches_summary": sha256(external_all_candidates) == external_all["candidates_sha256"],
        "external_all_contrast_hash_matches_summary": sha256(external_all_contrasts) == external_all["contrasts_sha256"],
        "regime_hash_matches_summary": sha256(regime_results) == regime["result_sha256"],
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "core_full_rows": 45000,
        "frequency_rows": 28000,
        "gap_crossed_rows": int(len(gap_frame)),
        "external_score_rows": int(len(scores)),
        "external_candidate_fold_rows": int(len(candidates)),
        "external_all_observation_score_rows": int(len(all_scores)),
        "external_all_observation_candidate_fold_rows": int(len(all_candidates)),
        "regime_shift_rows": int(len(regime_frame)),
        "signatures": {
            "core": read(ROOT / "outputs" / "full" / "summary.json")["signature"],
            "gap_crossed": gap["signature"],
            "external_rolling": external["signature"],
            "external_rolling_all_observations": external_all["signature"],
            "diagnostic_adjustment": diagnostics["input_sha256"],
            "regime_shift": regime["signature"],
        },
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(f"all-experiment verification failed: {checks}")


if __name__ == "__main__":
    main()
