"""Reproduce the numerical summaries reported in the CSSC article."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def check_close(a, b, label, atol=1e-7):
    if not np.isclose(float(a), float(b), rtol=1e-10, atol=atol):
        raise AssertionError(f"{label}: {a} != {b}")

def integrity():
    rows = list(csv.DictReader((ROOT / "SHA256SUMS.csv").open(newline="", encoding="utf-8")))
    if not rows:
        raise AssertionError("Empty checksum manifest")
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha(path) != row["sha256"]:
            raise AssertionError(f"Checksum mismatch: {row['path']}")
    return len(rows)

def wasserstein():
    laws = pd.read_parquet(R / "adaptive_w1/adaptive_w1_by_law.parquet")
    decisions = pd.read_parquet(R / "adaptive_w1/adaptive_w1_decision_checks.parquet")
    by_method = pd.read_csv(R / "adaptive_w1/adaptive_w1_by_method.csv")
    coverage = pd.read_csv(R / "adaptive_w1/adaptive_w1_coverage.csv").set_index("condition")
    assert len(laws) == 50000 and len(decisions) == 450000
    assert laws.groupby("method_id").size().eq(10000).all()
    assert decisions.groupby("method_id").size().eq(90000).all()
    for method, frame in laws.groupby("method_id"):
        row = by_method.loc[(by_method.method_id == method) & (by_method.bound == "combined")].iloc[0]
        for field, series in (
            ("bracket_midpoint_mean_kw", frame.combined_midpoint_kw),
            ("lower_mean_kw", frame.combined_lower_kw),
            ("upper_mean_kw", frame.combined_upper_kw),
        ):
            check_close(series.mean(), row[field], f"{method} {field}")
    labels = {
        "primitive": "primitive",
        "combined_direct_pair_contrast": "combined_direct_pair_contrast",
        "combined_modewise_per_competitor": "combined_modewise_per_competitor",
        "combined_uniform_L_full_action": "combined_uniform_L_full_action",
    }
    for name, column in labels.items():
        frame = coverage.loc[name]
        passed = decisions[column].astype(bool)
        assert int(passed.sum()) == int(frame.passing_rows), name
        assert not (passed & decisions.mode_error.astype(bool)).any(), name
    gev = laws.loc[laws.method_id.eq("GEV")]
    width = gev.combined_upper_kw - gev.combined_lower_kw
    assert width.median() < 300 and width.quantile(.95) < 300
    return {"laws": len(laws), "decisions": len(decisions),
            "gev_mean_upper_mw": round(gev.combined_upper_kw.mean()/1000, 3),
            "contrast_pass": int(decisions.combined_direct_pair_contrast.sum()),
            "modewise_pass": int(decisions.combined_modewise_per_competitor.sum()),
            "full_action_pass": int(decisions.combined_uniform_L_full_action.sum())}

def ablation():
    metrics = pd.read_parquet(R / "tail_ablation/history_cell_metrics.parquet",
                              columns=["dgp","replication_id","cell_id","method_id","oracle_region","regret_cny"])
    summary = pd.read_csv(R / "tail_ablation/paired_summary.csv")
    assert len(metrics) == 540000
    assert metrics.groupby(["dgp","method_id"]).size().eq(90000).all()
    report = {}
    for dgp in ("primary","nb_weibull"):
        g = metrics.loc[metrics.dgp.eq(dgp)]
        for region in ("actual","contract","capacity","all_nine_cells"):
            z = g if region == "all_nine_cells" else g.loc[g.oracle_region.eq(region)]
            h = z.groupby(["replication_id","method_id"]).regret_cny.mean().unstack()
            diff = h["TAIL"] - h["TAIL-EMP-EXCESS"]
            assert len(diff) == 10000
            row = summary.loc[(summary.dgp.eq(dgp)) & (summary.oracle_region.eq(region))
                              & (summary.alternative.eq("TAIL-EMP-EXCESS"))
                              & (summary.metric.eq("regret_cny"))].iloc[0]
            check_close(diff.mean(), row.mean_diff_full_minus_alternative, f"{dgp}/{region} mean")
            se = diff.std(ddof=1) / math.sqrt(len(diff))
            bound = t.ppf(.975, len(diff)-1) * se
            check_close(diff.mean()-bound, row.ci95_lower, f"{dgp}/{region} low")
            check_close(diff.mean()+bound, row.ci95_upper, f"{dgp}/{region} high")
            if region == "all_nine_cells":
                report[dgp] = round(diff.mean()/1000, 3)
    return {"rows": len(metrics), "tail_minus_empirical_excess_thousand_cny": report}

def other():
    common = pd.read_csv(R / "high_pstar/high_pstar_common_contract_cells.csv")
    assert len(common) == 6 and set(common.p_star.round(2)) == {.90,.95,.99}
    assert common.cells.eq(4).all() and common.paired_histories.eq(10000).all()
    contract = pd.read_csv(R / "contract_action/contract_action_error_by_method_cell.csv")
    assert len(contract) == 15 and contract.histories.eq(10000).all()
    focal = pd.read_csv(R / "primary/focal_maxt_intervals.csv")
    paired = pd.read_csv(R / "primary/paired_regret_contrasts_10000.csv")
    region_map = {"Actual":"actual_maximum_demand","Contract":"contracted_maximum_demand","Capacity":"capacity"}
    for name, region in region_map.items():
        f = focal.loc[focal["first"].eq("GEV") & focal["second"].eq("KDE") & focal.region.eq(name)].iloc[0]
        p = paired.loc[paired.scope.eq("oracle_region") & paired.oracle_mode.eq(region)
                       & paired.method_a.eq("GEV") & paired.method_b.eq("KDE")].iloc[0]
        check_close(f.mean_thousand_cny, p.paired_regret_difference_a_minus_b/1000, f"GEV-KDE {name}")
        assert f.simultaneous_95_low * f.simultaneous_95_high > 0
    assert (R / "external/Figure4_external_selected.csv").is_file()
    return {"high_pstar_pairs": len(common), "contract_diagnostics": len(contract), "focal_contrasts": len(focal)}

def solver_allowance():
    path = R / "solver_allowance/full_allowance_rows.parquet"
    summary = json.loads((R / "solver_allowance/summary.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(path, columns=[
        "solver_zeta_contract_only_cny", "solver_zeta_full_numerical_cny",
        "objective_evaluation_allowance_selected_cny",
        "objective_evaluation_allowance_max_cny",
        "saved_candidate_selection_gap_cny",
        "combined_direct_pair_contrast", "combined_modewise_per_competitor",
        "combined_uniform_L_full_action"])
    assert len(frame) == 450000 and summary["decision_rows"] == len(frame)
    expected = (frame.solver_zeta_contract_only_cny
                + frame.objective_evaluation_allowance_selected_cny
                + frame.objective_evaluation_allowance_max_cny
                + frame.saved_candidate_selection_gap_cny)
    assert np.allclose(frame.solver_zeta_full_numerical_cny, expected, rtol=0, atol=1e-9)
    assert int(frame.combined_direct_pair_contrast.sum()) == 59410
    assert int(frame.combined_modewise_per_competitor.sum()) == 27352
    assert int(frame.combined_uniform_L_full_action.sum()) == 31677
    return {"rows": len(frame), "full_allowance_max_cny": round(float(frame.solver_zeta_full_numerical_cny.max()), 6),
            "added_allowance_max_cny": round(float((frame.solver_zeta_full_numerical_cny - frame.solver_zeta_contract_only_cny).max()), 9)}

def main():
    report = {"status":"PASS","checksum_files":integrity(),"wasserstein":wasserstein(),
              "ablation":ablation(),"other":other(),"solver_allowance":solver_allowance()}
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

