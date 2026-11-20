"""Recompute Wasserstein brackets and tariff checks from archived primary fits."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import compute_primitive_bounds as original
import numpy as np
import pandas as pd
from scipy import special, stats


ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "exp1" / "code"
REV = ROOT / "CSSC_REVIEW_REVISION_20261112"
FIT_ROOT = REV / "primary_cv_crps_decisions_20261115/support_policy_audit/b92088b6a7e3c5ae"
BOUND_ROOT = REV / "primary_cv_crps_primitive_bounds_20261115"
ORACLE_CSV = REV / "source_data/data/source/primary/oracle_mode_values_9_cells.csv"
MODES = ("capacity", "actual_maximum_demand", "contracted_maximum_demand")
LAW_KEYS = ["replication_id", "method_id"]


def oracle_survival(x: np.ndarray) -> np.ndarray:
    dgp = original._ORACLE.dgp
    bulk = x < dgp.threshold_kw
    out = np.empty_like(x)
    out[bulk] = 1.0 - original.true_cdf_vec(x[bulk])
    tail = stats.genpareto.sf(x[~bulk] - dgp.threshold_kw, dgp.xi, scale=dgp.sigma_kw)
    out[~bulk] = -np.expm1(-dgp.tail_rate * tail)
    return out


def fitted_survival_endpoints(dist, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return P(X > x) and P(X >= x), preserving support atoms."""
    engine = original._ENGINE
    if isinstance(dist, engine.DiscreteDistribution):
        tail = np.r_[np.cumsum(dist.probabilities[::-1])[::-1], 0.0]
        return (tail[np.searchsorted(dist.support, x, side="right")],
                tail[np.searchsorted(dist.support, x, side="left")])
    if isinstance(dist, original._AUDIT.NonnegativeGEV):
        survival = stats.genextreme.sf(x, dist.shape_c, loc=dist.loc, scale=dist.scale) / dist.a
        return survival, survival
    if isinstance(dist, engine.NonnegativeGaussianKDE):
        survival = special.ndtr((dist.sample[None, :] - x[:, None]) / dist.bandwidth)
        survival = survival.sum(axis=1) / dist._normalizer
        return survival, survival
    if isinstance(dist, engine.TailProcessDistribution):
        baseline = np.sort(dist.baseline)
        right = np.searchsorted(baseline, x, side="right") / baseline.size
        left = np.searchsorted(baseline, x, side="left") / baseline.size
        tail = stats.genpareto.sf(np.maximum(0.0, x - dist.threshold), dist.xi, scale=dist.sigma)
        if dist.count_model == "poisson":
            log_pgf = -dist.count_mean * tail
        else:
            odds = (1.0 - dist.nb_probability) / dist.nb_probability
            log_pgf = -dist.nb_size * np.log1p(odds * tail)
        event_survival = -np.expm1(log_pgf)
        return 1.0 - right + right * event_survival, 1.0 - left + left * event_survival
    raise TypeError(type(dist))


def initial_partition(dist, tail_goal: float) -> tuple[np.ndarray, float]:
    engine = original._ENGINE
    dgp = original._ORACLE.dgp
    knots = [dgp.baseline_lower_kw, dgp.threshold_kw]
    if isinstance(dist, engine.DiscreteDistribution):
        knots.extend(dist.support)
    elif isinstance(dist, engine.TailProcessDistribution):
        knots.extend(dist.baseline)
        knots.append(dist.threshold)
        if dist.xi < 0.0:
            knots.append(dist.threshold - dist.sigma / dist.xi)
    elif isinstance(dist, original._AUDIT.NonnegativeGEV):
        knots.extend(x for x in (dist.lower, dist.upper, dist.loc) if math.isfinite(x))
    tail_start = dgp.threshold_kw
    if isinstance(dist, engine.TailProcessDistribution):
        tail_start = max(tail_start, dist.threshold, float(np.max(dist.baseline)))
    if isinstance(dist, engine.DiscreteDistribution):
        tail_start = max(tail_start, float(np.max(dist.support)))

    cutoff = tail_start
    for _ in range(100):
        tail = (original.estimated_tail_stoploss_upper(engine, dist, cutoff)
                + original.true_tail_stoploss_upper(cutoff))
        if math.isfinite(tail) and tail <= tail_goal:
            break
        cutoff *= 2.0
    else:
        raise RuntimeError(f"Could not bound the remaining tail for {type(dist).__name__}")
    scale = dgp.baseline_lower_kw
    grid = scale * np.expm1(np.linspace(0.0, np.log1p(cutoff / scale), 257))
    grid[-1] = cutoff
    return np.unique(np.r_[grid, [x for x in knots if 0.0 < x < cutoff]]), tail


def cell_bounds(a, b, right_a, left_b, truth_a, truth_b):
    # Inside (a,b), each survival lies between its right and left endpoint limits.
    # The endpoint cross-differences therefore bound the entire signed difference.
    dx = b - a
    delta_min = left_b - truth_a
    delta_max = right_a - truth_b
    lower = dx * np.maximum(0.0, np.maximum(delta_min, -delta_max))
    upper = dx * np.maximum(np.abs(delta_min), np.abs(delta_max))
    # Local numerical tolerance scales with survival, including in long tail cells.
    allowance = dx * (right_a + left_b + truth_a + truth_b) * (128 * np.finfo(float).eps)
    return lower, upper, allowance


def adaptive_bounds(dist, max_width: float, max_refinements: int) -> dict:
    grid, tail = initial_partition(dist, min(10.0, max_width / 4.0))
    right, left = fitted_survival_endpoints(dist, grid)
    truth = oracle_survival(grid)
    a, b = grid[:-1], grid[1:]
    cell_limit = len(a) + max_refinements
    right_a, left_b = right[:-1], left[1:]
    truth_a, truth_b = truth[:-1], truth[1:]
    while True:
        lower, upper, allowance = cell_bounds(a, b, right_a, left_b, truth_a, truth_b)
        error = upper - lower
        split_count = min(256, cell_limit - len(a), int(np.count_nonzero(error > 0.0)))
        if split_count <= 0 or float(error.sum()) + tail <= max_width:
            break
        ids = np.argpartition(error, -split_count)[-split_count:]
        mids = a[ids] + 0.5 * (b[ids] - a[ids])
        if np.any((mids <= a[ids]) | (mids >= b[ids])):
            raise ArithmeticError("Adaptive partition reached floating-point resolution")
        right_mid, left_mid = fitted_survival_endpoints(dist, mids)
        truth_mid = oracle_survival(mids)
        a = np.r_[a, mids]
        b = np.r_[b, b[ids]]
        b[ids] = mids
        right_a = np.r_[right_a, right_mid]
        left_b = np.r_[left_b, left_b[ids]]
        left_b[ids] = left_mid
        truth_a = np.r_[truth_a, truth_mid]
        truth_b = np.r_[truth_b, truth_b[ids]]
        truth_b[ids] = truth_mid
    allowance_total = float(allowance.sum()) + 128 * np.finfo(float).eps * tail
    lower_total = max(0.0, float(lower.sum()) - allowance_total)
    upper_total = float(upper.sum()) + tail + allowance_total
    if not 0.0 <= lower_total <= upper_total < math.inf:
        raise ArithmeticError("Invalid Wasserstein bracket")
    return {"adaptive_lower_kw": lower_total, "adaptive_upper_kw": upper_total,
            "adaptive_midpoint_kw": 0.5 * (lower_total + upper_total),
            "adaptive_width_kw": upper_total - lower_total, "adaptive_cells": len(a),
            "adaptive_tail_kw": tail, "adaptive_cutoff_kw": float(grid[-1]),
            "adaptive_fp_allowance_kw": allowance_total,
            "adaptive_width_target_met": upper_total - lower_total <= max_width}


def init_worker() -> None:
    original.init_worker(str(CODE / "legacy_engine"), str(CODE / "legacy/run_support_policy_audit.py"),
                         128, 512, 64)


def run_chunk(task: tuple) -> tuple[list[dict], pd.DataFrame]:
    chunk, bounds, methods, limit, width, refinements = task
    fit_path = FIT_ROOT / chunk["fits"]
    stem = f'{chunk["start"]:05d}_{chunk["end"]-1:05d}'
    decision_path = BOUND_ROOT / f"primitive_decisions_{stem}.parquet"
    distance_path = BOUND_ROOT / f"primitive_distances_{stem}.parquet"
    for path, expected in ((fit_path, chunk["fits_sha256"]), (decision_path, bounds["decision_sha256"]),
                           (distance_path, bounds["dist_sha256"])):
        if original.sha256(path) != expected:
            raise ValueError(f"Archive hash mismatch for {path}")
    records = []
    for line in fit_path.read_text(encoding="utf-8").splitlines():
        fit = json.loads(line)
        rep, method = int(fit["replication_id"]), fit["method_id"]
        if rep < limit and method in methods:
            if fit["support_policy"] != "no_upper" or fit["upper_B_kw"] is not None:
                raise ValueError(f"Unexpected support policy for {rep}/{method}")
            dist = original.reconstruct_law(original._ENGINE, fit["law"])
            record = adaptive_bounds(dist, width, refinements)
            records.append(dict(replication_id=rep, method_id=method, **record))
    distances = pd.read_parquet(distance_path, columns=LAW_KEYS + [
        "w1_kw", "w1_cdf_riemann_lower_kw", "w1_cdf_riemann_upper_kw"])
    distances = distances[distances.replication_id.lt(limit) & distances.method_id.isin(methods)]
    laws = distances.merge(pd.DataFrame(records), on=LAW_KEYS, validate="one_to_one")
    if len(laws) != len(records) or len(laws) != (min(chunk["end"], limit) - chunk["start"]) * len(methods):
        raise ValueError(f"Law keys differ from archive for {stem}")
    # Both numerical constructions enclose the same integral in exact arithmetic.
    laws["combined_lower_kw"] = np.maximum(laws.adaptive_lower_kw, laws.w1_cdf_riemann_lower_kw)
    laws["combined_upper_kw"] = np.minimum(laws.adaptive_upper_kw, laws.w1_cdf_riemann_upper_kw)
    laws["combined_midpoint_kw"] = 0.5 * (laws.combined_lower_kw + laws.combined_upper_kw)
    decision_fields = LAW_KEYS + ["cell_id", "oracle_mode", "selected_mode", "oracle_margin_cny",
                                  "solver_zeta_bound_cny", "full_regret_cny",
                                  "primitive_gap_certificate_numerical",
                                  "primitive_gap_certificate_distributional_envelope"]
    decisions = pd.read_parquet(decision_path, columns=decision_fields)
    decisions = decisions[decisions.replication_id.lt(limit) & decisions.method_id.isin(methods)]
    if len(decisions) != 9 * len(laws) or decisions.duplicated(LAW_KEYS + ["cell_id"]).any():
        raise ValueError(f"Incomplete decision keys for {stem}")
    return laws.to_dict("records"), decisions


def decision_checks(laws: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    oracle = pd.read_csv(ORACLE_CSV).set_index("cell_id")
    decisions = decisions.merge(laws[LAW_KEYS + [
        "w1_cdf_riemann_upper_kw", "adaptive_upper_kw", "combined_upper_kw"]],
                                on=LAW_KEYS, validate="many_to_one")
    rate = float(original._SETTINGS["tariff"]["demand_rate"])
    policies = original._SETTINGS["tariff"]["policies"]
    kappas = {c["id"]: float(policy["kappa"]) for policy in policies for c in policy["cells"]}
    decisions["mode_error"] = decisions.selected_mode.ne(decisions.oracle_mode)
    coverage = []
    for name, upper in (("archived", "w1_cdf_riemann_upper_kw"), ("adaptive", "adaptive_upper_kw"),
                        ("combined", "combined_upper_kw")):
        for cell_id, d in decisions.groupby("cell_id"):
            kappa = kappas[cell_id]
            coefficients = dict(zip(MODES, (0.0, rate, rate * kappa), strict=True))
            L = max(coefficients.values())
            row = oracle.loc[cell_id]
            mode = row.oracle_mode
            if not d.oracle_mode.eq(mode).all():
                raise ValueError(f"Oracle mode mismatch for {cell_id}")
            best = float(row[f"oracle_value_{mode}_cny"])
            competitor_gaps = {m: float(row[f"oracle_value_{m}_cny"]) - best for m in MODES if m != mode}
            margin = min(competitor_gaps.values())
            if margin <= 0.0 or not np.allclose(d.oracle_margin_cny, margin, atol=1e-8, rtol=0.0):
                raise ValueError(f"Oracle gap mismatch for {cell_id}")
            water, zeta = d[upper], d.solver_zeta_bound_cny
            direct_pass = np.ones(len(d), dtype=bool)
            modewise_pass = np.ones(len(d), dtype=bool)
            for competitor, gap in competitor_gaps.items():
                coefficient = coefficients[mode] + coefficients[competitor]
                direct_k = (coefficient if "capacity" in (mode, competitor)
                            else rate * max(1.0, abs(1.0 - kappa)))
                direct_pass &= (direct_k * water + zeta < gap).to_numpy()
                modewise_pass &= (coefficient * water + zeta < gap).to_numpy()
            pooled_k = coefficients[mode] + max(coefficients[m] for m in competitor_gaps)
            checks = {"direct_pair_contrast": direct_pass, "modewise_per_competitor": modewise_pass,
                      "modewise_pooled_archived_baseline": pooled_k * water + zeta < margin,
                      "uniform_two_L_baseline": 2 * L * water + zeta < margin,
                      "uniform_L_full_action": L * water + zeta < margin}
            for label, passed in checks.items():
                decisions.loc[d.index, f"{name}_{label}"] = passed
            decisions.loc[d.index, f"{name}_full_regret_bound_cny"] = L * water + zeta
    decisions["primitive"] = decisions.primitive_gap_certificate_numerical
    decisions["coarse_distributional"] = decisions.primitive_gap_certificate_distributional_envelope
    flags = [label for label in decisions if label.startswith(("archived_", "adaptive_", "combined_"))
             and not label.endswith(("_kw", "_cny"))] + ["primitive", "coarse_distributional"]
    for label in flags:
        flag = decisions[label].astype(bool)
        coverage.append({"condition": label, "rows": len(flag), "passing_rows": int(flag.sum()),
                         "coverage_percent": 100 * float(flag.mean()),
                         "observed_mode_selection_violations": int((flag & decisions.mode_error).sum())})
    return decisions, pd.DataFrame(coverage)


def summarize_brackets(laws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, fit in laws.groupby("method_id"):
        for name, lower, upper in (("archived", "w1_cdf_riemann_lower_kw", "w1_cdf_riemann_upper_kw"),
                                   ("adaptive", "adaptive_lower_kw", "adaptive_upper_kw"),
                                   ("combined", "combined_lower_kw", "combined_upper_kw")):
            widths = fit[upper] - fit[lower]
            q50, q90, q99 = widths.quantile([0.50, 0.90, 0.99])
            rows.append({"method_id": method, "bound": name, "histories": len(fit),
                         "archived_quadrature_mean_kw": fit.w1_kw.mean(),
                         "bracket_midpoint_mean_kw": 0.5 * (fit[lower] + fit[upper]).mean(),
                         "lower_mean_kw": fit[lower].mean(),
                         "upper_mean_kw": fit[upper].mean(), "width_mean_kw": widths.mean(),
                         "width_p50_kw": q50, "width_p90_kw": q90, "width_p99_kw": q99,
                         "width_max_kw": widths.max()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--replications", type=int, default=10000)
    parser.add_argument("--methods", nargs="+", choices=("GEV", "KDE", "EMP", "EVENT-EMP", "TAIL"),
                        default=("GEV", "KDE", "EMP", "EVENT-EMP", "TAIL"))
    parser.add_argument("--max-width-kw", type=float, default=300.0)
    parser.add_argument("--max-refinements", type=int, default=2048)
    args = parser.parse_args()
    if (not 1 <= args.replications <= 10000 or len(args.methods) != len(set(args.methods))
            or not 0.0 < args.max_width_kw < math.inf or args.max_refinements < 1):
        parser.error("Invalid replication, method, or refinement specification")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fit_manifest_path = FIT_ROOT / "audit_manifest.json"
    bound_manifest_path = BOUND_ROOT / "primitive_bounds_manifest.json"
    fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
    bound_manifest = json.loads(bound_manifest_path.read_text(encoding="utf-8"))
    if (fit_manifest["status"] != "PASS" or bound_manifest["status"] != "PASS"
            or original.sha256(fit_manifest_path) != bound_manifest["stored_audit_manifest_sha256"]):
        raise ValueError("Fit and archived bound manifests do not align")
    archived_chunks = {c["start"]: c for c in bound_manifest["tasks"]}
    tasks = []
    for fit_chunk in fit_manifest["chunks"]:
        if fit_chunk["start"] < args.replications:
            match = archived_chunks[fit_chunk["start"]]
            if (fit_chunk["status"] != "PASS" or match["status"] != "PASS"
                    or fit_chunk["end"] != match["end"] or fit_chunk["fits_sha256"] != match["fits_sha256"]):
                raise ValueError(f"Fit and archived bound chunks do not align at {fit_chunk['start']}")
            tasks.append((fit_chunk, match, args.methods, args.replications, args.max_width_kw, args.max_refinements))
    law_parts, decision_parts = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for index, (labs, rows) in enumerate(pool.map(run_chunk, tasks), 1):
            law_parts.extend(labs)
            decision_parts.append(rows)
            if index % 25 == 0 or index == len(tasks):
                print(f"Processed {index}/{len(tasks)} chunks", flush=True)
    laws = pd.DataFrame(law_parts)
    if len(laws) != args.replications * len(args.methods) or laws.duplicated(LAW_KEYS).any():
        raise ValueError("Incomplete aggregate law archive")
    init_worker()
    decisions, coverage = decision_checks(laws, pd.concat(decision_parts, ignore_index=True))
    law_summary = summarize_brackets(laws)
    violations = int((decisions.full_regret_cny > decisions.combined_full_regret_bound_cny + 1e-4).sum())
    disjoint = int((laws.combined_lower_kw > laws.combined_upper_kw).sum())
    summary = {"status": "PASS" if violations == disjoint == coverage.observed_mode_selection_violations.sum() == 0
               else "FAIL", "histories": args.replications, "laws": len(laws), "decisions": len(decisions),
               "methods": args.methods, "workers": args.workers, "target_width_kw": args.max_width_kw,
               "max_refinements": args.max_refinements,
               "width_target_met": int(laws.adaptive_width_target_met.sum()),
               "full_regret_bound_violations": violations, "disjoint_with_archived_bracket": disjoint,
               "quadrature_points_outside_adaptive_bracket": int((
                   (laws.w1_kw < laws.adaptive_lower_kw) | (laws.w1_kw > laws.adaptive_upper_kw)).sum()),
               "interval_arithmetic": False,
               "inputs_sha256": {"fit_manifest": original.sha256(fit_manifest_path),
                                  "archived_bounds_manifest": original.sha256(bound_manifest_path),
                                  "oracle_values": original.sha256(ORACLE_CSV),
                                  "code": original.sha256(Path(__file__)),
                                  "primitives_code": original.sha256(Path(original.__file__)),
                                  "engine_code": original.sha256(CODE / "legacy_engine/scripts/run_revision_experiment.py"),
                                  "engine_config": original.sha256(CODE / "legacy_engine/configs/revision.yaml"),
                                  "fit_code": original.sha256(CODE / "legacy/run_support_policy_audit.py")}}
    laws.to_parquet(output / "adaptive_w1_by_law.parquet", index=False)
    decisions.to_parquet(output / "adaptive_w1_decision_checks.parquet", index=False)
    law_summary.to_csv(output / "adaptive_w1_by_method.csv", index=False)
    coverage.to_csv(output / "adaptive_w1_coverage.csv", index=False)
    original.write_json_atomic(output / "adaptive_w1_summary.json", summary)
    print(law_summary.to_string(index=False))
    print(coverage.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
