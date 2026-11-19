"""Paired excess-law ablation and fixed-threshold sensitivity on the frozen CSSC histories."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
import hashlib
import json
import math
import os
from pathlib import Path
import time

import run_nb_weibull_cv_crps as nb

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

import run_primary_no_nominal_ks as primary


ROOT = nb.ROOT
REVISION = ROOT / "CSSC_REVIEW_REVISION_20261112"
HISTORIES = REVISION / "history_reconstructed"
PRIMARY = REVISION / "primary_cv_crps_decisions_20261115"
PRIMARY_CV = REVISION / "primary_cv_crps_20261115"
NB = REVISION / "nb_weibull_cv_crps_20261115"
BASE = nb.DEFAULT_BASE
OUTPUT = REVISION / "tail_ablation_20261119"
FIXED = 0.95
TASK_ID = "cssc-tail-ablation-20261119"
REGIONS = nb.ORACLE_REGION
METHODS = ("TAIL", "TAIL-EMP-EXCESS", "TAIL-FIXED-Q95-SENS")
DECISION_COLUMNS = ["replication_id", "method_id", "cell_id", "S_kva", "oracle_mode",
                    "oracle_best_value_cny", "oracle_contract_D_kw", "selected_mode",
                    "estimated_contract_D_kw", "estimated_mean_kw", "regret_cny_per_month"]
METRICS = ("regret_cny", "mode_regret_cny", "within_contract_regret_cny", "action_correct",
           "mean_abs_error_pct", "q_pstar_abs_error_pct", "stoploss_abs_error_pct")
INTEGRITY = {}


def verify(path, digest):
    if primary.sha256(path) != digest:
        raise AssertionError(f"archive hash mismatch for {path}")
    return path


@cache
def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def from_saved(law, engine):
    return engine.TailProcessDistribution(
        law["threshold_kw"], law["xi"], law["sigma_kw"],
        np.asarray(law["baseline_monthly_maxima_kw"]),
        law["count_model"], law["count_mean"], law["count_variance"],
        law["nb_size"], law["nb_probability"],
    )


def components(events, months, threshold, engine):
    values = events.cluster_max_kw.to_numpy(float)
    time_ids = events.month_id.to_numpy()
    positions = np.searchsorted(months, time_ids)
    if np.any(positions >= len(months)) or not np.array_equal(months[positions], time_ids):
        raise AssertionError("event month does not map to saved training panel")
    above = values > threshold
    counts = np.bincount(positions[above], minlength=len(months))
    baseline = np.zeros(len(months))
    np.maximum.at(baseline, positions[~above], values[~above])
    return values[above], baseline, engine._count_parameters(counts)


def assert_components(dist, baseline, count_params):
    if not np.array_equal(dist.baseline, baseline):
        raise AssertionError("subthreshold maxima differ from archived full fit")
    actual = dist.count_model, dist.count_mean, dist.count_variance, dist.nb_size, dist.nb_probability
    if actual != count_params:
        raise AssertionError(f"count fit differs from archived full fit: {actual} vs {count_params}")


def empirical_law(dist, tail_values, engine):
    """Push the ECDF through the saved count PGF and keep its original bulk law."""
    bulk, bulk_counts = np.unique(dist.baseline, return_counts=True)
    tail, tail_counts = np.unique(tail_values, return_counts=True)
    if len(tail) == 0 or np.max(bulk) > dist.threshold or np.min(tail) <= dist.threshold:
        raise AssertionError("empirical support fails the fitted threshold split")
    survival = (tail_counts.sum() - np.cumsum(tail_counts)) / tail_counts.sum()
    if dist.count_model == "poisson":
        tail_cdf = np.exp(-dist.count_mean * survival)
    else:
        prob = dist.nb_probability
        tail_cdf = np.exp(-dist.nb_size * np.log1p(((1.0 - prob) / prob) * survival))
    p0 = dist._pgf(0.0)
    masses = np.r_[p0 * bulk_counts / sum(bulk_counts), np.diff(np.r_[p0, tail_cdf])]
    support = np.r_[bulk, tail]
    law = engine.DiscreteDistribution(support, masses)
    if not math.isclose(law.cdf(dist.threshold), p0, rel_tol=1e-12):
        raise AssertionError("empirical replacement changed the zero-count probability")
    return law


def fixed_fit(events, months, engine, settings, rep, saved, quantile, folds, exact_audit):
    """Apply the fixed candidate's archived fold checks and refit on all months."""
    if len(folds) != 4 or {row["fold"] for row in folds} != set(range(4)):
        raise AssertionError("fixed candidate requires the four saved held-out folds")
    fold_pass = all(
        row["status"] == "PASS" and row["fit_months"] == 18 and row["validation_months"] == 6
        and np.isfinite(row["crps"]) for row in folds
    )
    cfg = settings["tail"]
    threshold = float(np.quantile(events.cluster_max_kw.to_numpy(float), FIXED))
    tail, baseline, params = components(events, months, threshold, engine)
    if not fold_pass:
        raise AssertionError(f"fixed 0.95 candidate fails the original CV eligibility rule: {rep}")
    if len(tail) < cfg["minimum_exceedances_main"]:
        raise AssertionError(f"fixed 0.95 candidate fails full-history count eligibility: {rep}")
    if quantile == FIXED:
        assert_components(saved, baseline, params)
        if saved.threshold != threshold:
            raise AssertionError("full and fixed 0.95 thresholds differ")
        return saved, {"fixed_status": "PASS_ARCHIVED_Q95", "fixed_bootstrap_width": None,
                       "fixed_bootstrap_would_fallback": False}
    excesses = tail - threshold
    xi, _, sigma = map(float, stats.genpareto.fit(excesses, floc=0.0))
    invalid = (not np.isfinite([xi, sigma]).all() or sigma <= 0.0 or xi >= 1.0
               or (xi < 0.0 and threshold - sigma / xi < tail.max() - 1e-9))
    if invalid:
        xi, sigma = 0.0, float(excesses.mean())
    audit_width = None
    if exact_audit:
        g = engine.fit_process_tail.__globals__
        rng = engine.seed_rng(settings["seed_root"], 2, 0, rep, 20, 2)
        audit_width = float(g["_shape_ci_width"](excesses, 500, rng))
    return engine.TailProcessDistribution(threshold, xi, sigma, baseline, *params), {
        "fixed_status": "FALLBACK_EXPONENTIAL" if invalid else "PASS_POINT_FIT",
        "fixed_bootstrap_width": audit_width,
        "fixed_bootstrap_would_fallback": None if audit_width is None else
        bool(invalid or audit_width > cfg["shape_ci_width_gate"]),
    }


def read_primary(lo, hi, decision_manifest, cv_manifest):
    ids = list(range(lo, hi))
    raw = HISTORIES / "raw"
    month_df = pd.read_parquet(raw / "monthly_maxima_10000.parquet",
                               filters=[("replication_id", "in", ids)])
    event_df = pd.read_parquet(raw / "event_clusters_10000.parquet",
                               filters=[("replication_id", "in", ids)])
    parts = []
    saved = {}
    for item in decision_manifest["chunks"]:
        if not lo <= item["start"] < hi:
            continue
        path = PRIMARY / "support_policy_audit" / decision_manifest["signature"][:16]
        fit_path = verify(path / item["fits"], item["fits_sha256"])
        with fit_path.open(encoding="utf-8") as stream:
            for line in stream:
                if '"method_id": "TAIL"' in line[:95]:
                    record = json.loads(line)
                    saved[record["replication_id"]] = record["fit_diagnostics"], record["law"]
        path = verify(path / item["csv"], item["csv_sha256"])
        frame = pd.read_csv(path, usecols=DECISION_COLUMNS + [
            "estimated_contract_quantile_p_star_kw", "estimated_stoploss_kw"])
        parts.append(frame[frame.method_id == "TAIL"])
    decisions = pd.concat(parts, ignore_index=True)
    decisions = decisions.rename(columns={
        "estimated_contract_quantile_p_star_kw": "archived_quantile_kw",
        "estimated_stoploss_kw": "archived_stoploss_kw",
    })
    folds = {}
    for item in cv_manifest["chunks"]:
        if not (item["start"] < hi and lo < item["end"] and item["end"] - item["start"] == 100):
            continue
        stem = f"cv_{item['start']:05d}_{item['end']-1:05d}_folds.csv"
        data = pd.read_csv(verify(PRIMARY_CV / stem, item["fold_sha256"]))
        for rep, rows in data[data["quantile"].eq(FIXED) & data.replication_id.isin(ids)].groupby("replication_id"):
            folds[rep] = rows.to_dict("records")
    return month_df, event_df, decisions, saved, folds, None


def read_nb(lo, hi, manifest):
    fits, maxima, seeds, decisions = [], [], [], []
    for item in manifest["chunk_manifest"]:
        if not lo <= item["start"] < hi:
            continue
        read = {}
        for artifact in item["artifacts"]:
            path = Path(artifact["path"])
            kind = path.parent.name
            cols = {
                "fits": ["replication_id", "method_id", "diagnostics_json",
                         "law_parameters_json", "candidate_diagnostics_json"],
                "decisions": DECISION_COLUMNS + ["estimated_distribution_q_p_star_kw"],
                "monthly_maxima": None,
                "seeds": None,
            }[kind]
            local = NB / "chunks" / kind / path.name
            read[kind] = pd.read_parquet(verify(local, artifact["sha256"]), columns=cols)
        fits.append(read["fits"].loc[lambda f: f.method_id.eq("TAIL")])
        decisions.append(read["decisions"].loc[lambda f: f.method_id.eq("TAIL")])
        maxima.append(read["monthly_maxima"])
        seeds.append(read["seeds"])
    saved, folds = {}, {}
    for r in pd.concat(fits, ignore_index=True).itertuples(index=False):
        saved[r.replication_id] = json.loads(r.diagnostics_json), json.loads(r.law_parameters_json)
        folds[r.replication_id] = [f for f in json.loads(r.candidate_diagnostics_json)["fold_scores"]
                                   if f["quantile"] == FIXED]
    base = pd.concat(decisions, ignore_index=True).rename(columns={
        "estimated_distribution_q_p_star_kw": "archived_quantile_kw",
    })
    return pd.concat(maxima, ignore_index=True), None, base, saved, folds, (
        pd.concat(seeds, ignore_index=True).set_index("replication_id"))


def restored_nb(rep, monthly, seeds, oracle):
    regenerated, events, record = nb._generate_history(rep, oracle)
    keys = ["seed_entropy", "bit_generator", "monthly_maxima_sha256_f64le", "tail_counts_sha256_i32le"]
    if any(record[key] != seeds.loc[rep, key] for key in keys):
        raise AssertionError(f"NB reconstructed history fingerprint differs for {rep}")
    for col in ("month_id", "M_fixed15_kw", "tail_event_count", "tail_count_zero"):
        if not np.array_equal(regenerated[col], monthly[col]):
            raise AssertionError(f"NB reconstructed {col} differs for {rep}")
    return events


class CachedDistribution:
    def __init__(self, distribution):
        self.ppf = cache(distribution.ppf)
        self.mean = cache(distribution.mean)
        self.stop_loss = cache(distribution.stop_loss)


def decision_metrics(engine, dist, oracle, cell, actual, evaluate_full=False):
    policy, truth, size = cell["policy"], cell["oracle"], cell["S_kva"]
    target_q = policy.contract_probability
    true_mean, true_q, true_stoploss = cell["true_functionals"]
    if actual is None or evaluate_full:
        est = engine.optimized_decision(dist, size, policy)
        action, contract = est["selected_mode"], est["contract_D_kw"]
        cost = engine.true_cost_of_estimated_action(action, contract, oracle, size, policy)
        regret = max(0.0, cost - truth["best_value_cny"])
        mean, quantile = dist.mean(), dist.ppf(target_q)
    else:
        action, contract, regret, mean, quantile = (
            actual.selected_mode, actual.estimated_contract_D_kw, actual.regret_cny_per_month,
            actual.estimated_mean_kw, actual.archived_quantile_kw)
    stoploss = dist.stop_loss(policy.alpha * truth["contract_D_kw"])
    if actual is not None:
        if (actual.oracle_mode != truth["selected_mode"]
            or not np.allclose([actual.S_kva, actual.oracle_best_value_cny, actual.oracle_contract_D_kw],
                               [size, truth["best_value_cny"], truth["contract_D_kw"]], rtol=1e-12)):
            raise AssertionError("tariff truth differs from the saved decision input")
        if evaluate_full and (actual.selected_mode != action or not np.allclose(
            [actual.estimated_contract_D_kw, actual.regret_cny_per_month,
             actual.estimated_mean_kw, actual.archived_quantile_kw],
            [contract, regret, mean, quantile], rtol=1e-10, atol=1e-6)):
            raise AssertionError("rebuilt full TAIL does not reproduce saved decisions")
        if hasattr(actual, "archived_stoploss_kw") and not math.isclose(
                stoploss, actual.archived_stoploss_kw, rel_tol=1e-11):
            raise AssertionError("rebuilt full TAIL does not reproduce the saved stop-loss")
    mode_loss = truth["mode_costs"][action] - truth["best_value_cny"]
    within = regret - mode_loss
    if within < -1e-6:
        raise AssertionError(f"negative within-contract loss: {within}")
    return {
        "oracle_region": REGIONS[truth["selected_mode"]], "selected_mode": action,
        "regret_cny": regret, "mode_regret_cny": mode_loss,
        "within_contract_regret_cny": max(0.0, within),
        "action_correct": float(action == truth["selected_mode"]),
        "mean_abs_error_pct": 100.0 * abs(mean / true_mean - 1),
        "q_pstar_abs_error_pct": 100.0 * abs(quantile / true_q - 1),
        "stoploss_abs_error_pct": 100.0 * abs(stoploss / true_stoploss - 1),
        "estimated_contract_kw": contract,
    }


def run_chunk(dgp, lo, hi, source, audit_to):
    engine = primary.load_engine(BASE)
    settings = nb.read_settings(BASE)
    if dgp == "primary":
        monthly, events, archived, fits, folds, seeds = read_primary(lo, hi, *source)
        oracle = engine.OracleDistribution(engine.dgp_from_settings(settings, 30.0))
        cells = engine.policies_and_cells(settings, oracle, alpha120_only=False)
    else:
        settings["seed_root"] = nb.SEED_ROOT
        monthly, events, archived, fits, folds, seeds = read_nb(lo, hi, source)
        oracle = nb.NonPoissonWeibullOracle()
        cells = nb.build_plan(settings, engine, oracle)
    for cell in cells:
        policy = cell["policy"]
        cell["true_functionals"] = (
            oracle.mean(), oracle.ppf(policy.contract_probability),
            oracle.stop_loss(policy.alpha * cell["oracle"]["contract_D_kw"]),
        )
    month_groups = dict(tuple(monthly.groupby("replication_id")))
    event_groups = None if events is None else dict(tuple(events.groupby("replication_id")))
    decision_groups = dict(tuple(archived.groupby("replication_id")))
    ids = set(range(lo, hi))
    if any(set(table) != ids for table in (month_groups, fits, folds, decision_groups)):
        raise AssertionError("incomplete or mismatched archive coverage")
    rows, diagnostics = [], []
    for rep in range(lo, hi):
        months = month_groups[rep].sort_values("month_id")
        panel = (restored_nb(rep, months, seeds, oracle) if dgp == "nb_weibull" else
                 event_groups[rep])
        month_ids = months.month_id.to_numpy()
        if len(month_ids) != 24 or len(np.unique(month_ids)) != 24:
            raise AssertionError("ablation requires 24 training months")
        diag, full_law = fits[rep]
        if diag["fit_status"] != "PASS" or full_law["class"] != "TailProcessDistribution":
            raise AssertionError("full fit did not return an eligible GPD law")
        q = diag["selected_threshold_quantile"]
        full = from_saved(full_law, engine)
        if full.threshold != float(np.quantile(panel.cluster_max_kw.to_numpy(float), q)):
            raise AssertionError("archived threshold differs from the event-panel quantile")
        tail, baseline, params = components(panel, month_ids, full.threshold, engine)
        assert_components(full, baseline, params)
        empirical = empirical_law(full, tail, engine)
        fixed, fixed_status = fixed_fit(panel, month_ids, engine, settings, rep, full, q,
                                        folds[rep], rep < audit_to)
        saved_rows = {r.cell_id: r for r in decision_groups[rep].itertuples(index=False)}
        if len(saved_rows) != len(cells) or len(decision_groups[rep]) != len(cells):
            raise AssertionError("archive duplicates or omits a tariff cell")
        methods = [
            ("TAIL", CachedDistribution(full)),
            ("TAIL-EMP-EXCESS", CachedDistribution(empirical)),
            ("TAIL-FIXED-Q95-SENS", CachedDistribution(fixed)),
        ]
        diagnostics.append({
            "dgp": dgp, "replication_id": rep, "selected_q": q,
            "selected_threshold_kw": full.threshold, "selected_count_model": full.count_model,
            "selected_count_mean": full.count_mean, "selected_tail_events": len(tail),
            "selected_sample_max_kw": max(tail), "n_events": len(panel),
            "fixed_threshold_kw": fixed.threshold, "fixed_count_model": fixed.count_model,
            "fixed_count_mean": fixed.count_mean,
            "fixed_cv_crps_kw": float(np.mean([f["crps"] for f in folds[rep]])),
            "fixed_cv_folds_passed": len(folds[rep]),
            **fixed_status,
        })
        for cell in cells:
            actual = saved_rows[cell["cell_id"]]
            for method, dist in methods:
                saved = actual if method == "TAIL" or (method == "TAIL-FIXED-Q95-SENS" and q == FIXED) else None
                metrics = decision_metrics(engine, dist, oracle, cell, saved,
                                           evaluate_full=rep < audit_to and saved is not None)
                rows.append({"dgp": dgp, "replication_id": rep, "cell_id": cell["cell_id"],
                             "method_id": method, **metrics})
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def summaries(decisions):
    keys = ["dgp", "replication_id", "oracle_region", "method_id"]
    regional = decisions.groupby(keys, as_index=False)[list(METRICS)].mean()
    extra = decisions.groupby(["dgp", "replication_id", "method_id"], as_index=False)[list(METRICS)].mean()
    extra["oracle_region"] = "all_nine_cells"
    paired = pd.concat([regional, extra], ignore_index=True)
    out = []
    for (dgp, region), frame in paired.groupby(["dgp", "oracle_region"]):
        for method in METHODS[1:]:
            reference = frame[frame.method_id.eq("TAIL")].sort_values("replication_id")
            comparator = frame[frame.method_id.eq(method)].sort_values("replication_id")
            if not np.array_equal(reference.replication_id, comparator.replication_id):
                raise AssertionError("paired summary dropped or reordered histories")
            n = len(reference)
            for metric in METRICS:
                diff = reference[metric].to_numpy() - comparator[metric].to_numpy()
                est = float(diff.mean())
                se = float(diff.std(ddof=1) / np.sqrt(n))
                critical = float(stats.t.ppf(0.975, n - 1))
                out.append({"dgp": dgp, "oracle_region": region, "alternative": method,
                            "metric": metric, "paired_history_n": n,
                            "mean_full": float(reference[metric].mean()),
                            "mean_alternative": float(comparator[metric].mean()),
                            "mean_diff_full_minus_alternative": est, "paired_se": se,
                            "ci95_lower": est - critical * se, "ci95_upper": est + critical * se})
    return pd.DataFrame(out)


def plot_contrasts(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    regions = ["actual", "contract", "capacity"]
    sub = summary[summary.metric.eq("regret_cny") & summary.oracle_region.isin(regions)]
    for ax, dgp, title in zip(axes, ("primary", "nb_weibull"), ("Poisson–GPD", "NB–Weibull")):
        for i, (method, label, color) in enumerate(zip(
                METHODS[1:], ("Empirical excess", "Fixed 0.95 sensitivity"), ("#126B77", "#A35D38"))):
            rows = sub[sub.dgp.eq(dgp) & sub.alternative.eq(method)].set_index("oracle_region").loc[regions]
            mean = rows.mean_diff_full_minus_alternative / 1000
            limits = np.vstack([mean - rows.ci95_lower / 1000, rows.ci95_upper / 1000 - mean])
            ax.errorbar(mean, np.arange(len(regions)) + (i - 0.5) * 0.22, xerr=limits, fmt="o",
                        capsize=3, label=label, color=color)
        ax.axvline(0, color="#666666", lw=1)
        ax.set(yticks=np.arange(len(regions)), yticklabels=regions, title=title,
               xlabel="Full TAIL minus variant (thousand CNY/month)")
        ax.invert_yaxis()
        ax.legend(frameon=False, loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def source_inputs():
    primary_dir = PRIMARY / "support_policy_audit"
    decision_path, = primary_dir.glob("b92088b6a7e3c5ae/audit_manifest.json")
    cv_path = PRIMARY_CV / "run_manifest.json"
    with (HISTORIES / "offline_history_manifest.json").open(encoding="utf-8") as stream:
        history_manifest = json.load(stream)
    for label in ("monthly_maxima", "event_clusters"):
        verify(HISTORIES / "raw" / f"{label}_10000.parquet",
               history_manifest["outputs"][label + "_sha256"])
    nb_path = NB / "run_manifest.json"
    for path, key in ((decision_path, "primary full decisions"),
                      (cv_path, "primary archived CV"),
                      (nb_path, "NB frozen generator and fits"),
                      (HISTORIES / "offline_history_manifest.json", "primary event panels")):
        INTEGRITY[key] = primary.sha256(path)
    for path in (Path(__file__), Path(nb.__file__), Path(primary.__file__),
                 Path(primary.__file__).with_name("primary_cv_crps.py"),
                 BASE / "scripts" / "run_revision_experiment.py", BASE / "configs" / "revision.yaml"):
        INTEGRITY[path.name] = primary.sha256(path)
    nb_manifest = json_file(nb_path)
    if (INTEGRITY[Path(nb.__file__).name] != nb_manifest["runner_code_sha256"] or
        INTEGRITY["run_revision_experiment.py"] != nb_manifest["engine_sha256"] or
        INTEGRITY["revision.yaml"] != nb_manifest["tariff_config_sha256"] or
        INTEGRITY[Path(primary.__file__).name] != nb_manifest["no_upper_fit_source_sha256"] or
        nb_manifest["seed_root"] != nb.SEED_ROOT or nb_manifest["seed_namespace"] != nb.SEED_NAMESPACE):
        raise AssertionError("local NB generator or estimator differs from its archived provenance")
    return {"primary": (json_file(decision_path), json_file(cv_path)), "nb_weibull": nb_manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exact-audit-first", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 0 < args.histories <= 10000 or args.histories % 25:
        parser.error("--histories must be a positive multiple of 25, at most 10000")
    if not 0 <= args.exact_audit_first <= args.histories:
        parser.error("--exact-audit-first must be between zero and --histories")
    started = time.perf_counter()
    sources = source_inputs()
    args.output.mkdir(parents=True, exist_ok=True)
    starts = range(0, args.histories, 100)
    decisions, fits = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for lo in starts:
            for dgp, source in sources.items():
                hi = min(lo + 100, args.histories)
                future = pool.submit(run_chunk, dgp, lo, hi, source, args.exact_audit_first)
                futures[future] = (dgp, lo, hi)
        for future in as_completed(futures):
            chunk_decisions, chunk_fits = future.result()
            decisions.append(chunk_decisions)
            fits.append(chunk_fits)
            dgp, lo, hi = futures[future]
            if lo % 500 == 0:
                print(f"{dgp} {lo}:{hi} complete", flush=True)
    sort_keys = ["dgp", "replication_id", "cell_id", "method_id"]
    decisions = pd.concat(decisions, ignore_index=True).sort_values(sort_keys)
    fits = pd.concat(fits, ignore_index=True).sort_values(["dgp", "replication_id"])
    expected = args.histories * 2 * 9 * len(METHODS)
    if len(decisions) != expected or decisions.duplicated(sort_keys).any():
        raise AssertionError("unexpected count or duplicate paired decision records")
    if len(fits) != args.histories * 2 or fits.duplicated(["dgp", "replication_id"]).any():
        raise AssertionError("unexpected count or duplicate paired fit records")
    same = decisions.merge(fits[fits.selected_q.eq(FIXED)][["dgp", "replication_id"]])
    cols = [metric for metric in METRICS if metric in same]
    a = same[same.method_id.eq("TAIL")][cols].to_numpy()
    b = same[same.method_id.eq("TAIL-FIXED-Q95-SENS")][cols].to_numpy()
    if not np.array_equal(a, b):
        raise AssertionError("fixed and CV comparisons differ when CV picked q=0.95")
    contrast = summaries(decisions)
    decisions.to_parquet(args.output / "history_cell_metrics.parquet", index=False)
    fits.to_parquet(args.output / "history_fit_checks.parquet", index=False)
    contrast.to_csv(args.output / "paired_summary.csv", index=False)
    plot_contrasts(contrast, args.output / "paired_regret.png")
    manifest = {
        "status": "PASS", "task_id": TASK_ID, "histories_per_dgp": args.histories,
        "decision_rows": int(len(decisions)), "fits": int(len(fits)),
        "fixed_threshold": FIXED, "workers": args.workers,
        "same_q_checked": fits[fits.selected_q.eq(FIXED)].groupby("dgp").size().to_dict(),
        "fixed_cv_eligible": fits[fits.fixed_cv_folds_passed.eq(4)].groupby("dgp").size().to_dict(),
        "fixed_status_counts": {dgp: frame.fixed_status.value_counts().to_dict()
                                for dgp, frame in fits.groupby("dgp")},
        "exact_shape_bootstrap_audit": {
            "first_history_ids_per_dgp": args.exact_audit_first,
            "nonmatching_q_bootstrapped_n": int(fits.fixed_bootstrap_width.notna().sum()),
            "max_shape_interval_width": (float(fits.fixed_bootstrap_width.max())
                                         if fits.fixed_bootstrap_width.notna().any() else None),
            "would_fallback_n": int(fits.fixed_bootstrap_would_fallback.eq(True).sum()),
        },
        "fixed_sensitivity_limitation": "For histories whose archived adaptive threshold was not 0.95, "
            "the fixed quantile uses the same archived four-fold eligibility checks and full-sample "
            "point-fit count/support/finite-mean handling, with the 500-resample GPD shape-width "
            "gate omitted. Pilot shape bootstraps are diagnostics only.",
        "source_hashes": INTEGRITY,
        "output_hashes": {p.name: primary.sha256(p) for p in sorted(args.output.iterdir())
                          if p.name in {"history_cell_metrics.parquet", "history_fit_checks.parquet",
                                        "paired_summary.csv", "paired_regret.png"}},
        "duration_seconds": time.perf_counter() - started,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
