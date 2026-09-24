"""Event-level robustness DGPs and paired estimator evaluation for E7-E10."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from extreme_demand.decision import decision_from_draws
from extreme_demand.estimators import fit_empirical, fit_gev, fit_kde, fit_point, fit_process_tail
from extreme_demand.metrics import paired_result_row
from extreme_demand.policy import PolicyConfig, capacity_fee


ROBUSTNESS_METHODS = (
    "POINT-3M",
    "EMP-JOINT",
    "KDE-MC-JOINT",
    "GEV-JOINT",
    "TAIL-NODECLUSTER",
    "TAIL-JOINT",
)


def _entropy_values(seed: np.random.SeedSequence) -> list[int]:
    return [int(value) for value in np.atleast_1d(seed.entropy)]


def policy_for_scenario(base: PolicyConfig, scenario: dict[str, Any], settings: dict[str, Any]) -> PolicyConfig:
    policy_id = str(scenario["policy"])
    if policy_id == "main":
        return base
    if policy_id == "low_voltage":
        return replace(base, demand_rate=37.6, capacity_rate=23.5, voltage_level="1_35_kV")
    values = settings["policy_author_scenarios"][policy_id]
    return replace(
        base,
        alpha=float(values["alpha"]),
        kappa=float(values["kappa"]),
        delta=float(values["delta"]),
        contract_evidence_status="author_robustness_scenario",
    )


def _tail_fraction(family: str, uniforms: np.ndarray) -> np.ndarray:
    common = np.asarray(uniforms, dtype=float)
    if common.size == 0:
        return np.empty(0, dtype=float)
    if family in {"bounded_beta", "clustered"}:
        return stats.beta.ppf(common, 2.0, 3.0)
    if family == "lognormal":
        return np.clip(stats.lognorm.ppf(common, s=0.7, scale=np.exp(-1.0)), 0.0, 1.0)
    if family == "gpd":
        return np.clip(stats.genpareto.ppf(common, 0.20, scale=0.28), 0.0, 1.0)
    if family == "state_mixture":
        return np.select([common < 0.55, common < 0.90], [0.25, 0.55], default=1.0)
    raise ValueError(f"unsupported robustness truth family: {family}")


def _b4_peak_factor(settings: dict[str, Any]) -> float:
    template = settings["b4_normalized_template"]
    duration = np.asarray(template["duration_shares"], dtype=float)
    energy = np.asarray(template["energy_shares"], dtype=float)
    if not np.isclose(duration.sum(), 1.0) or not np.isclose(energy.sum(), 1.0):
        raise ValueError("B4 normalized duration and energy shares must each sum to one")
    intensity = energy / duration
    return float(intensity.max() / np.quantile(intensity, 0.75))


def generate_robustness_pool(
    scenario: dict[str, Any],
    settings: dict[str, Any],
    month_count: int,
    rng: np.random.Generator,
    pool_role: str,
    store_clusters: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical = settings["physical"]
    monthly_rows: list[dict[str, Any]] = []
    cluster_frames: list[pd.DataFrame] = []
    waveform_factor = _b4_peak_factor(settings) if scenario["waveform"] == "B4_NORMALIZED" else 1.0
    for month_id in range(month_count):
        heat_count = max(1, int(rng.poisson(float(physical["heat_rate_per_month"]))))
        beta_a = float(physical["base_beta_a"])
        beta_b = float(physical["base_beta_b"])
        base_fraction = rng.beta(beta_a, beta_b, size=heat_count)
        if scenario["waveform"] == "B4_NORMALIZED":
            base_fraction = np.clip(base_fraction ** (1.0 / waveform_factor), 0.0, 1.0)
        base_eaf = float(physical["base_eaf_floor_kw"]) + float(physical["base_eaf_span_kw"]) * base_fraction
        energy_noise = rng.normal(1.0, float(scenario["energy_cv"]), size=heat_count)
        base_eaf = np.clip(base_eaf * energy_noise, 0.0, float(physical["eaf_cap_kw"]) * 0.95)
        background_month = (
            float(physical["background_mean_kw"])
            + float(scenario["background_shift_kw"])
            + rng.normal(0.0, float(physical["background_month_sd_kw"]))
        )
        background = np.maximum(0.0, background_month + rng.normal(0.0, float(physical["background_event_sd_kw"]), size=heat_count))

        rare_cluster_count = int(rng.poisson(float(physical["rare_rate_per_month"])))
        rare_flag = np.zeros(heat_count, dtype=bool)
        rare_ids = np.full(heat_count, np.nan)
        rare_addition = np.zeros(heat_count, dtype=float)
        if rare_cluster_count:
            starts = rng.choice(heat_count, size=min(rare_cluster_count, heat_count), replace=False)
            fractions = _tail_fraction(str(scenario["family"]), rng.random(len(starts)))
            cluster_uniforms = rng.random(len(starts))
            for cluster_id, (start, fraction, cluster_uniform) in enumerate(zip(starts, fractions, cluster_uniforms, strict=True)):
                indices = [int(start)]
                if scenario["family"] == "clustered" and start + 1 < heat_count and cluster_uniform < 0.60:
                    indices.append(int(start + 1))
                for index in indices:
                    rare_flag[index] = True
                    rare_ids[index] = float(cluster_id)
                    headroom = max(0.0, float(physical["eaf_cap_kw"]) - base_eaf[index])
                    rare_addition[index] = (
                        fraction
                        * float(physical["rare_headroom_fraction"])
                        * headroom
                        * (1.0 - float(scenario["flexibility_fraction"]))
                    )
        eaf = np.minimum(float(physical["eaf_cap_kw"]), base_eaf + rare_addition)
        fixed = np.minimum(float(physical["poc_bound_kw"]), eaf + background)
        uplift = np.minimum(5000.0, np.abs(rng.normal(0.0, float(physical["sliding_uplift_scale_kw"]), size=heat_count)))
        sliding = np.minimum(float(physical["poc_bound_kw"]), fixed + uplift)
        selected = sliding if scenario["window"] == "sliding" else fixed
        monthly_rows.append(
            {
                "month_id": month_id,
                "pool_role": pool_role,
                "M_fixed15_kw": float(fixed.max()),
                "M_sliding15_kw": float(sliding.max()),
                "M_selected_kw": float(selected.max()),
                "heat_count": heat_count,
                "rare_cluster_count": rare_cluster_count,
                "max_eaf_kw": float(eaf.max()),
                "max_poc_kw": float(selected.max()),
            }
        )
        if store_clusters:
            cluster_frames.append(
                pd.DataFrame(
                    {
                        "month_id": month_id,
                        "heat_id": np.arange(heat_count, dtype=int),
                        "cluster_max_kw": selected,
                        "cluster_max_fixed_kw": fixed,
                        "cluster_max_sliding_kw": sliding,
                        "rare_event_flag": rare_flag,
                        "rare_cluster_id": rare_ids,
                    }
                )
            )
    monthly = pd.DataFrame(monthly_rows)
    clusters = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame()
    return monthly, clusters


def _fit_methods(
    history: pd.DataFrame,
    clusters: pd.DataFrame,
    settings: dict[str, Any],
    estimator_settings: dict[str, Any],
    mode: str,
    seed_sequence: np.random.SeedSequence,
) -> tuple[list[Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    child = seed_sequence.spawn(7)
    generators = [np.random.default_rng(value) for value in child]
    draws = int(settings["decision_draws"][mode])
    maxima = history["M_selected_kw"].to_numpy(dtype=float)
    month_frame = history[["month_id", "M_selected_kw"]].rename(columns={"M_selected_kw": "M_fixed15_kw"})
    tail = estimator_settings["tail"]
    fits = [
        fit_point(maxima, draws),
        fit_empirical(maxima, draws, generators[0]),
        fit_kde(maxima, draws, generators[1]),
        fit_gev(
            maxima,
            draws,
            generators[2],
            float(settings["physical"]["poc_bound_kw"]),
            float(estimator_settings["gev"]["truncation_fraction_gate"]),
        ),
    ]
    for decluster, rng_index in ((False, 3), (True, 5)):
        fits.append(
            fit_process_tail(
                month_frame,
                clusters,
                draws,
                generators[rng_index],
                generators[rng_index + 1],
                [float(value) for value in tail["threshold_quantiles"]],
                int(tail["minimum_exceedances_main"]),
                int(tail["minimum_exceedances_fallback"]),
                int(settings["bootstrap_replicates"][mode]),
                float(tail["holdout_fraction"]),
                float(tail["shape_ci_width_gate"]),
                float(settings["physical"]["poc_bound_kw"]),
                float(tail["truncation_fraction_gate"]),
                int(settings["validation_draws"][mode]),
                float(tail["stability_penalty_weight"]),
                decluster=decluster,
            )
        )
    stream_seeds = {
        "EMP-JOINT": [("method_mc", child[0])],
        "KDE-MC-JOINT": [("method_mc", child[1])],
        "GEV-JOINT": [("method_mc", child[2])],
        "TAIL-NODECLUSTER": [("method_mc", child[3]), ("bootstrap", child[4])],
        "TAIL-JOINT": [("method_mc", child[5]), ("bootstrap", child[6])],
    }
    ancestry = {
        "POINT-3M": {
            "type": "deterministic",
            "parent_entropy": _entropy_values(seed_sequence),
            "parent_spawn_key": list(seed_sequence.spawn_key),
        }
    }
    registry_rows: list[dict[str, Any]] = []
    for method_id, streams in stream_seeds.items():
        ancestry[method_id] = {
            "type": "seed_sequence_children",
            "streams": [
                {
                    "stream_name": stream_name,
                    "entropy": _entropy_values(stream_seed),
                    "spawn_key": list(stream_seed.spawn_key),
                }
                for stream_name, stream_seed in streams
            ],
        }
        registry_rows.extend(
            {
                "method_id": method_id,
                "stream": stream_name,
                "seed_entropy": json.dumps(_entropy_values(stream_seed)),
                "spawn_key": json.dumps(list(stream_seed.spawn_key)),
            }
            for stream_name, stream_seed in streams
        )
    return fits, ancestry, registry_rows


def evaluate_robustness_scenario(task: dict[str, Any]) -> dict[str, Any]:
    scenario = task["scenario"]
    settings = task["settings"]
    mode = str(task["mode"])
    if not bool(settings["common_random_numbers_across_scenarios"]):
        raise ValueError("Gate 8 requires common random numbers across registered scenarios")
    base_seed = np.random.SeedSequence(
        [int(settings["seed_root"]), int(settings["seed_namespace_prefix"])]
    )
    pool_seeds = base_seed.spawn(2)
    monthly_train, clusters_train = generate_robustness_pool(
        scenario,
        settings,
        int(settings["train_pool_months"][mode]),
        np.random.default_rng(pool_seeds[0]),
        "train",
        True,
    )
    monthly_test, _ = generate_robustness_pool(
        scenario,
        settings,
        int(settings["test_pool_months"][mode]),
        np.random.default_rng(pool_seeds[1]),
        "test",
        False,
    )
    pool_ancestry = {
        "train": {
            "entropy": _entropy_values(pool_seeds[0]),
            "spawn_key": list(pool_seeds[0].spawn_key),
        },
        "test": {
            "entropy": _entropy_values(pool_seeds[1]),
            "spawn_key": list(pool_seeds[1].spawn_key),
        },
    }
    for frame, role in ((monthly_train, "train"), (monthly_test, "test"), (clusters_train, "train")):
        frame["scenario_id"] = str(scenario["id"])
        frame["experiment"] = str(scenario["experiment"])
        frame["seed_root"] = int(settings["seed_root"])
        frame["seed_ancestry"] = json.dumps(pool_ancestry[role], sort_keys=True)
        frame["run_signature"] = str(task["run_signature"])
        frame["config_hash"] = str(task["config_hash"])
    truth = monthly_test["M_selected_kw"].to_numpy(dtype=float)
    cfg = policy_for_scenario(task["base_policy"], scenario, settings)
    training_lengths = [int(value) for value in settings["training_months"][mode]]
    maximum_history = max(training_lengths)
    replications = int(settings["replications"][mode])
    tie_fraction = 0.001
    paired_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = [
        {
            "scenario_id": scenario["id"],
            "replication_id": None,
            "training_months": None,
            "method_id": "DGP-POOL",
            "stream": f"{role}_pool",
            "seed_entropy": json.dumps(value["entropy"]),
            "spawn_key": json.dumps(value["spawn_key"]),
            "seed_root": int(settings["seed_root"]),
            "run_signature": str(task["run_signature"]),
            "config_hash": str(task["config_hash"]),
        }
        for role, value in pool_ancestry.items()
    ]
    replication_seeds = base_seed.spawn(replications)
    for replication, replication_seed in enumerate(replication_seeds):
        history_seed, *fit_seeds = replication_seed.spawn(1 + len(training_lengths))
        rng = np.random.default_rng(history_seed)
        selected = rng.choice(len(monthly_train), size=maximum_history, replace=False)
        seed_rows.append(
            {
                "scenario_id": scenario["id"],
                "replication_id": replication,
                "training_months": None,
                "method_id": "HISTORY-SAMPLING",
                "stream": "history",
                "seed_entropy": json.dumps(history_seed.entropy if isinstance(history_seed.entropy, list) else [history_seed.entropy]),
                "spawn_key": json.dumps(history_seed.spawn_key),
                "seed_root": int(settings["seed_root"]),
                "run_signature": str(task["run_signature"]),
                "config_hash": str(task["config_hash"]),
            }
        )
        for length, fit_seed in zip(training_lengths, fit_seeds, strict=True):
            ids = selected[:length]
            history = monthly_train.iloc[ids].copy().reset_index(drop=True)
            old_months = history["month_id"].to_numpy(dtype=int)
            mapping = {int(old): new for new, old in enumerate(old_months)}
            history["month_id"] = np.arange(length, dtype=int)
            clusters = clusters_train[clusters_train["month_id"].isin(old_months)].copy()
            clusters["month_id"] = clusters["month_id"].map(mapping).astype(int)
            fits, method_ancestry, method_seed_rows = _fit_methods(
                history, clusters, settings, task["estimator_settings"], mode, fit_seed
            )
            history_metadata = {
                "entropy": _entropy_values(history_seed),
                "spawn_key": list(history_seed.spawn_key),
            }
            seed_rows.extend(
                {
                    **row,
                    "scenario_id": scenario["id"],
                    "replication_id": replication,
                    "training_months": length,
                    "seed_root": int(settings["seed_root"]),
                    "run_signature": str(task["run_signature"]),
                    "config_hash": str(task["config_hash"]),
                }
                for row in method_seed_rows
            )
            for fit in fits:
                row_ancestry = json.dumps(
                    {
                        "history": history_metadata,
                        "fit": method_ancestry[fit.method_id],
                    },
                    sort_keys=True,
                )
                fit_rows.append(
                    fit.as_row(
                        scenario_id=scenario["id"],
                        experiment=scenario["experiment"],
                        family=scenario["family"],
                        replication_id=replication,
                        training_months=length,
                        seed_root=int(settings["seed_root"]),
                        seed_ancestry=row_ancestry,
                        run_signature=task["run_signature"],
                        config_hash=task["config_hash"],
                    )
                )
            for utilization in settings["utilization_grid"]:
                S_kva = float(truth.mean() / float(utilization))
                capacity_monthly = float(capacity_fee(S_kva, cfg))
                tie_epsilon = tie_fraction * 3.0 * capacity_monthly
                oracle = decision_from_draws("DIST-ORACLE", truth, S_kva, cfg, tie_epsilon)
                ordered_costs = sorted(oracle.mode_costs.values())
                oracle_margin = float(ordered_costs[1] - ordered_costs[0])
                for fit in fits:
                    row_ancestry = json.dumps(
                        {
                            "history": history_metadata,
                            "fit": method_ancestry[fit.method_id],
                        },
                        sort_keys=True,
                    )
                    decision = decision_from_draws(fit.method_id, fit.draws_kw, S_kva, cfg, tie_epsilon)
                    paired_rows.append(
                        paired_result_row(
                            decision,
                            oracle,
                            truth,
                            S_kva,
                            cfg,
                            scenario_id=scenario["id"],
                            experiment=scenario["experiment"],
                            family=scenario["family"],
                            window=scenario["window"],
                            policy_id=scenario["policy"],
                            waveform=scenario["waveform"],
                            flexibility_fraction=float(scenario["flexibility_fraction"]),
                            replication_id=replication,
                            training_months=length,
                            utilization_target=float(utilization),
                            capacity_fee_monthly=capacity_monthly,
                            oracle_margin=oracle_margin,
                            normalized_oracle_margin=oracle_margin / (3.0 * capacity_monthly),
                            normalized_regret=0.0,
                            seed_root=int(settings["seed_root"]),
                            seed_ancestry=row_ancestry,
                            run_signature=task["run_signature"],
                            config_hash=task["config_hash"],
                        )
                    )
    paired = pd.DataFrame(paired_rows)
    if set(paired["method_id"]) != set(ROBUSTNESS_METHODS):
        raise RuntimeError("robustness method registry mismatch")
    paired["normalized_regret"] = paired["regret"] / (3.0 * paired["capacity_fee_monthly"])
    q999 = float(np.quantile(truth, 0.999))
    feasible = {
        str(value): bool(q999 <= (truth.mean() / float(value)) * float(settings["physical"]["power_factor"]))
        for value in settings["utilization_grid"]
    }
    return {
        "scenario_id": scenario["id"],
        "monthly_train": monthly_train,
        "monthly_test": monthly_test,
        "clusters_train": clusters_train,
        "paired": paired,
        "fits": pd.DataFrame(fit_rows),
        "seeds": pd.DataFrame(seed_rows),
        "truth_mean_kw": float(truth.mean()),
        "truth_q999_kw": q999,
        "feasibility": feasible,
    }
