"""Structured transformations for the five registered manuscript tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def policy_evidence_table(policy: Mapping[str, Any]) -> pd.DataFrame:
    rates = policy["rates"]
    rule = policy["contracted_rule"]
    measurement = policy["demand_measurement"]
    timing = policy["decision_timing"]
    rows = [
        {"item": "Charging modes", "main_value": ", ".join(policy["modes"]), "evidence_status": "official national policy", "sensitivity": "none"},
        {"item": "110 kV rates", "main_value": f"demand={rates['110_kV']['demand_cny_per_kw_month']}; capacity={rates['110_kV']['capacity_cny_per_kva_month']}", "evidence_status": "official Yunnan rates effective 2026-08-01", "sensitivity": "lower-voltage official rate pair"},
        {"item": "Contract tolerance alpha", "main_value": str(rule["tolerance_alpha"]), "evidence_status": str(rule["evidence_status"]), "sensitivity": "1.00 and 1.10 author scenarios"},
        {"item": "Excess multiplier kappa", "main_value": str(rule["excess_total_multiplier_kappa"]), "evidence_status": str(rule["evidence_status"]), "sensitivity": "1.50 and 2.50 author scenarios"},
        {"item": "Minimum contract ratio delta", "main_value": str(rule["minimum_contract_ratio_delta"]), "evidence_status": str(rule["evidence_status"]), "sensitivity": "0.30 and 0.50 author scenarios"},
        {"item": "Demand measurement", "main_value": str(measurement["main_variant"]), "evidence_status": str(measurement["evidence_status"]), "sensitivity": ", ".join(measurement["robustness_variants"])},
        {"item": "Mode commitment / D update", "main_value": f"{timing['mode_commitment_months']} months / {timing['contract_update_months']} month", "evidence_status": "historical timing scenario", "sensitivity": "one-month theory variant"},
    ]
    return pd.DataFrame(rows)


def provenance_table(generator: Mapping[str, Any], assumptions: Mapping[str, Any], robustness: Mapping[str, Any]) -> pd.DataFrame:
    furnace = generator["furnace"]
    heat_energy = furnace["heat_energy"]
    background = generator["background"]
    rows = [
        {"parameter_group": "EAF equipment", "value_or_range": "80 t AC; 80 MW cap; about 5,000 heats/year", "source_class": "B6 same-device literature", "paper_use": "main engineering envelope"},
        {"parameter_group": "Tap-to-tap target", "value_or_range": "60-70 min", "source_class": "B6 same-device literature", "paper_use": "generator calibration gate"},
        {"parameter_group": "Stage power shares", "value_or_range": str(furnace["stages"]["raw_power_ratio"]), "source_class": "author scenario", "paper_use": "central B6-labelled waveform"},
        {"parameter_group": "Heat energy", "value_or_range": f"mean={heat_energy['mean_mwh']}; CV={heat_energy['cv']}; range={heat_energy['lower_mwh']}-{heat_energy['upper_mwh']} MWh", "source_class": "author calibration within B6 cleaning boundary", "paper_use": "central generator"},
        {"parameter_group": "Background load", "value_or_range": f"mean={background['mean_mw']} MW; daily amplitude={background['diurnal_amplitude_mw']} MW", "source_class": "author scenario", "paper_use": "main plus +/-5 MW sensitivity"},
        {"parameter_group": "B4 waveform", "value_or_range": "six normalized duration and energy shares", "source_class": str(robustness["b4_normalized_template"]["source_role"]), "paper_use": "alternative waveform only"},
        {"parameter_group": "B5 variation", "value_or_range": "12% upper within-cluster variation", "source_class": "B5 relative evidence", "paper_use": "engineering sensitivity"},
        {"parameter_group": "Flexibility", "value_or_range": "25% rare increment; 100% theoretical upper", "source_class": "author scenario / Lee-inspired bound", "paper_use": "E10 sensitivity, not field estimate"},
        {"parameter_group": "Korean data", "value_or_range": "10 factories, March-September 2019", "source_class": "published public dataset", "paper_use": "external interface stress test"},
        {"parameter_group": "Synthetic calendar", "value_or_range": f"{assumptions['synthetic_calendar']['days_per_month']} days/month; {assumptions['synthetic_calendar']['operating_pattern']}", "source_class": "author scenario", "paper_use": "simulation calendar"},
    ]
    return pd.DataFrame(rows)


def methods_table() -> pd.DataFrame:
    rows = [
        ("POINT-3M", "training monthly maxima", "training mean", "three-mode choice and D"),
        ("PATENT-HIST-3M", "training monthly maxima", "mean/median/q90/max candidate screen", "deterministic three-mode choice and D"),
        ("TAI-MARGIN", "training monthly maxima", "rolling center plus validated margin", "three-mode choice and D"),
        ("EMP-JOINT", "training monthly maxima", "empirical resampling", "probabilistic three-mode choice and D"),
        ("KDE-MC-JOINT", "training monthly maxima", "boundary-corrected KDE", "probabilistic three-mode choice and D"),
        ("B2-CONTRACT", "process-tail distribution", "gated tail draws", "contract mode forced; D only"),
        ("GEV-JOINT", "training monthly maxima", "GEV with physical truncation gate", "probabilistic three-mode choice and D"),
        ("TAIL-NODECLUSTER", "heat maxima without cross-heat grouping", "POT magnitude plus monthly count", "ablation three-mode choice and D"),
        ("TAIL-JOINT", "process-event maxima and training months", "gated POT/count reconstruction with fallbacks", "probabilistic three-mode choice and D"),
        ("DIST-ORACLE", "independent fixed test pool", "empirical truth distribution", "evaluation reference only"),
    ]
    return pd.DataFrame(rows, columns=["method_id", "information_set", "distribution_rule", "decision_output"])


def _cluster_interval(values: pd.Series) -> tuple[float, float, float]:
    sample = values.dropna().to_numpy(dtype=float)
    mean = float(sample.mean()) if sample.size else float("nan")
    if sample.size < 2:
        return mean, float("nan"), float("nan")
    critical = float(stats.t.ppf(0.975, sample.size - 1))
    halfwidth = critical * float(sample.std(ddof=1) / np.sqrt(sample.size))
    return mean, mean - halfwidth, mean + halfwidth


def primary_results_table(paired: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    required = {"physical_scenario_id", "replication_id", "method_id", "training_months", "test_cost", "regret", "mode_error", "excess_probability", "capacity_fee_monthly"}
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"paired results missing table columns: {missing}")
    frame = paired.copy()
    frame["normalized_regret"] = frame["regret"] / (3.0 * frame["capacity_fee_monthly"])
    clustered = frame.groupby(["method_id", "training_months", "physical_scenario_id", "replication_id"], observed=True).agg(
        mean_test_cost=("test_cost", "mean"),
        normalized_regret=("normalized_regret", "mean"),
        mode_error=("mode_error", "mean"),
        excess_probability=("excess_probability", "mean"),
    ).reset_index()
    rows = []
    for (method_id, training_months), group in clustered.groupby(["method_id", "training_months"], observed=True):
        regret_mean, regret_low, regret_high = _cluster_interval(group["normalized_regret"])
        rows.append(
            {
                "method_id": method_id,
                "training_months": int(training_months),
                "mean_test_cost_cny": float(group["mean_test_cost"].mean()),
                "mean_normalized_regret": regret_mean,
                "regret_ci95_low": regret_low,
                "regret_ci95_high": regret_high,
                "mode_error_rate": float(group["mode_error"].mean()),
                "mean_excess_probability": float(group["excess_probability"].mean()),
                "cluster_count": len(group),
            }
        )
    result = pd.DataFrame(rows)
    order = {method: index for index, method in enumerate(method_order)}
    result["method_order"] = result["method_id"].map(order)
    return result.sort_values(["training_months", "method_order"]).drop(columns="method_order").reset_index(drop=True)


def combined_gates_table(gate_frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for source, frame in gate_frames:
        required = {"gate", "status", "evidence", "mandatory_action"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{source} gate table missing columns: {missing}")
        for row in frame.to_dict("records"):
            rows.append({"source": source, **{key: row[key] for key in ["gate", "status", "evidence", "mandatory_action"]}})
    return pd.DataFrame(rows)
