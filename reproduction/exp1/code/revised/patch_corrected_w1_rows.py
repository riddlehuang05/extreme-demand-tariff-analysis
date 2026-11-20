"""Update R5 numerical-audit rows for the corrected primary TAIL history."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODES = ["capacity", "actual_maximum_demand", "contracted_maximum_demand"]
KEY = ["replication_id", "method_id", "cell_id"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primitive", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--base-repo", type=Path, required=True)
    parser.add_argument("--primitive-script", type=Path, required=True)
    parser.add_argument("--law-script", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("cssc_primitive_patch", args.primitive_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import primitive-bounds module")
    primitive_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = primitive_module
    spec.loader.exec_module(primitive_module)
    primitive_module.init_worker(str(args.base_repo), str(args.law_script), 128, 512, 64)
    fit = [json.loads(x) for x in args.fits.read_text(encoding="utf-8").splitlines() if x]
    tail = next(x for x in fit if x["method_id"] == "TAIL" and x["replication_id"] == 8185)
    law = primitive_module.reconstruct_law(primitive_module._ENGINE, tail["law"])
    primary = pd.read_csv(args.primary)
    primary = primary[(primary.replication_id == 8185) & (primary.method_id == "TAIL")].sort_values("cell_id")
    r4 = pd.read_parquet(args.primitive)
    r4 = r4[(r4.replication_id == 8185) & (r4.method_id == "TAIL")].set_index(KEY).loc[
        pd.MultiIndex.from_frame(primary[KEY])].reset_index()
    if len(primary) != 9 or len(r4) != 9:
        raise AssertionError("expected nine corrected TAIL cells")
    audit_dir = args.source_data / "data" / "numerical_audit"
    dec_path = audit_dir / "decisions_08150.parquet"
    pair_path = audit_dir / "pairwise_08150.parquet"
    dec = pd.read_parquet(dec_path)
    pairs = pd.read_parquet(pair_path)
    old_dec = dec[(dec.replication_id == 8185) & (dec.method_id == "TAIL")].sort_values("cell_id")
    old_pairs = pairs[(pairs.replication_id == 8185) & (pairs.method_id == "TAIL")]
    if len(old_dec) != 9 or len(old_pairs) != 18:
        raise AssertionError("archived numerical-audit rows are incomplete")
    oracle_path = args.source_data / "data" / "source" / "primary" / "oracle_mode_values_9_cells.csv"
    oracle = pd.read_csv(oracle_path).set_index("cell_id")
    rate = 36.0
    cap_rate = 22.5
    amended = []
    new_pairs = []
    for source_row, bounds_row in zip(primary.itertuples(index=False), r4.itertuples(index=False)):
        cell = source_row.cell_id
        alpha = float(source_row.alpha)
        kappa = float(source_row.kappa)
        s = float(source_row.S_kva)
        d = float(source_row.estimated_contract_D_kw)
        lo, hi = 0.4 * s, s
        x = alpha * d
        left_cdf = primitive_module.distribution_cdf_left(primitive_module._ENGINE, law, x)
        right_cdf = float(primitive_module.distribution_cdf_vec(
            primitive_module._ENGINE, law, np.array([x]))[0])
        slope_left = rate * (1 + kappa * alpha * (left_cdf - 1))
        slope_right = rate * (1 + kappa * alpha * (right_cdf - 1))
        subgradient = slope_left if slope_left > 0 else slope_right if slope_right < 0 else 0.0
        gap = max(0.0, subgradient * (d - lo) if subgradient >= 0 else -subgradient * (hi - d))
        zeta_round = 512 * np.finfo(float).eps * max(1, rate * (d + kappa * alpha * d))
        zcon = gap + zeta_round
        reconstructed = np.array([
            cap_rate * s, rate * float(law.mean()),
            rate * (d + kappa * float(primitive_module.distribution_stoploss_vec(
                primitive_module._ENGINE, law, np.array([x]))[0])),
        ])
        saved = np.array([source_row.estimated_mode_cost_capacity,
                          source_row.estimated_mode_cost_actual,
                          source_row.estimated_mode_cost_contract], dtype=float)
        eta = abs(reconstructed - saved) + 512 * np.finfo(float).eps * np.maximum(
            1, np.maximum(abs(reconstructed), abs(saved)))
        selected_index = MODES.index(source_row.selected_mode)
        theta = max(0.0, saved[selected_index] - saved.min())
        zfull = zcon + eta[selected_index] + eta.max() + theta
        r = dict(zip(r4.columns, bounds_row))
        r["solver_zeta_con_numerical_cny"] = zcon
        r["solver_zeta_full_numerical_cny"] = zfull
        r["objective_evaluation_allowance_selected_cny"] = eta[selected_index]
        r["objective_evaluation_allowance_max_cny"] = eta.max()
        r["saved_candidate_selection_gap_cny"] = theta
        r["solver_zeta_status"] = "NUMERICAL_BOUND_NOT_INTERVAL_CERTIFIED"
        r["primitive_epsilon_numerical_upper_cny"] = r["primitive_epsilon_cny"]
        r["old_pseudo_pairwise_condition_pass"] = r["w1_certificate_pairwise_upper"]
        W = float(r["w1_cdf_riemann_upper_kw"])
        values = np.array([float(oracle.loc[cell, f"oracle_value_{mode}_cny"]) for mode in MODES])
        oracle_index = MODES.index(source_row.oracle_mode)
        best = values[oracle_index]
        L = np.array([0.0, rate, rate * kappa])
        all_pair = True
        radius = math.inf
        for j, mode in enumerate(MODES):
            if j == oracle_index:
                continue
            oracle_gap = values[j] - best
            modewise_bound = (L[oracle_index] + L[j]) * W + zfull
            pair_ok = modewise_bound < oracle_gap
            all_pair &= pair_ok
            radius = min(radius, oracle_gap / (L[oracle_index] + L[j]))
            new_pairs.append({
                "replication_id": 8185, "method_id": "TAIL", "cell_id": cell,
                "oracle_mode": source_row.oracle_mode, "competitor_mode": mode,
                "oracle_pairwise_gap_cny": oracle_gap,
                "L_oracle": L[oracle_index], "L_competitor": L[j],
                "w1_upper_kw": W, "solver_allowance_cny": zfull,
                "pairwise_w1_bound_cny": modewise_bound,
                "pairwise_w1_condition_pass": pair_ok,
                "solver_allowance_policy": "common_zeta_full_numerical_for_all_competitors",
                "solver_zeta_status": "NUMERICAL_BOUND_NOT_INTERVAL_CERTIFIED",
            })
        gamma = float(source_row.oracle_margin_cny)
        uniform_L = rate * max(1, kappa)
        pseudo = (L[oracle_index] + max(L[j] for j in range(3) if j != oracle_index)) * W + zfull
        r["pseudo_pairwise_same_zeta_full_condition_pass"] = pseudo < gamma
        r["pairwise_w1_numerical_condition_pass"] = all_pair
        r["pairwise_w1_theoretical_radius_kw"] = radius
        r["uniform_w1_theoretical_radius_kw"] = gamma / (2 * uniform_L)
        r["uniform_w1_numerical_cost_bound_cny"] = 2 * uniform_L * W + zfull
        r["uniform_w1_numerical_condition_pass"] = r["uniform_w1_numerical_cost_bound_cny"] < gamma
        r["primitive_numerical_condition_pass"] = 2 * r["primitive_epsilon_numerical_upper_cny"] + zfull < gamma
        r["primitive_numerical_has_directed_rounding"] = False
        r["primitive_numerical_is_exact_supremum"] = False
        if r["pairwise_w1_numerical_condition_pass"] and not source_row.action_correct:
            raise AssertionError("numerical certificate violated")
        amended.append({column: r[column] for column in dec.columns})
    new_dec = pd.DataFrame(amended)
    new_pair = pd.DataFrame(new_pairs)[pairs.columns]
    if new_dec.isna().any().any() or new_pair.isna().any().any():
        print("nullable decision fields:", new_dec.columns[new_dec.isna().any()].tolist())
        print("nullable pair fields:", new_pair.columns[new_pair.isna().any()].tolist())
    dec = dec.set_index(KEY)
    new_dec = new_dec.set_index(KEY)
    dec.loc[new_dec.index, :] = new_dec[dec.columns].to_numpy()
    pairs = pairs.set_index(KEY + ["competitor_mode"])
    new_pair = new_pair.set_index(KEY + ["competitor_mode"])
    pairs.loc[new_pair.index, :] = new_pair[pairs.columns].to_numpy()
    dec = dec.reset_index()
    pairs = pairs.reset_index()
    dec.to_parquet(dec_path, index=False)
    pairs.to_parquet(pair_path, index=False)
    report = {
        "status": "PASS", "corrected_history": 8185, "corrected_decision_rows": 9,
        "corrected_pair_rows": 18, "old_decision_sha256": sha(args.primitive),
        "patched_decision_chunk_sha256": sha(dec_path),
        "patched_pair_chunk_sha256": sha(pair_path),
        "old_direct_count": None,
        "new_modewise_count": int(new_dec.pairwise_w1_numerical_condition_pass.sum()),
        "new_uniform_count": int(new_dec.uniform_w1_numerical_condition_pass.sum()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
