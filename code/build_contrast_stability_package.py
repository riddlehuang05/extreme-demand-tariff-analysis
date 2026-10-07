#!/usr/bin/env python3
"""Deterministic post-processing for tariff-contrast Wasserstein stability.

Reads only the saved primary oracle table, saved numerical-audit decision and
competitor rows, and existing Figure S3 input tables. It generates no histories
and fits no predictive laws. Run from an extracted Supplementary Source Data
package with: python code/build_contrast_stability_package.py
Dependencies: Python 3.12, numpy, pandas, pyarrow, matplotlib, Pillow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

MODES = ["capacity", "actual_maximum_demand", "contracted_maximum_demand"]
SHORT = dict(zip(MODES, ["cap", "act", "con"]))
RD, KAPPA, RC = 36.0, 2.0, 22.5
METHODS = ["TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"]
COLORS = dict(zip(METHODS, ["#176B87", "#985D88", "#D17828", "#39836C", "#666666"]))
MARKERS = dict(zip(METHODS, ["o", "s", "^", "D", "v"]))
REGIONS = ["actual_maximum_demand", "contracted_maximum_demand", "capacity"]
REGION_LABELS = dict(zip(REGIONS, ["Actual", "Contract", "Capacity"]))


def k_pair(m: str, n: str, rd: float = RD, kappa: float = KAPPA) -> float:
    pair = frozenset((m, n))
    if pair == frozenset(("capacity", "actual_maximum_demand")):
        return rd
    if pair == frozenset(("capacity", "contracted_maximum_demand")):
        return rd * kappa
    if pair == frozenset(("actual_maximum_demand", "contracted_maximum_demand")):
        return rd * max(1.0, abs(1.0 - kappa))
    raise ValueError(f"Invalid distinct tariff pair: {m}, {n}")


def mode_l(m: str, kappa: float = KAPPA) -> float:
    return {"capacity": 0.0, "actual_maximum_demand": RD,
            "contracted_maximum_demand": RD * kappa}[m]


def compute_radii(root: Path, out: Path) -> pd.DataFrame:
    source = root / "data/source/primary/oracle_mode_values_9_cells.csv"
    oracle = pd.read_csv(source, float_precision="round_trip")
    value_cols = {
        "capacity": "oracle_value_capacity_cny",
        "actual_maximum_demand": "oracle_value_actual_maximum_demand_cny",
        "contracted_maximum_demand": "oracle_value_contracted_maximum_demand_cny",
    }
    rows = []
    for r in oracle.itertuples(index=False):
        vals = {m: float(getattr(r, col)) for m, col in value_cols.items()}
        mstar = r.oracle_mode
        assert mstar in MODES
        best = vals[mstar]
        competitors = [m for m in MODES if m != mstar]
        gaps = {m: vals[m] - best for m in competitors}
        assert all(g > 0 for g in gaps.values()), f"Oracle mode not unique in {r.cell_id}"
        gamma = min(gaps.values())
        Lstar = mode_l(mstar)
        pair_vals = {m: gaps[m] / (mode_l(m) + Lstar) for m in competitors}
        contrast_vals = {m: gaps[m] / k_pair(m, mstar) for m in competitors}
        uniform = gamma / (2.0 * RD * max(1.0, KAPPA))
        rp, rc = min(pair_vals.values()), min(contrast_vals.values())
        assert rc + 1e-9 >= rp and rp + 1e-9 >= uniform, r.cell_id
        rows.append({
            "cell_id": r.cell_id, "alpha": r.alpha, "utilization": r.utilization,
            "oracle_mode": SHORT[mstar], "V_capacity_cny": vals["capacity"],
            "V_actual_cny": vals["actual_maximum_demand"],
            "V_contract_cny": vals["contracted_maximum_demand"],
            "V_star_cny": best, "gamma_F_cny": gamma,
            "gap_to_capacity_cny": vals["capacity"] - best,
            "gap_to_actual_cny": vals["actual_maximum_demand"] - best,
            "gap_to_contract_cny": vals["contracted_maximum_demand"] - best,
            "relative_gap_percent": 100.0 * gamma / best,
            "rho_contrast_kw": rc, "rho_modewise_pair_kw": rp,
            "rho_uniform_kw": uniform, "contrast_over_pair": rc / rp,
            "pair_binding_competitor": SHORT[min(pair_vals, key=pair_vals.get)],
            "contrast_binding_competitor": SHORT[min(contrast_vals, key=contrast_vals.get)],
            "radius_order_pass": True,
        })
    result = pd.DataFrame(rows).sort_values(["alpha", "utilization"]).reset_index(drop=True)
    assert len(result) == 9 and result.radius_order_pass.all()
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "primary_tariff_radii.csv", index=False, float_format="%.17g")
    return result


def numerical_audit(root: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dpaths = sorted((root / "data/numerical_audit").glob("decisions_*.parquet"))
    ppaths = sorted((root / "data/numerical_audit").glob("pairwise_*.parquet"))
    assert len(dpaths) == 200 and len(ppaths) == 200
    decisions = pd.concat([pd.read_parquet(p) for p in dpaths], ignore_index=True)
    pairs = pd.concat([pd.read_parquet(p) for p in ppaths], ignore_index=True)
    keys = ["replication_id", "method_id", "cell_id"]
    assert len(decisions) == 450_000 and decisions.replication_id.nunique() == 10_000
    assert len(pairs) == 900_000 and not decisions[keys].duplicated().any()
    assert not pairs[keys + ["competitor_mode"]].duplicated().any()
    pairs["direct_K_cny_per_kw"] = [
        k_pair(m, n) for m, n in zip(pairs.oracle_mode, pairs.competitor_mode)
    ]
    pairs["direct_contrast_bound_cny"] = (
        pairs.direct_K_cny_per_kw * pairs.w1_upper_kw + pairs.solver_allowance_cny
    )
    pairs["direct_contrast_pass"] = (
        pairs.direct_contrast_bound_cny < pairs.oracle_pairwise_gap_cny
    )
    grouped = pairs.groupby(keys, as_index=False, sort=False).agg(
        direct_contrast_pass=("direct_contrast_pass", "all"),
        modewise_pair_pass=("pairwise_w1_condition_pass", "all"),
        competitor_count=("competitor_mode", "size"),
    )
    assert (grouped.competitor_count == 2).all()
    d = decisions.merge(grouped, on=keys, validate="one_to_one")
    assert len(d) == 450_000
    assert d.direct_contrast_pass.sum() == 57_501
    assert d.modewise_pair_pass.sum() == 26_393
    assert d.primitive_numerical_condition_pass.sum() == 26_975
    assert d.uniform_w1_numerical_condition_pass.sum() == 5_329
    assert d.primitive_gap_certificate_distributional_envelope.sum() == 0
    assert ((~d.modewise_pair_pass) | d.direct_contrast_pass).all()
    d["observed_mode_error"] = d.selected_mode != d.oracle_mode
    checks = [
        ("Primitive numerical check", "primitive_numerical_condition_pass"),
        ("Contrast-Wasserstein numerical check", "direct_contrast_pass"),
        ("Mode-wise pairwise-Wasserstein numerical check", "modewise_pair_pass"),
        ("Uniform-Wasserstein numerical check", "uniform_w1_numerical_condition_pass"),
        ("Distributional envelope", "primitive_gap_certificate_distributional_envelope"),
    ]
    rows = []
    for label, col in checks:
        passed = d[col].astype(bool)
        rows.append({
            "condition": label, "total_rows": len(d), "passing_rows": int(passed.sum()),
            "coverage_percent": 100.0 * float(passed.mean()),
            "observed_mode_selection_violations": int((passed & d.observed_mode_error).sum()),
            "solver_zeta_status": "NUMERICAL_BOUND_NOT_INTERVAL_CERTIFIED",
        })
    summary = pd.DataFrame(rows)
    assert summary.loc[summary.condition != "Distributional envelope",
                       "observed_mode_selection_violations"].eq(0).all()
    by_method_cell = d.groupby(["method_id", "cell_id"], as_index=False).agg(
        n=("replication_id", "size"),
        primitive_count=("primitive_numerical_condition_pass", "sum"),
        contrast_w1_count=("direct_contrast_pass", "sum"),
        modewise_pair_w1_count=("modewise_pair_pass", "sum"),
        uniform_w1_count=("uniform_w1_numerical_condition_pass", "sum"),
    )
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "numerical_condition_coverage.csv", index=False, float_format="%.9f")
    by_method_cell.to_csv(out / "coverage_by_method_cell.csv", index=False)
    # Keep a compact competitor-level table; full input rows remain in numerical_audit/.
    comp = pairs.groupby(["oracle_mode", "competitor_mode"], as_index=False).agg(
        rows=("direct_contrast_pass", "size"),
        direct_pass=("direct_contrast_pass", "sum"),
        modewise_pass=("pairwise_w1_condition_pass", "sum"),
    )
    comp.to_csv(out / "competitor_pair_summary.csv", index=False)
    return summary, by_method_cell


class AtomLaw:
    def __init__(self, x, w=None):
        self.x = np.asarray(x, dtype=float)
        self.w = np.ones(len(self.x)) / len(self.x) if w is None else np.asarray(w, dtype=float)
        order = np.argsort(self.x)
        self.x, self.w = self.x[order], self.w[order]
        assert np.isclose(self.w.sum(), 1.0) and np.all(self.w > 0)
        self.knots = np.unique(self.x)

    def cdf(self, t):
        return float(self.w[self.x <= t].sum())

    def mean(self):
        return float(self.x @ self.w)

    def H(self, t):
        return float(np.maximum(self.x - t, 0.0) @ self.w)

    def qset(self, p):
        cw = np.cumsum(self.w)
        j = int(np.searchsorted(cw, p, side="left"))
        right = self.x[j + 1] if j + 1 < len(self.x) and abs(cw[j] - p) < 1e-13 else self.x[j]
        return float(self.x[j]), float(right)


class UniformLaw:
    def __init__(self, a, b):
        self.a, self.b = float(a), float(b)
        self.knots = np.array([a, b], dtype=float)

    def cdf(self, t):
        return float(np.clip((t - self.a) / (self.b - self.a), 0, 1))

    def mean(self):
        return (self.a + self.b) / 2

    def H(self, t):
        if t <= self.a:
            return self.mean() - t
        if t >= self.b:
            return 0.0
        return (self.b - t) ** 2 / (2 * (self.b - self.a))

    def qset(self, p):
        q = self.a + p * (self.b - self.a)
        return q, q


def exact_discrete_w1(f, g):
    knots = np.unique(np.r_[f.knots, g.knots])
    return float(sum((b - a) * abs(f.cdf((a + b) / 2) - g.cdf((a + b) / 2))
                     for a, b in zip(knots[:-1], knots[1:])))


def contract_opt(law, alpha, kappa, lo, hi):
    candidates = np.unique(np.r_[lo, hi, np.clip(law.knots / alpha, lo, hi)])
    costs = np.array([d + kappa * law.H(alpha * d) for d in candidates])
    j = int(np.argmin(costs))
    return float(candidates[j]), float(costs[j])


def run_smoke(out: Path):
    cases = [
        ("two_point", AtomLaw([5, 70], [.6, .4]), AtomLaw([8, 80], [.55, .45])),
        ("atom_at_quantile", AtomLaw([10, 40, 90], [.2, .6, .2]), AtomLaw([12, 45, 85], [.2, .6, .2])),
        ("empirical_samples", AtomLaw(np.linspace(9, 89, 24)), AtomLaw(np.linspace(11, 94, 24))),
        ("continuous_uniform", UniformLaw(5, 85), UniformLaw(12, 95)),
        ("lower_boundary", AtomLaw([2, 7], [.3, .7]), AtomLaw([3, 9], [.4, .6])),
        ("upper_boundary", AtomLaw([100, 160], [.4, .6]), AtomLaw([110, 170], [.5, .5])),
        ("quantile_plateau", AtomLaw([10, 80], [.5, .5]), AtomLaw([20, 70], [.5, .5])),
    ]
    records, radii = [], []
    for name, f, g in cases:
        w1 = exact_discrete_w1(f, g)
        for kappa in (1.0, 2.0, 3.0):
            for alpha in (1.2, 1.5):
                for lo, hi in ((20.0, 50.0), (1.0, 180.0)):
                    _, cf = contract_opt(f, alpha, kappa, lo, hi)
                    _, cg = contract_opt(g, alpha, kappa, lo, hi)
                    vf = np.array([RC * 100.0, RD * f.mean(), RD * cf])
                    vg = np.array([RC * 100.0, RD * g.mean(), RD * cg])
                    for i in range(3):
                        for j in range(i + 1, 3):
                            diff = abs((vg[i] - vg[j]) - (vf[i] - vf[j]))
                            bound = k_pair(MODES[i], MODES[j], RD, kappa) * w1
                            tol = 512 * np.finfo(float).eps * max(1.0, abs(vf).max(), abs(vg).max())
                            records.append({
                                "case": name, "kappa": kappa, "alpha": alpha,
                                "lower": lo, "upper": hi, "pair": f"{SHORT[MODES[i]]}--{SHORT[MODES[j]]}",
                                "W1": w1, "contrast_change": diff, "bound": bound,
                                "floating_tolerance": tol, "pass": diff <= bound + tol,
                            })
                    oracle_i = int(np.argmin(vf))
                    gaps = vf - vf[oracle_i]
                    if min(gaps[j] for j in range(3) if j != oracle_i) > 1e-9:
                        old = min(gaps[j] / (mode_l(MODES[j], kappa) + mode_l(MODES[oracle_i], kappa))
                                  for j in range(3) if j != oracle_i)
                        new = min(gaps[j] / k_pair(MODES[j], MODES[oracle_i], RD, kappa)
                                  for j in range(3) if j != oracle_i)
                        radii.append({"case": name, "kappa": kappa, "alpha": alpha,
                                      "lower": lo, "upper": hi, "rho_modewise": old,
                                      "rho_contrast": new, "pass": new + 1e-12 >= old})
    df, dr = pd.DataFrame(records), pd.DataFrame(radii)
    assert df["pass"].all() and dr["pass"].all()
    assert len(df) == 252 and 80 <= len(dr) <= 84
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "smoke_contrasts.csv", index=False)
    dr.to_csv(out / "smoke_radii.csv", index=False)
    pd.DataFrame([{"status": "PASS", "contrast_checks": len(df),
                   "radius_checks": len(dr), "random_numbers_generated": 0}]).to_csv(
        out / "smoke_summary.csv", index=False)
    return len(df), len(dr)



def run_es_smoke(out: Path):
    cases = [
        ("atom_interior", AtomLaw([10, 30, 90], [.2, .6, .2]), 20.0, 40.0),
        ("lower_outside", AtomLaw([5, 10], [.5, .5]), 20.0, 40.0),
        ("upper_outside", AtomLaw([50, 100], [.5, .5]), 20.0, 40.0),
        ("quantile_plateau_intersects", AtomLaw([10, 50], [.5, .5]), 20.0, 40.0),
        ("atom_at_lower_boundary", AtomLaw([10, 20, 60], [.2, .6, .2]), 20.0, 40.0),
        ("atom_at_upper_boundary", AtomLaw([10, 40, 60], [.2, .6, .2]), 20.0, 40.0),
    ]
    rows = []
    alpha, kappa = 1.0, 2.0
    p = 1.0 - 1.0 / (alpha * kappa)
    for name, law, lo, hi in cases:
        d, phi = contract_opt(law, alpha, kappa, lo, hi)
        t = alpha * d
        psi = t + law.H(t) / (1.0 - p)
        qlo, qhi = law.qset(p)
        intersects = max(qlo, alpha * lo) <= min(qhi, alpha * hi)
        ordinary = qlo + law.H(qlo) / (1.0 - p)
        resid = RD * phi - (RD / alpha) * psi
        ordinary_resid = RD * phi - (RD / alpha) * ordinary
        assert abs(resid) < 1e-9
        if intersects:
            assert abs(ordinary_resid) < 1e-8
        else:
            assert ordinary_resid > 1e-8
        rows.append({
            "case": name, "p_star": p, "quantile_set_lower": qlo,
            "quantile_set_upper": qhi, "feasible_threshold_lower": alpha*lo,
            "feasible_threshold_upper": alpha*hi, "quantile_set_intersects": intersects,
            "selected_contract_level": d, "contract_cost_cny": RD*phi,
            "constrained_es_cost_cny": (RD/alpha)*psi,
            "constrained_identity_residual_cny": resid,
            "ordinary_es_cost_cny": (RD/alpha)*ordinary,
            "ordinary_es_residual_cny": ordinary_resid,
            "interpretation": "ORDINARY_ES_EQUALITY" if intersects else "CONSTRAINED_ES_ONLY",
        })
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "es_atoms_boundary_audit.csv", index=False)
    return rows

def save_figure_s3(root: Path, out: Path, coverage: pd.DataFrame):
    src = root / "data/source"
    w = pd.read_csv(src / "r5_primitive_bounds/w1_summary_by_method.csv")
    primitive = pd.read_csv(src / "r5_primitive_bounds/primitive_summary_by_method_cell_oracle.csv")
    support = pd.read_csv(src / "dgp_external_support/support_five_policy_paired_vs_no_upper_first1000.csv")
    oracle = pd.read_csv(src / "primary/oracle_mode_values_9_cells.csv").sort_values(["alpha", "utilization"])
    cells = oracle.cell_id.tolist()
    labels = [f"{a:.2f}/{u:.3f}" for a, u in zip(oracle.alpha, oracle.utilization)]
    assert len(cells) == 9 and set(w.method_id) == set(METHODS)
    out.mkdir(parents=True, exist_ok=True)
    # Persist plotting inputs alongside the output for transparent Figure S3 provenance.
    w.to_csv(out / "FigureS3_W1_input.csv", index=False)
    primitive.to_csv(out / "FigureS3_primitive_cell_input.csv", index=False)
    support.drop(columns=[c for c in support.columns if "maximum_abs" in c], errors="ignore").to_csv(
        out / "FigureS3_support_input.csv", index=False)
    coverage.to_csv(out / "FigureS3_coverage_input.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "lines.linewidth": 1.15, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "path", "savefig.facecolor": "white",
        "figure.facecolor": "white", "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": True,
    })
    def panel(ax, letter, title):
        ax.grid(axis="y", color="#dddddd", linewidth=.4)
        ax.set_axisbelow(True)
        ax.set_title(f"({letter})  {title}", loc="left", pad=7, fontweight="bold")
    fig, aa = plt.subplots(2, 3, figsize=(7.01, 5.85))
    fig.subplots_adjust(left=.13, right=.98, bottom=.22, top=.94, wspace=.50, hspace=.80)
    a, b, c = aa[0]
    for i, method in enumerate(METHODS):
        row = w[w.method_id == method].iloc[0]
        a.plot([row.cdf_lower_mean_kw / 1000, row.cdf_upper_mean_kw / 1000],
               [4-i, 4-i], color=COLORS[method])
        a.plot(row.w1_mean_kw / 1000, 4-i, marker=MARKERS[method], color=COLORS[method], ms=4)
    a.set_yticks(range(5), METHODS[::-1])
    a.set(xscale="log", xlabel="Mean $W_1$ (MW)")
    panel(a, "a", "Numerical $W_1$ bounds")
    for method in METHODS:
        z = primitive[primitive.method_id == method].set_index("cell_id").loc[cells]
        b.plot(np.arange(9) + (METHODS.index(method)-2)*.09,
               z.numerical_certificate_coverage*100, linestyle="none",
               marker=MARKERS[method], ms=3.1, color=COLORS[method])
    b.set_xticks(range(9), labels, rotation=70, ha="right")
    b.set_ylabel("Coverage (%)")
    panel(b, "b", "Primitive bound")
    plot_rows = coverage.set_index("condition").loc[
        ["Primitive numerical check", "Contrast-Wasserstein numerical check",
         "Mode-wise pairwise-Wasserstein numerical check",
         "Uniform-Wasserstein numerical check", "Distributional envelope"]]
    vals = plot_rows.coverage_percent.to_numpy()
    barlabels = ["Primitive", "Contrast $W_1$", "Mode-wise $W_1$", "Uniform $W_1$", "Envelope"]
    c.barh(range(5), vals, color="#777777", height=.55)
    c.set_yticks([])
    c.set_ylim(4.65, -.65)
    for i, lab in enumerate(barlabels):
        c.text(.01, i-.34, lab, transform=c.get_yaxis_transform(),
               ha="left", va="bottom", fontsize=6.3)
    c.set_xlim(0, max(7.1, 1.3*max(vals)))
    c.set_xlabel("Decision coverage (%)")
    for i, val in enumerate(vals):
        c.text(val+.12, i, f"{val:.2f}%", va="center", fontsize=6.8)
    panel(c, "c", "Coverage")

    policies = ["legacy_oracle_B", "training_B_c10", "training_B_c20", "training_B_c50"]
    pc, pm = ["#333333", "#D17828", "#39836C", "#985D88"], ["s", "^", "D", "o"]
    for j, method in enumerate(["TAIL", "GEV", "TAIL"]):
        ax = aa[1, j]
        for k, policy in enumerate(policies):
            z = support[(support.method_id == method) & (support.comparison_policy == policy)].set_index("cell_id").loc[cells]
            metric = "mode_changed" if j == 2 else "regret_cny_per_month"
            scale = .01 if j == 2 else 1000
            ax.errorbar(np.arange(9)+(k-1.5)*.14,
                        z[metric+"_paired_mean_difference"]/scale,
                        yerr=z[metric+"_paired_se"]/scale, fmt=pm[k], color=pc[k],
                        ms=2.8, capsize=1.3, lw=.7)
        ax.axhline(0, lw=.7, color="#555")
        ax.set_xticks(range(9), labels, rotation=70, ha="right")
        ax.set_ylabel("Δ regret (10³ CNY/month)" if j == 0 else ("Changed modes (%)" if j == 2 else None))
        panel(ax, "def"[j], f"{method}: support sensitivity" if j < 2 else "Mode switches")
    fig.legend(handles=[Line2D([], [], color=col, marker=mr, label=lab, ms=3, lw=0)
                        for col, mr, lab in zip(pc, pm, ["Oracle reference", "Training B × 10",
                                                          "Training B × 20", "Training B × 50"])],
               loc="lower center", bbox_to_anchor=(.52, .012), ncol=4, frameon=False, columnspacing=1.)
    base = out / "FigureS3"
    for ext in ("pdf", "svg", "eps", "png"):
        fig.savefig(base.with_suffix("."+ext), format=ext, dpi=600 if ext == "png" else 160)
    with Image.open(base.with_suffix(".png")) as im:
        im.convert("RGB").save(base.with_suffix(".tiff"), format="TIFF",
                               compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="extracted source-data package root")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "data/derived_publication/tariff_contrast_wasserstein"
    radii = compute_radii(root, out)
    coverage, by_method_cell = numerical_audit(root, out)
    smoke_n, radius_n = run_smoke(out)
    es_rows = run_es_smoke(out)
    save_figure_s3(root, out, coverage)
    input_paths = [
        root / "data/source/primary/oracle_mode_values_9_cells.csv",
        root / "data/source/r5_primitive_bounds/w1_summary_by_method.csv",
        root / "data/source/r5_primitive_bounds/primitive_summary_by_method_cell_oracle.csv",
        root / "data/source/dgp_external_support/support_five_policy_paired_vs_no_upper_first1000.csv",
    ]
    input_paths += sorted((root / "data/numerical_audit").glob("decisions_*.parquet"))
    input_paths += sorted((root / "data/numerical_audit").glob("pairwise_*.parquet"))
    input_manifest = []
    for src in input_paths:
        if not src.is_file():
            raise FileNotFoundError(src)
        input_manifest.append({
            "source_path": src.relative_to(root).as_posix(),
            "size_bytes": src.stat().st_size,
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        })
    pd.DataFrame(input_manifest).to_csv(out / "source_input_hashes.csv", index=False)
    checks = {
        "status": "PASS", "primary_decisions": 450000, "histories": 10000,
        "methods": 5, "primary_cells": 9,
        "radii_order": "rho_contrast >= rho_modewise_pair >= rho_uniform; 9/9",
        "coverage": coverage.to_dict(orient="records"),
        "smoke": {"status": "PASS", "contrast_checks": smoke_n,
                  "radius_checks": radius_n, "random_numbers_generated": 0},
        "ES_audit": {"cases": len(es_rows), "identity_pass": True,
                     "ordinary_ES_equalities": sum(x["interpretation"] == "ORDINARY_ES_EQUALITY" for x in es_rows)},
        "solver_zeta_status": "NUMERICAL_BOUND_NOT_INTERVAL_CERTIFIED",
        "histories_generated": 0, "predictive_laws_refit": 0,
        "primary_decisions_rerun": 0,
    }
    (out / "audit_summary.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    manifest = {}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "file_hashes.json":
            manifest[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "file_hashes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(out),
                      "direct_coverage": 57501, "modewise_coverage": 26393,
                      "smoke": [smoke_n, radius_n]}, indent=2))


if __name__ == "__main__":
    main()

