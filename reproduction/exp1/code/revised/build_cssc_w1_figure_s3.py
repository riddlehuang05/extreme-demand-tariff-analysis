"""Rebuild the numerical-bound summary and Figure S3 from revised fitted laws."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODES = ("capacity", "actual_maximum_demand", "contracted_maximum_demand")
METHODS = ("TAIL", "EVENT-EMP", "GEV", "KDE", "EMP")
COLORS = dict(zip(METHODS, ("#176B87", "#985D88", "#D17828", "#39836C", "#666666")))
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D", "v")))
RATE, KAPPA = 36.0, 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_k(first: str, second: str) -> float:
    pair = frozenset((first, second))
    if pair == frozenset(("capacity", "actual_maximum_demand")):
        return RATE
    if pair == frozenset(("capacity", "contracted_maximum_demand")):
        return RATE * KAPPA
    if pair == frozenset(("actual_maximum_demand", "contracted_maximum_demand")):
        return RATE * max(1.0, abs(1.0 - KAPPA))
    raise ValueError(f"Unknown tariff-mode pair: {first}, {second}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    args = parser.parse_args()
    bounds = args.bounds.resolve()
    source = args.source_data.resolve()
    manifest_path = bounds / "primitive_bounds_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "PASS" or manifest.get("max_replications") != 10000
            or len(manifest.get("tasks", [])) != 400):
        raise ValueError("Full revised numerical-bound run is incomplete")
    summary = json.loads((bounds / "primitive_bounds_summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "PASS" or summary.get("n_history_laws") != 50000
            or summary.get("n_decision_rows") != 450000):
        raise ValueError("Revised numerical-bound summary is incomplete")
    for name, expected in manifest.get("summary_files_sha256", {}).items():
        if sha256(bounds / name) != expected:
            raise ValueError(f"Numerical-bound summary hash mismatch: {name}")
    w1 = pd.read_csv(bounds / "w1_summary_by_method.csv")
    primitive = pd.read_csv(bounds / "primitive_summary_by_method_cell_oracle.csv")
    oracle = pd.read_csv(source / "data/source/primary/oracle_mode_values_9_cells.csv")
    if (set(w1.method_id) != set(METHODS) or not w1.n_histories.eq(10000).all()
            or len(primitive) != 45 or not primitive.n.eq(10000).all()
            or len(oracle) != 9):
        raise ValueError("Numerical-bound plotting summaries lack required rows")
    columns = ["replication_id", "method_id", "cell_id", "oracle_mode", "selected_mode",
               "w1_cdf_riemann_upper_kw", "solver_zeta_bound_cny",
               "primitive_gap_certificate_numerical", "w1_certificate_pairwise_upper",
               "w1_certificate_uniform_upper", "primitive_gap_certificate_distributional_envelope"]
    parts = []
    for task in manifest["tasks"]:
        path = Path(task["decision_path"])
        if not path.is_file() or sha256(path) != task["decision_sha256"]:
            raise ValueError(f"Numerical-bound chunk hash mismatch: {path}")
        parts.append(pd.read_parquet(path, columns=columns))
    decisions = pd.concat(parts, ignore_index=True)
    if (len(decisions) != 450000 or decisions.replication_id.nunique() != 10000
            or decisions.duplicated(["replication_id", "method_id", "cell_id"]).any()):
        raise ValueError("Numerical-bound decision keys are incomplete")
    direct = np.zeros(len(decisions), dtype=bool)
    competitor_rows = []
    value_columns = {mode: "oracle_value_" + mode + "_cny" for mode in MODES}
    for cell in oracle.itertuples(index=False):
        mask = decisions.cell_id.eq(cell.cell_id).to_numpy()
        selected = decisions.loc[mask]
        if not selected.oracle_mode.eq(cell.oracle_mode).all():
            raise ValueError(f"Oracle mode mismatch in cell {cell.cell_id}")
        best = float(getattr(cell, value_columns[cell.oracle_mode]))
        all_pass = np.ones(len(selected), dtype=bool)
        for other in MODES:
            if other == cell.oracle_mode:
                continue
            gap = float(getattr(cell, value_columns[other])) - best
            if gap <= 0:
                raise ValueError(f"Nonpositive oracle contrast gap in {cell.cell_id}")
            radius = (pair_k(cell.oracle_mode, other)
                      * selected.w1_cdf_riemann_upper_kw.to_numpy(float)
                      + selected.solver_zeta_bound_cny.to_numpy(float))
            passed = radius < gap
            all_pass &= passed
            competitor_rows.append({"cell_id": cell.cell_id, "oracle_mode": cell.oracle_mode,
                                    "competitor_mode": other, "rows": len(selected),
                                    "direct_pass": int(passed.sum())})
        direct[mask] = all_pass
    decisions["direct_contrast_pass"] = direct
    flags = [
        ("Primitive numerical check", "primitive_gap_certificate_numerical"),
        ("Contrast-Wasserstein numerical check", "direct_contrast_pass"),
        ("Mode-wise pairwise-Wasserstein numerical check", "w1_certificate_pairwise_upper"),
        ("Uniform-Wasserstein numerical check", "w1_certificate_uniform_upper"),
        ("Distributional envelope", "primitive_gap_certificate_distributional_envelope"),
    ]
    coverage_rows = []
    mode_error = decisions.selected_mode.ne(decisions.oracle_mode).to_numpy()
    for label, key in flags:
        passed = decisions[key].to_numpy(bool)
        violations = int(np.sum(passed & mode_error))
        if violations:
            raise ValueError(f"Observed mode error under {label}: {violations}")
        coverage_rows.append({"condition": label, "total_rows": len(decisions),
                              "passing_rows": int(passed.sum()),
                              "coverage_percent": 100.0 * passed.mean(),
                              "observed_mode_selection_violations": violations})
    if not np.all(~decisions.w1_certificate_pairwise_upper.to_numpy(bool) | direct):
        raise ValueError("Direct contrast condition is weaker than pairwise condition")
    out = source / "data/derived_publication/tariff_contrast_wasserstein"
    out.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out / "numerical_condition_coverage.csv", index=False)
    pd.DataFrame(competitor_rows).to_csv(out / "competitor_pair_summary.csv", index=False)
    w1.to_csv(out / "FigureS3_W1_input.csv", index=False)
    primitive.to_csv(out / "FigureS3_primitive_cell_input.csv", index=False)
    coverage.to_csv(out / "FigureS3_coverage_input.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
                         "axes.titlesize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.1))
    fig.subplots_adjust(left=.08, right=.98, bottom=.29, top=.86, wspace=.40)
    for i, method in enumerate(METHODS):
        row = w1[w1.method_id.eq(method)].iloc[0]
        axes[0].plot([row.cdf_lower_mean_kw/1000, row.cdf_upper_mean_kw/1000],
                     [4-i, 4-i], color=COLORS[method])
        axes[0].plot(row.w1_mean_kw/1000, 4-i, marker=MARKERS[method],
                     color=COLORS[method], markersize=4)
    axes[0].set_yticks(range(5), METHODS[::-1])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Mean $W_1$ (MW)")
    axes[0].set_title("(a)  Numerical $W_1$ bounds", loc="left")
    oracle = oracle.sort_values(["alpha", "utilization"])
    for method in METHODS:
        subset = primitive[primitive.method_id.eq(method)].set_index("cell_id").loc[oracle.cell_id]
        axes[1].plot(np.arange(9) + (METHODS.index(method)-2)*.09,
                     subset.numerical_certificate_coverage.to_numpy(float)*100,
                     linestyle="none", marker=MARKERS[method], markersize=3.1,
                     color=COLORS[method])
    axes[1].set_xticks(range(9), [f"{a:.2f}/{u:.3f}" for a, u in zip(oracle.alpha, oracle.utilization)],
                      rotation=65, ha="right")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_title("(b)  Primitive bound", loc="left")
    values = coverage.coverage_percent.to_numpy(float)
    labels = ["Primitive", "Contrast $W_1$", "Mode-wise $W_1$", "Uniform $W_1$", "Envelope"]
    axes[2].barh(range(5), values, color="#777777", height=.58)
    axes[2].set_yticks(range(5), labels, fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, max(7.1, 1.25*float(values.max())))
    axes[2].set_xlabel("Decision coverage (%)")
    axes[2].set_title("(c)  Coverage", loc="left")
    for i, value in enumerate(values):
        axes[2].text(value + .08, i, f"{value:.2f}%", va="center", fontsize=7)
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=.4)
        axis.set_axisbelow(True)
    pdf = out / "FigureS3.pdf"
    fig.savefig(pdf)
    fig.savefig(out / "FigureS3.svg")
    fig.savefig(out / "FigureS3.png", dpi=500)
    plt.close(fig)
    target = args.supplement / "figures/FigureS3.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf.read_bytes())
    audit = {"status": "PASS", "decision_rows": len(decisions),
             "source_manifest_sha256": sha256(manifest_path),
             "source_summary_sha256": sha256(bounds / "primitive_bounds_summary.json"),
             "coverage": coverage_rows, "figure_pdf_sha256": sha256(pdf)}
    (out / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
