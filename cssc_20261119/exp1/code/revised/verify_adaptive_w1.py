"""Cross-check archived quadrature outliers and the longest GEV cutoff in x-space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import compute_adaptive_w1 as adaptive
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import beta, genextreme, genpareto


def verification_interval(row) -> tuple[float, float]:
    start = 25 * (row.replication_id // 25)
    path = adaptive.FIT_ROOT / f"rep_{start:05d}_{start+24:05d}_fits.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    law, = [f["law"] for f in records
            if (f["replication_id"], f["method_id"]) == (row.replication_id, row.method_id)]
    if row.method_id == "GEV":
        gev = genextreme(law["shape_c"], loc=law["loc_kw"], scale=law["scale_kw"])
        weight = gev.sf(0)

        def fit_survival(x):
            return gev.sf(x) / weight
    elif row.method_id == "KDE":
        sample = np.asarray(law["sample_kw"])
        bandwidth = law["bandwidth_kw"]
        weight = ndtr(sample / bandwidth).sum()

        def fit_survival(x):
            return ndtr((sample - x) / bandwidth).sum() / weight
    else:
        raise ValueError(f"Independent verification formula missing for {row.method_id}")

    dgp = adaptive.original._ORACLE.dgp
    tail = genpareto(dgp.xi, loc=dgp.threshold_kw, scale=dgp.sigma_kw)

    def oracle_survival(x):
        if x >= dgp.threshold_kw:
            return -np.expm1(-dgp.tail_rate * tail.sf(x))
        b = (x - dgp.baseline_lower_kw) / (dgp.threshold_kw - dgp.baseline_lower_kw)
        return 1.0 - np.exp(-dgp.tail_rate) * beta.cdf(b, dgp.beta_a, dgp.beta_b) ** dgp.baseline_events

    scale = dgp.baseline_lower_kw

    def integrand(log_coordinate):
        x = scale * np.expm1(log_coordinate)
        return abs(fit_survival(x) - oracle_survival(x)) * scale * np.exp(log_coordinate)

    edges = np.linspace(0.0, np.log1p(row.adaptive_cutoff_kw / scale), 61)
    integrals = [quad(integrand, l, h, epsabs=1e-7, epsrel=1e-10) for l, h in zip(edges[:-1], edges[1:])]
    finite_quad, quad_error = np.sum(integrals, axis=0)
    return float(finite_quad), float(quad_error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    adaptive.init_worker()
    laws = pd.read_parquet(args.result / "adaptive_w1_by_law.parquet")
    outside = (laws.w1_kw < laws.adaptive_lower_kw) | (laws.w1_kw > laws.adaptive_upper_kw)
    cases = pd.concat([laws.loc[outside], laws[laws.method_id.eq("GEV")].nlargest(1, "adaptive_cutoff_kw")])
    cases = cases.drop_duplicates(["replication_id", "method_id"])
    records = []
    for row in cases.itertuples(index=False):
        estimate, quad_error = verification_interval(row)
        lower, upper = estimate - quad_error, estimate + quad_error + row.adaptive_tail_kw
        inside = row.adaptive_lower_kw <= lower <= upper <= row.adaptive_upper_kw
        records.append({"replication_id": row.replication_id, "method_id": row.method_id,
                        "archived_quadrature_kw": row.w1_kw, "finite_x_space_quadrature_kw": estimate,
                        "reported_x_space_quadrature_error_kw": quad_error,
                        "tail_upper_kw": row.adaptive_tail_kw, "adaptive_lower_kw": row.adaptive_lower_kw,
                        "adaptive_upper_kw": row.adaptive_upper_kw, "x_space_estimate_and_tail_inside": inside})
    report = pd.DataFrame(records)
    report.to_csv(args.result / "adaptive_w1_independent_quadrature.csv", index=False)
    print(report.to_string(index=False))
    if not report.x_space_estimate_and_tail_inside.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
