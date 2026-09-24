from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import integrate


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_revision_experiment as experiment  # noqa: E402


def stable_count_max_survival(distribution: experiment.TailProcessDistribution, survival: float) -> float:
    survival = float(survival)
    if distribution.count_model == "poisson":
        return float(-math.expm1(-distribution.count_mean * survival))
    assert distribution.nb_size is not None and distribution.nb_probability is not None
    p = distribution.nb_probability
    log_pgf = -distribution.nb_size * math.log1p(((1.0 - p) / p) * survival)
    return float(-math.expm1(log_pgf))


def transformed_tail_integral(
    distribution: experiment.TailProcessDistribution, survival_max: float
) -> float:
    xi = distribution.xi
    if survival_max <= 0.0 or distribution.count_mean <= 0.0:
        return 0.0
    power = 1.0 / (1.0 - xi)
    factor = survival_max ** (1.0 - xi) * power

    nodes, weights = np.polynomial.legendre.leggauss(256)
    units = 0.5 * (nodes + 1.0)
    values = []
    for unit in units:
        survival = survival_max * float(unit) ** power
        if survival <= np.finfo(float).tiny:
            ratio = distribution.count_mean
        else:
            ratio = stable_count_max_survival(distribution, survival) / survival
        values.append(factor * ratio)
    return float(distribution.sigma * 0.5 * np.dot(weights, values))


def rebuild_tail(rate: float, rate_index: int, replication: int):
    settings = experiment.load_settings()
    dgp = experiment.dgp_from_settings(settings, rate)
    oracle = experiment.OracleDistribution(dgp)
    seed_root = int(settings["seed_root"])
    history_rng = experiment.seed_rng(seed_root, 3, rate_index, replication, 10)
    monthly, events = experiment.generate_event_history(
        int(settings["training_months"]), dgp, history_rng
    )
    return experiment.fit_tail_exact(
        monthly,
        events,
        settings,
        "frequency",
        experiment.seed_rng(seed_root, 3, rate_index, replication, 20, 1),
        experiment.seed_rng(seed_root, 3, rate_index, replication, 20, 2),
        10.0 * oracle.ppf(0.999),
    )


def main() -> None:
    source = ROOT / "outputs" / "frequency" / "revision_results.csv"
    frame = pd.read_csv(source)
    fits = frame.query("method_id == 'TAIL'").drop_duplicates(
        ["tail_rate_per_month", "replication_id"]
    )
    unstable = fits[
        fits["xi_hat"].notna()
        & (fits["xi_hat"] >= 0.5)
        & ~fits["fit_status"].str.startswith("FALLBACK")
    ].copy()
    rates = [float(value) for value in experiment.load_settings()["frequency_rates"]]
    rows = []
    for record in unstable.itertuples():
        rate = float(record.tail_rate_per_month)
        rate_index = rates.index(rate)
        distribution, diagnostics = rebuild_tail(rate, rate_index, int(record.replication_id))
        if not isinstance(distribution, experiment.TailProcessDistribution):
            raise RuntimeError("reconstructed fit status differs from stored primary TAIL fit")
        for survival_max in (1.0, 0.5, 0.1, 0.01, 0.0001):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                old = distribution._tail_integral(survival_max)
            new = transformed_tail_integral(distribution, survival_max)
            rows.append(
                {
                    "tail_rate_per_month": rate,
                    "replication_id": int(record.replication_id),
                    "fit_status": diagnostics["fit_status"],
                    "xi_hat": distribution.xi,
                    "count_model": distribution.count_model,
                    "survival_upper_limit": survival_max,
                    "original_tail_integral": old,
                    "transformed_tail_integral": new,
                    "absolute_difference": abs(old - new),
                    "relative_difference": abs(old - new) / max(abs(new), np.finfo(float).eps),
                    "original_integration_warning": any(
                        issubclass(item.category, integrate.IntegrationWarning) for item in caught
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    output = ROOT / "tables" / "tail_quadrature_stability_audit.csv"
    audit.to_csv(output, index=False)
    report = {
        "audited_primary_tail_fits_with_xi_at_least_0_5": int(len(unstable)),
        "audited_integral_evaluations": int(len(audit)),
        "original_integration_warnings": int(audit.get("original_integration_warning", pd.Series(dtype=bool)).sum()),
        "maximum_relative_difference": float(audit["relative_difference"].max()) if len(audit) else 0.0,
        "maximum_absolute_difference": float(audit["absolute_difference"].max()) if len(audit) else 0.0,
        "status": "PASS" if len(audit) == 0 or audit["relative_difference"].max() < 1e-6 else "REVIEW",
    }
    report_path = ROOT / "reports" / "tail_quadrature_stability_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
