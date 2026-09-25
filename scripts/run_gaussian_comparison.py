"""A nonnegative Gaussian reference on the primary 1,000 training histories."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import run_revision_experiment as primary


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/extensions"
BOOTSTRAP_SEED = 2026092502
GRID_POINTS = 10001


class NonnegativeGaussian:
    """Fit a Gaussian by moments, then condition its predictive law on M >= 0."""

    def __init__(self, sample):
        values = np.asarray(sample, dtype=float)
        if values.size < 2 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Expected a finite, nonnegative sample")
        self.loc = float(values.mean())
        self.scale = float(values.std(ddof=0))
        if self.scale <= 0:
            raise ValueError("Degenerate Gaussian fit")
        self.normalizer = float(special.ndtr(self.loc / self.scale))

    def cdf(self, x):
        return np.clip((special.ndtr((np.asarray(x) - self.loc) / self.scale)
                        - special.ndtr(-self.loc / self.scale)) / self.normalizer, 0, 1)

    def ppf(self, probability):
        return float(self.loc + self.scale * special.ndtri(
            special.ndtr(-self.loc / self.scale) + probability * self.normalizer))

    def mean(self):
        return float(self.stop_loss(0.0))

    def stop_loss(self, threshold):
        t = np.asarray(threshold)
        z = (t - self.loc) / self.scale
        return ((self.loc - t) * special.ndtr(-z)
                + self.scale * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)) / self.normalizer


def check_solver(distribution, cell, label, replication):
    policy, capacity = cell["policy"], cell["S_kva"]
    lower, upper = policy.delta * capacity, policy.contract_upper_ratio * capacity
    chosen = primary.optimized_decision(distribution, capacity, policy)
    d = chosen["contract_D_kw"]
    objective = lambda value: policy.demand_rate * (value + policy.kappa * float(
        distribution.stop_loss(policy.alpha * value)))
    grid = np.linspace(lower, upper, GRID_POINTS)
    if isinstance(distribution, NonnegativeGaussian):
        costs = policy.demand_rate * (grid + policy.kappa * distribution.stop_loss(policy.alpha * grid))
    elif isinstance(distribution, primary.DiscreteDistribution):
        losses = np.maximum(distribution.support[None, :] - policy.alpha * grid[:, None], 0)
        costs = policy.demand_rate * (grid + policy.kappa * (losses @ distribution.probabilities))
    else:
        grid = np.linspace(lower, upper, 101)
        costs = np.array([objective(value) for value in grid])
    solution = optimize.minimize_scalar(objective, bounds=(lower, upper), method="bounded",
                                        options={"xatol": 1e-5})
    independent = min(objective(lower), objective(upper), float(solution.fun))
    if isinstance(distribution, primary.DiscreteDistribution):
        knots = np.clip(distribution.support / policy.alpha, lower, upper)
        independent = min(independent, min(objective(value) for value in knots))
    chosen_cost = objective(d)
    excess = max(0.0, chosen_cost - min(independent, float(costs.min())))
    if excess > 1e-5 * max(1.0, policy.demand_rate):
        raise AssertionError((label, replication, cell["cell_id"], excess))
    return dict(distribution=label, replication_id=replication, cell_id=cell["cell_id"],
                solution_location="lower" if np.isclose(d, lower) else "upper" if np.isclose(d, upper) else "interior",
                candidate_cost_cny=chosen_cost, independent_cost_cny=independent,
                grid_cost_cny=float(costs.min()), grid_points=len(grid),
                excess_cost_cny=excess, grid_excess_cny=float(costs.min()) - chosen_cost)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = primary.load_settings()
    dgp = primary.dgp_from_settings(settings)
    oracle = primary.OracleDistribution(dgp)
    cells = primary.policies_and_cells(settings, oracle, False)
    source = pd.read_csv(ROOT / "data/derived/primary_rows.csv.gz")
    reference = source[source.method_id == "EMP"].set_index(["replication_id", "cell_id"])
    records, validations, worked = [], [], []
    for replication in range(1000):
        monthly, _ = primary.generate_event_history(24, dgp, primary.seed_rng(
            settings["seed_root"], 2, 0, replication, 10))
        sample = monthly.M_fixed15_kw.to_numpy()
        empirical = primary.DiscreteDistribution(sample)
        gaussian = NonnegativeGaussian(sample)
        for cell in cells:
            policy, capacity, truth = cell["policy"], cell["S_kva"], cell["oracle"]
            ref = reference.loc[(replication, cell["cell_id"])]
            empirical_action = primary.optimized_decision(empirical, capacity, policy)
            empirical_regret = max(0, primary.true_cost_of_estimated_action(
                empirical_action["selected_mode"], empirical_action["contract_D_kw"], oracle,
                capacity, policy) - truth["best_value_cny"])
            if not np.isclose(sample.mean() / oracle.mean() - 1, ref.mean_relative_error, atol=1e-11):
                raise AssertionError("Regenerated history differs from the supplied primary history")
            if not np.isclose(empirical_regret, ref.regret_cny_per_month, rtol=1e-9, atol=1e-6):
                raise AssertionError("Regenerated empirical decision differs from supplied data")
            action = primary.optimized_decision(gaussian, capacity, policy)
            cost = primary.true_cost_of_estimated_action(action["selected_mode"],
                action["contract_D_kw"], oracle, capacity, policy)
            regret = max(0, cost - truth["best_value_cny"])
            threshold = policy.alpha * truth["contract_D_kw"]
            errors = [abs(action["mode_costs"][mode] - truth["mode_costs"][mode])
                      for mode in truth["mode_costs"]]
            records.append(dict(replication_id=replication, method_id="GAUSSIAN", cell_id=cell["cell_id"],
                oracle_mode=truth["selected_mode"], selected_mode=action["selected_mode"],
                action_correct=action["selected_mode"] == truth["selected_mode"],
                q99_relative_error=gaussian.ppf(.99) / oracle.ppf(.99) - 1,
                mean_relative_error=gaussian.mean() / oracle.mean() - 1,
                stoploss_relative_error=float(gaussian.stop_loss(threshold)) / oracle.stop_loss(threshold) - 1,
                d_mv=2 * max(errors) / truth["margin_cny"],
                regret_cny_per_month=regret, regret_over_oracle_value=regret / truth["best_value_cny"],
                estimated_contract_D_kw=action["contract_D_kw"]))
            if replication < 100:
                for label, distribution in [("GAUSSIAN", gaussian), ("EMP", empirical)]:
                    validations.append(check_solver(distribution, cell, label, replication))
            if replication == 0:
                for mode in truth["mode_costs"]:
                    worked.append(dict(replication_id=0, method_id="GAUSSIAN", cell_id=cell["cell_id"],
                        mode=mode, estimated_value_cny=action["mode_costs"][mode],
                        oracle_value_cny=truth["mode_costs"][mode],
                        estimated_contract_D_kw=action["contract_D_kw"],
                        oracle_contract_D_kw=truth["contract_D_kw"], selected_mode=action["selected_mode"],
                        regret_cny_per_month=regret))
        if (replication + 1) % 100 == 0:
            print(f"Gaussian histories: {replication + 1}/1000", flush=True)
    extra = [
        ("ORACLE", oracle),
        ("KDE", primary.NonnegativeGaussianKDE(np.array([80000.,100000.,140000.,170000.,220000.]))),
        ("GEV", primary.TruncatedGEV(-.15, 130000., 20000., 10 * oracle.ppf(.999))),
        ("TAIL_POISSON", primary.TailProcessDistribution(80000., .15, 12000., np.array([70000.,75000.]), "poisson", 30., 30., None, None)),
        ("TAIL_NEGATIVE_BINOMIAL", primary.TailProcessDistribution(80000., .15, 12000., np.array([70000.,75000.]), "negative_binomial", 30., 60., 30., .5)),
        ("POINT_MASS", primary.DiscreteDistribution(np.array([500000.]))),
        ("FLAT_OPTIMUM", primary.DiscreteDistribution(np.array([120000.,180000.]), np.array([1 - 1 / 2.4, 1 / 2.4]))),
    ]
    for label, distribution in extra:
        for cell in cells:
            validations.append(check_solver(distribution, cell, label, -1))
    result = pd.DataFrame(records)
    result.to_csv(OUT / "gaussian_decisions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(validations).to_csv(OUT / "contract_solver_comparison.csv", index=False)
    pd.DataFrame(worked).to_csv(OUT / "worked_tariff_example.csv", index=False)
    combined = pd.concat([source, result], ignore_index=True)
    regional = combined.groupby(["replication_id", "method_id", "oracle_mode"]).regret_cny_per_month.mean().unstack("method_id")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 1000, (5000, 1000))
    summaries = []
    for region, frame in regional.groupby(level="oracle_mode"):
        for comparator in ("EMP", "KDE", "GEV", "TAIL", "EVENT-EMP"):
            differences = (frame.GAUSSIAN - frame[comparator]).to_numpy()
            low, high = np.quantile(differences[indices].mean(axis=1), [.025, .975])
            summaries.append(dict(oracle_mode=region, contrast="GAUSSIAN-" + comparator,
                mean_difference_cny=differences.mean(), ci95_low=low, ci95_high=high,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=5000, histories=1000))
    pd.DataFrame(summaries).to_csv(OUT / "gaussian_regional_contrasts.csv", index=False)
    fidelity = result.groupby("replication_id")[["q99_relative_error", "mean_relative_error", "stoploss_relative_error"]].agg(lambda x: np.abs(x).mean())
    fidelity.agg(["median", "mean"]).to_csv(OUT / "gaussian_fidelity.csv", index_label="summary")
    design = pd.read_csv(ROOT / "data/derived/primary_design.csv")
    design["sufficient_w1_radius_kw"] = design.oracle_margin_cny / (2 * 36 * 2)
    design["radius_percent_oracle_mean"] = 100 * design.sufficient_w1_radius_kw / oracle.mean()
    design.to_csv(OUT / "mode_stability_radii.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"Solver comparisons: {len(validations)}; maximum excess cost: "
          f"{max(row['excess_cost_cny'] for row in validations):.8g} CNY", flush=True)


if __name__ == "__main__":
    main()
