# Revision design frozen before execution

Date: 2026-09-15

## Primary claims and evidence

1. The oracle must not contribute fixed Monte Carlo error. The DGP oracle is evaluated from its analytic CDF and deterministic one-dimensional integration.
2. Predictive-distribution Monte Carlo should not determine tariff actions. EMP and KDE functionals are analytic; TAIL, GEV, and event-level empirical functionals are evaluated deterministically.
3. The event-level TAIL pipeline must not be compared only with monthly-maxima baselines. The main design therefore includes an event-level empirical comparator and a monthly-maxima GEV comparator.
4. Diagnostic evidence must respect cell and method structure. AUC is calculated within cell and macro-averaged; regret associations are reported within method and after within-method ranking.
5. The optimized-mode-value ratio is a retrospective certificate. The threshold `d_MV < 1` is evaluated directly and is not described as an independent prospective predictor.
6. Tail-event frequency is varied separately from amplitude at lambda = 0.2, 1, 5, and 30 events/month.

## Frozen main design

- Training history: 24 months.
- Selection split for TAIL: first 19 months fit, final 5 months validate.
- Final TAIL refit: selected threshold is held fixed and all 24 months are used to refit GPD and count parameters.
- Finite-moment gate: final GPD is accepted only if `xi < 1`; otherwise the exponential fallback is used when possible, then the empirical monthly-maxima fallback.
- Methods: TAIL, EVENT-EMP, GEV, KDE, EMP.
- Main replications: 1,000 independent histories, paired across methods.
- Tariff cells: the original nine prespecified alpha/utilization cells, with capacities rebuilt from the analytic oracle mean.
- Primary uncertainty unit: replication identifier.
- No result-dependent estimator tuning after the smoke run.

## Frequency sensitivity

- Tail-event rates: 0.2, 1, 5, 30 per month.
- Replications: 200 per rate.
- Tariff block: alpha = 1.20 and the original seven utilization locations.
- Other DGP and tariff parameters are unchanged.

## Residual diagnostic analysis

- Existing 45,000-row full output is reused without rerunning the main simulation.
- The nine tariff cells are split into three equal-cell strata by analytic relative oracle gap.
- For each diagnostic, AUC is calculated within cell and then macro-averaged within gap stratum.
- Replication-cluster GEE uses `mode_error ~ diagnostic percentile + cell fixed effects + method fixed effects` with replication as the cluster.
- Five-fold grouped cross-validation keeps all rows from one replication in the same fold and compares fixed effects alone with fixed effects plus each diagnostic.

## External all-observation sensitivity

- The same four rolling origins, 12 candidate specifications, 50,000 prediction draws, and factory-cluster bootstrap are used for `natural_tail` and `all_observations`.
- Candidate choice uses leave-one-training-month-out validation only; the test month is excluded from selection.
- Both variants use fixed, non-overlapping 15-minute windows and the same seven methods, including K-BLOCK.
- Variant scores are paired by factory, origin, and method before the factory-cluster sensitivity summary is computed.

## Operating-regime sensitivity

- Task ID: `ASMBI_REGIME_SHIFT_V1`.
- Internal control: all 24 training months follow the target DGP.
- Each ablation changes one early-regime parameter during months 1--12; months 13--24 and the target oracle remain at the target DGP.
- Scenarios: stationary target, rate `5 -> 30`, GPD scale `8000 -> 12000`, and GPD shape `0.05 -> 0.15`.
- The four scenarios share the same early random-number namespace and exactly the same late 12-month samples within replication; late-segment hashes are checked directly.
- Each scenario contains 200 paired replications, five methods, and nine tariff cells. No result-dependent tuning is allowed after the two-replication smoke run.
