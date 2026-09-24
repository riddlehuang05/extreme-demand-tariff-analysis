# Experiment design

The controlled experiments compare predictive distributions through both
distributional accuracy and the tariff decisions they imply. The analytic
data-generating distribution defines the oracle tariff choice and expected
cost used to calculate regret.

## Estimation and tariff evaluation

The oracle is evaluated using its analytic cumulative distribution function
and deterministic one-dimensional integration. EMP and KDE tariff functionals
are analytic; TAIL, GEV, and EVENT-EMP functionals are evaluated deterministically.

The primary comparison includes two event-level methods, TAIL and EVENT-EMP,
and three monthly-maxima methods, GEV, KDE, and EMP.

| Setting | Specification |
| --- | --- |
| Training history | 24 months |
| TAIL threshold selection | Fit on the first 19 months; validate on the final five |
| TAIL refit | Hold the selected threshold fixed; refit GPD and count parameters using all 24 months |
| Finite-mean condition | Accept the final GPD when `xi < 1`; otherwise use an exponential fallback when possible, then empirical monthly maxima |
| Replications | 1,000 independent histories, paired across methods |
| Tariff settings | Nine prespecified alpha/utilization cells; capacities calculated from the analytic oracle mean |
| Uncertainty unit | Replication identifier |

Estimator settings were specified before the full simulation.

## Event-frequency sensitivity

Event rates are 0.2, 1, 5, and 30 per month, with 200 histories per rate.
The tariff block uses alpha = 1.20 and seven utilization settings. Other
data-generating and tariff parameters are held fixed.

## Decision diagnostics

The diagnostic analysis uses the 45,000 primary decision records. The nine
tariff cells form three equal-size strata according to the analytic relative
oracle gap. AUC is calculated within each cell and macro-averaged within each
stratum. Regret associations are reported within method and after
within-method ranking.

A generalized estimating equation clusters by replication and uses
`mode_error ~ diagnostic percentile + cell fixed effects + method fixed effects`.
Five-fold grouped cross-validation keeps all records from a replication in
the same fold and compares fixed effects alone with fixed effects plus each
diagnostic. The optimized-mode-value ratio `d_MV < 1` is evaluated as a
retrospective certificate of mode agreement.

## External predictive evaluation

The `natural_tail` and `all_observations` variants use four rolling origins,
12 candidate specifications, 50,000 predictive draws, and factory-cluster
bootstrap intervals. Candidate selection uses leave-one-training-month-out
validation, excluding the test month.

Both variants use fixed, non-overlapping 15-minute windows and seven methods,
including K-BLOCK. Variant scores are paired by factory, origin, and method
before the sensitivity summary is calculated. These analyses assess predictive
performance on observed loads; the controlled simulations assess tariff regret.

## Operating-regime sensitivity

The stationary control uses the target distribution for all 24 training
months. Each changing-regime scenario alters one parameter during months
1–12; months 13–24 and the target oracle use the target distribution:

- Event rate: 5 to 30 per month.
- GPD scale: 8,000 to 12,000.
- GPD shape: 0.05 to 0.15.

The four scenarios share the early random-number namespace and the same
late 12-month samples within replication. Each scenario uses 200 paired
histories, five methods, and nine tariff cells. Settings were specified before
the full run. See [configs/regime_shift_sensitivity.yaml](../configs/regime_shift_sensitivity.yaml).
