# Reproduce the computational study

The main repository command, `python scripts/reproduce_results.py`, checks the released records and recomputes the reported numerical summaries. This directory contains the underlying simulation, fitting, decision, numerical-bound, and external-validation code.

## Inputs and environment

Use Python 3.12 and install the pinned packages:

```text
python -m pip install -r reproduction/exp1/requirements_cssc.txt
```

Download `CSSC_Reproduction_Records.zip` from the repository's [CSSC submission release](https://github.com/riddlehuang05/extreme-demand-tariff-analysis/releases/tag/cssc-submission-v1) and extract it at the repository root. It contains `source_data/` with the decision, fitted-law, numerical-audit, and analysis records. The archive SHA-256 is `7b17f78a19e35a122702cb8ec6e92e65927c87e96365ef55c7ce35bc4b884ea4`. Generate the synthetic histories from the fixed seeds with the command below.

The external example uses the public South Korean manufacturing-load data cited in the paper. The derived inputs, rolling-validation script, and prediction records are in `reproduction/external/`.

## Primary simulation

The synthetic inputs are generated from the fixed seeds and published mechanism:

```text
python reproduction/generate_primary_histories.py --output work/histories --replications 10000
```

The generator uses seed root `20261001` and namespace `(2, 0, replication_id, 10)`. Its first two histories were checked row for row against the original inputs. The following command fits all five methods to the regenerated histories, selects TAIL thresholds using four-fold held-out CRPS, and evaluates the nine tariff cells:

```text
python reproduction/exp1/code/revised/run_primary_no_nominal_ks.py --base-repo reproduction/exp1/code/legacy_engine --data work/histories --run-manifest work/histories/offline_history_manifest.json --output work/primary --start 0 --end 10000 --chunk-size 25 --workers 8
```

For a smaller smoke run, replace `--end 10000` with `--end 1` and `--workers 8` with `--workers 1`. This one-history path was executed successfully from the repository code.

## Further experiments and checks

The `reproduction/exp1/code/revised/` directory includes the negative-binomial–Weibull challenge, matched excess-law ablation, adaptive Wasserstein calculation, high-contract-quantile analysis, and source-to-table scripts. `reproduction/exp1/code/legacy_engine/` contains the generator, five estimators, and tariff decision engine. The complete archived inputs and outputs are in `source_data/data/source/`; `source_data/README.md` describes the experiments and `source_data/SOURCE_DATA_INDEX.csv` maps each archived file to its scientific use.

The lightweight summary command requires no separate download:

```text
python scripts/reproduce_results.py
```

It verifies the released hashes and checks the adaptive Wasserstein bounds, 450,000 numerical decision checks, full solver allowances, paired comparisons, and event-level ablation summaries.
