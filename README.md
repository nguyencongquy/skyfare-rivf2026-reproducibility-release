# SkyFare RIVF 2026 Reproducibility Code

This snapshot contains the temporal evaluation pipeline, frozen configuration,
reviewer analysis code, and synthetic schema examples. Third-party fare
observations, acquisition software, fitted models, and row-level predictions
are excluded from this distribution.

Use Python 3.11 and install `requirements.txt`. The synthetic CSVs illustrate
schema only. They contain fabricated identifiers, timestamps and values.
A scientific run requires authorized study inputs matching the archived hashes.

## Revised evaluation results

The prospective study uses Test 1 (29 July to 8 August 2026) and Test 2
(9 to 19 August 2026). The tables in `revision_evidence` report each test
and their pooled result. The B1 tables use the original development-only fit
for Test 1 and a refit at the Test 2 boundary for Test 2. The classification,
point and quantile comparators use expanding windows. The ranker uses its
prespecified 84-day window, advanced to the same boundary. The development-only
B1 comparison for both tests is retained in `revision_evidence/previous_b1`.

For the revised pooled comparison, the paired 95% confidence intervals favour
the ensemble for MAPE, WIS and NDCG@5. The proper-score interval includes zero.
The same directory contains development proper scores by fold and seed,
calibration and cutoff receipts, pre-calibration quantile coverage and WIS
bounds, the AUC tie analysis, and the development target provenance check.

## Recompute B1

These commands run from the repository root. The development and prior
directories are the `inputs` and `sealed_test_1` directories from the
authorized Test 1 R2 bundle. The archive directory holds the verified Test 1
and Test 2 result archives. The original B1 results and training ZIPs supply
the verified Test 1 comparator predictions. Keep these inputs local.

```bash
export PYTHONPATH=src
python -m skyfare.evaluation.rivf_revision.refit_baselines \
  --development /path/to/inputs --prior /path/to/sealed_test_1 \
  --archives /path/to/test-archives --out /path/to/refit-predictions \
  --reuse-archived-members
python -m skyfare.evaluation.rivf_revision.score_baselines \
  --archives /path/to/test-archives --results /path/to/B1-results.zip \
  --bundle /path/to/B1-training.zip --refit /path/to/refit-predictions \
  --out /path/to/recomputed-evidence
```

`--reuse-archived-members` selects the verified Test 2 refits for the
matching CatBoost quantile and LightGBM ranking jobs. The classifier and
point regressor are fitted locally. Omit the flag to fit all four comparators
locally. The released receipts record the prediction source, cutoff, seed,
input hashes and versions. The legacy training-bundle command is
`python tools/package_rivf_baselines.py --help`.

## Recompute the other checks

To audit exact test targets and reproduce the policy tables:

```bash
export PYTHONPATH=src
python -m skyfare.evaluation.rivf_revision.analyse \
  --archives /path/to/test-archives --raw-trip /path/to/trip-snapshots \
  --out /path/to/recomputed-evidence
```

For the development proper scores, AUC analysis and development target audit,
run `python -m skyfare.evaluation.rivf_revision.audit_feedback --help`. The
`scores` check also needs the verified V24 development selection results
archive in the archive directory. For interval coverage and WIS bounds, use
`python -m skyfare.evaluation.rivf_revision.quantile_bounds --help`. The
`supplementary` and `route_sensitivity` modules document N1–N3.

`schemas/rivf_model_inputs.json` documents the input and development-only
columns. The frozen protocol and SHA-256 records are under `configs` and
`artifacts/evidence`.
