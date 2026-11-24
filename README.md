# SkyFare RIVF 2026 Reproducibility Code

This snapshot contains the temporal evaluation pipeline, frozen configuration,
reviewer analysis code, and synthetic schema examples. Third-party fare
observations, acquisition software, fitted models, and row-level predictions
are excluded from this distribution.

Use Python 3.11 and install `requirements.txt`. The synthetic CSVs illustrate
schema only. They contain fabricated identifiers, timestamps and values.
A scientific run requires authorized study inputs matching the archived hashes.

From the root:

```bash
export PYTHONPATH=src
python -m skyfare.evaluation.rivf_revision.analyse \
  --archives /path/to/test-archives --raw-trip /path/to/trip-snapshots \
  --learned-results /path/to/B1-results.zip --training-bundle /path/to/B1-training.zip \
  --out /path/to/recomputed-evidence
```

For B2, B4 and B5 alone, omit both B1 ZIP arguments. With authorized inputs,
`python tools/package_rivf_baselines.py --help` describes how to rebuild the
private training bundle. The `supplementary` and `route_sensitivity` modules
in `skyfare.evaluation.rivf_revision` document the optional checks with `--help`.

`schemas/rivf_model_inputs.json` documents the input and development-only
columns. The raw-offer and processed-frame schemas are also included, along
with a fabricated raw-offer record. The frozen protocol and SHA-256 records are under `configs` and
`artifacts/evidence`. Aggregate revision results are in `revision_evidence`.
N1 and N2 retain the frozen cohorts and report post-hoc checks. N3 samples
whole directional routes across both test periods and holds observation days
fixed. Its intervals supplement the original day bootstrap.
