"""Fit one development-only comparator per task, then score both test blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.calibration.methods import calibrate_probability, calibrate_quantiles
from skyfare.models.candidate_models import fit_predict
from skyfare.models.selection_contract import QUANTILES, TEST_BLOCKS
from skyfare.models.temporal_runtime import (
    CLASS_CATEGORICAL,
    CLASS_NUMERIC,
    REG_CATEGORICAL,
    REG_NUMERIC,
    inner_split,
    sha256,
    write_json_atomic,
    write_parquet_atomic,
)

TASKS = ("CLASSIFICATION", "POINT", "DISTRIBUTION", "RANKING")
Q_COLUMNS = tuple(f"q{round(q * 100):02d}_relative_log" for q in QUANTILES)
PROJECTION = {
    "CLASSIFICATION": tuple(dict.fromkeys((
        "row_key", "feature_time", *CLASS_CATEGORICAL, *CLASS_NUMERIC,
    ))),
    "REGRESSION": tuple(dict.fromkeys((
        "row_key", "feature_time", "query_id", "prior_anchor_vnd", "regime",
        *REG_CATEGORICAL, *REG_NUMERIC,
    ))),
}
DEVELOPMENT_ONLY = {
    "CLASSIFICATION": ("label_time", "DROP_5PCT"),
    "REGRESSION": ("label_time", "target_anchor_relative_log", "target_session_price_vnd"),
}
EXPECTED_ROWS = {
    "TEST_1": {"CLASSIFICATION": 41_573, "REGRESSION": 95_962},
    "TEST_2": {"CLASSIFICATION": 42_146, "REGRESSION": 94_127},
}


def load_inputs(bundle: Path, source: str, window: str, cutoff: pd.Timestamp):
    development = pd.read_parquet(bundle / "inputs" / f"development_{source.lower()}.parquet")
    if set(development) != set(PROJECTION[source] + DEVELOPMENT_ONLY[source]):
        raise RuntimeError(f"Unexpected development columns for {source}")
    cutoff_ok = development[["feature_time", "label_time"]].lt(cutoff).all().all()
    if not cutoff_ok or not development["feature_time"].lt(development["label_time"]).all():
        raise RuntimeError(f"Development information boundary violated for {source}")
    if development["row_key"].duplicated().any():
        raise RuntimeError(f"Duplicate development row key for {source}")
    if source == "REGRESSION" and not development["prior_anchor_vnd"].gt(0).all():
        raise RuntimeError("Invalid development anchors")

    training = development
    if window == "RECENT84":
        training = training.loc[training["label_time"].ge(cutoff - pd.Timedelta(days=84))]
    elif window != "EXPANDING":
        raise RuntimeError(f"Unknown development window {window}")
    fit, head = inner_split(training, source)
    blocks = []
    for block, (start, end) in TEST_BLOCKS.items():
        frame = pd.read_parquet(bundle / "inputs" / f"{block.lower()}_{source.lower()}.parquet")
        if set(frame) != set(PROJECTION[source]) or len(frame) != EXPECTED_ROWS[block][source]:
            raise RuntimeError(f"Unexpected inference schema or row count for {block}/{source}")
        latest = pd.Timestamp(end) + pd.Timedelta(days=1)
        earliest = pd.Timestamp(start) - pd.Timedelta(days=int(source == "REGRESSION"))
        if not frame["feature_time"].between(earliest, latest, inclusive="left").all():
            raise RuntimeError(f"Unexpected feature timestamps for {block}/{source}")
        if source == "REGRESSION" and not frame["prior_anchor_vnd"].gt(0).all():
            raise RuntimeError("Invalid inference anchors")
        blocks.append(frame.assign(block_id=block))
    valid = pd.concat(blocks, ignore_index=True)
    if valid["row_key"].duplicated().any() or valid["row_key"].isin(development["row_key"]).any():
        raise RuntimeError(f"Scoring identity overlaps another block or development for {source}")
    if head["label_time"].max() > valid["feature_time"].min():
        raise RuntimeError(f"Calibration outcome occurs after first scoring feature for {source}")
    return fit, head, valid


def predict_task(task: str, job: dict, fit, head, valid, output: Path):
    raw_head, raw_test, metadata = fit_predict(job, fit, head, valid, smoke=False, artifact_dir=output)
    result = valid[["block_id", "row_key"]].copy()
    if task == "CLASSIFICATION":
        result["probability"], metadata["calibration"] = calibrate_probability(
            head["DROP_5PCT"], raw_head, raw_test
        )
    elif task == "POINT":
        result["predicted_price_vnd"] = np.maximum(valid["prior_anchor_vnd"] * np.exp(raw_test), 1.0)
    elif task == "DISTRIBUTION":
        calibrated, metadata["calibration"] = calibrate_quantiles(head, raw_head, valid, raw_test)
        result[list(Q_COLUMNS)] = calibrated
    else:
        result["score"] = raw_test
    if not np.isfinite(result.iloc[:, 2:].to_numpy()).all():
        raise RuntimeError(f"Non-finite prediction for {task}")
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--check-only", action="store_true", help="Verify inputs and boundaries without training")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = args.output_dir.resolve()
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["sha256"].items():
        if sha256(bundle / name) != digest:
            raise RuntimeError(f"Bundle content changed: {name}")
    protocol = json.loads((bundle / "configs/rivf_learned_baselines.json").read_text())
    if protocol["refit_for_test_2"] or protocol["fit_end_exclusive"] != TEST_BLOCKS["TEST_1"][0]:
        raise RuntimeError("Expected the locked development-only protocol")
    cutoff = pd.Timestamp(protocol["fit_end_exclusive"])
    run_hash = sha256(manifest_path)
    tasks = {}
    for task in TASKS:
        job = {**protocol["models"][task], "task": task, "seed": protocol["seed"]}
        source = "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION"
        fit, head, valid = load_inputs(bundle, source, job["window"], cutoff)
        stats = {
            "candidate_id": job["candidate_id"],
            "seed": job["seed"],
            "fit_rows": len(fit),
            "head_rows": len(head),
            "scored_rows": {key: int(n) for key, n in valid.groupby("block_id").size().items()},
            "fit_label_max": fit["label_time"].max(),
            "head_label_max": head["label_time"].max(),
        }
        print(f"{task}: {stats}", flush=True)
        if args.check_only:
            continue
        path = output / f"{task.lower()}.parquet"
        receipt = output / f"{task.lower()}.json"
        if path.is_file() and receipt.is_file():
            old = json.loads(receipt.read_text())
            if old["bundle_manifest_sha256"] == run_hash and old["predictions_sha256"] == sha256(path):
                print(f"{task}: verified existing predictions", flush=True)
                tasks[task] = old
                continue
            raise RuntimeError(f"Existing {task} result belongs to another run. Use a fresh output directory.")
        predictions, model = predict_task(task, job, fit, head, valid, output)
        write_parquet_atomic(path, predictions)
        tasks[task] = {
            **stats, "model": model, "predictions_sha256": sha256(path),
            "bundle_manifest_sha256": run_hash,
        }
        write_json_atomic(receipt, tasks[task])
    if not args.check_only:
        write_json_atomic(output / "RESULTS.json", {"bundle_manifest_sha256": run_hash, "tasks": tasks})


if __name__ == "__main__":
    main()
