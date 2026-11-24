"""Refit the four B1 comparators at the original Test 2 boundary."""

import argparse
import importlib.metadata
import json
import os
import tarfile
import tempfile
from hashlib import sha256 as digest
from pathlib import Path

import pandas as pd

from skyfare.evaluation.rivf_revision.evidence import (
    FEATURE_FILES, ROOT, preflight, read_json, read_table, test_archive, test_members,
)
from skyfare.evaluation.rivf_revision.learned_baselines import EXPECTED_ROWS, Q_COLUMNS, TASKS, predict_task
from skyfare.models.temporal_runtime import sha256, write_json_atomic, write_parquet_atomic


def reuse_archived_refit(archive_path, job, fit, head, valid):
    task = job["task"]
    identifier = f"{job['candidate_id']}__TEST_2__S{job['seed']}"
    prefix = f"test2_outputs/jobs/{identifier}"
    names = {f"{prefix}/done.json", f"{prefix}/predictions.parquet"}
    payloads = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for item in archive:
            if item.name in names:
                if not item.isfile() or item.name in payloads:
                    raise RuntimeError(f"Invalid original refit member: {item.name}")
                payloads[item.name] = archive.extractfile(item).read()
    if names != set(payloads):
        raise RuntimeError(f"Missing original refit result for {identifier}")
    record = json.loads(payloads[f"{prefix}/done.json"])
    stored = payloads[f"{prefix}/predictions.parquet"]
    preflight_file = ROOT / "artifacts/evidence/prospective_test_two/preflight.json"
    if (
        record["status"] != "COMPLETE" or record["test_labels_read"] is not False
        or record["preflight_sha256"] != sha256(preflight_file)
        or record["prediction_sha256"] != digest(stored).hexdigest()
        or record["job"]["job_id"] != identifier or record["job"]["fold"] != "TEST_2"
        or record["job"]["phase"] != "TEST_2_SCORE"
        or any(record["job"][key] != job[key] for key in ("candidate_id", "task", "configuration", "family", "seed", "window"))
        or (record["fit_rows"], record["head_rows"], record["rows"]) != (len(fit), len(head), len(valid))
    ):
        raise RuntimeError(f"Archived refit does not match B1 and the original Test 2 split: {identifier}")
    columns = Q_COLUMNS if task == "DISTRIBUTION" else ("score",)
    saved = read_table(stored, ["row_key", "feature_time", "label_time", *columns])
    aligned = valid[["row_key", "feature_time", "label_time"]].merge(
        saved, on=["row_key", "feature_time", "label_time"], validate="one_to_one",
    )
    if len(aligned) != len(valid) or len(saved) != len(valid):
        raise RuntimeError(f"Archived refit scores do not match the sealed features: {identifier}")
    predictions = aligned[["row_key", *columns]].assign(block_id="TEST_2")
    return predictions, {
        "calibration": record.get("calibration", {}), "archived_job": record["job"],
        "archived_prediction_sha256": record["prediction_sha256"],
        "archived_preflight_sha256": record["preflight_sha256"],
        "archived_code_sha256": record["code_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--archives", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task", choices=TASKS, nargs="+", default=TASKS)
    parser.add_argument("--reuse-archived-members", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    receipt = preflight(2)
    provenance = {}
    for root, prefix, suffix in (
        (args.development, "inputs", "training_frame"),
        (args.prior, "prior_test_1", "test_1_frame"),
    ):
        for stem in ("classification", "regression"):
            name = f"{stem}_{suffix}.parquet"
            key = f"{prefix}/{name}"
            provenance[key] = sha256(root / name)
            if provenance[key] != receipt["source_hashes"][key]:
                raise RuntimeError(f"Input differs from the original Test 2 refit: {key}")
    config = ROOT / "configs/rivf_learned_baselines.json"
    protocol = read_json(config)
    features = test_members(args.archives, 2, features_only=True)
    archive_path = test_archive(args.archives, 2) if args.reuse_archived_members else None
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "lightgbm", "catboost",
    )}
    with tempfile.TemporaryDirectory(prefix="skyfare-refit-") as temp:
        work = Path(temp)
        staging = work / "staging"
        staging.mkdir()
        for name in FEATURE_FILES.values():
            (staging / name).write_bytes(features[name])
            provenance[name] = sha256(staging / name)
        os.environ["SKYFARE_TEST_TWO_DEVELOPMENT_INPUT_ROOT"] = str(args.development.resolve())
        os.environ["SKYFARE_TEST_TWO_PRIOR_ROOT"] = str(args.prior.resolve())
        os.environ["SKYFARE_TEST_TWO_OUTPUT_ROOT"] = str(work)
        from skyfare.evaluation.prospective_test_two import runtime

        cutoff = pd.Timestamp(runtime.TEST_2[0])
        if not protocol["refit_for_test_2"] or pd.Timestamp(protocol["test_2_fit_end_exclusive"]) != cutoff:
            raise RuntimeError("B1 configuration does not match the frozen Test 2 boundary")
        for task in args.task:
            job = {**protocol["models"][task], "task": task, "seed": protocol["seed"]}
            source = "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION"
            fit, head, valid = runtime.training_frames(task, job["window"])
            forbidden = {"DROP_5PCT", "target_anchor_relative_log", "target_session_price_vnd", "target_price_vnd"}
            if forbidden.intersection(valid.columns) or len(valid) != EXPECTED_ROWS["TEST_2"][source]:
                raise RuntimeError(f"Unexpected Test 2 feature schema or size: {task}")
            if set(valid.row_key).intersection(fit.row_key) or set(valid.row_key).intersection(head.row_key):
                raise RuntimeError(f"Test 2 row entered training: {task}")
            if not (
                fit.label_time.max() < head.label_time.max() < cutoff
                and head.label_time.max() <= valid.feature_time.min()
            ):
                raise RuntimeError(f"Test 2 training crossed the information boundary: {task}")
            info = {
                "task": task, "candidate_id": job["candidate_id"], "window": job["window"],
                "seed": job["seed"], "cutoff_exclusive": cutoff, "fit_rows": len(fit),
                "head_rows": len(head), "scored_rows": len(valid), "fit_label_max": fit.label_time.max(),
                "head_label_max": head.label_time.max(), "first_scoring_feature": valid.feature_time.min(),
                "config_sha256": sha256(config), "input_sha256": provenance, "versions": versions,
            }
            name = task.lower()
            if archive_path is not None and task in {"DISTRIBUTION", "RANKING"}:
                predictions, metadata = reuse_archived_refit(archive_path, job, fit, head, valid)
                info["prediction_source"] = "VERIFIED_ORIGINAL_TEST_2_REFIT"
                info["test_2_archive_sha256"] = sha256(archive_path)
                info["verification_versions"] = info.pop("versions")
            else:
                print(f"{task}: fitting {len(fit):,} rows and calibrating on {len(head):,}", flush=True)
                predictions, metadata = predict_task(task, job, fit, head, valid.assign(block_id="TEST_2"), work)
                info["prediction_source"] = "NEW_LOCAL_TEST_2_REFIT"
            path = out / f"{name}.parquet"
            write_parquet_atomic(path, predictions)
            write_json_atomic(out / f"{name}.json", {
                **info, "model": metadata, "predictions_sha256": sha256(path),
            })
            print(f"{task}: verified predictions written for {len(predictions):,} rows", flush=True)


if __name__ == "__main__":
    main()
