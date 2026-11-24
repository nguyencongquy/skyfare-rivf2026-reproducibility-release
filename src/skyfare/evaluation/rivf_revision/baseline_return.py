"""Validate a Vast.ai return against the exact private bundle used to fit B1."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from skyfare.evaluation.rivf_revision.evidence import read_table
from skyfare.evaluation.rivf_revision.learned_baselines import EXPECTED_ROWS, Q_COLUMNS, TASKS

SCORE_COLUMNS = {
    "CLASSIFICATION": ("probability",),
    "POINT": ("predicted_price_vnd",),
    "DISTRIBUTION": Q_COLUMNS,
    "RANKING": ("score",),
}


def return_files(path: Path) -> dict[str, bytes]:
    names = {"RESULTS.json", *(f"{task.lower()}.{ext}" for task in TASKS for ext in ("json", "parquet"))}
    if path.is_dir():
        return {name: (path / name).read_bytes() for name in names}
    with ZipFile(path) as archive:
        expected = {f"results/{name}" for name in names}
        actual = [file.filename for file in archive.infolist() if not file.is_dir()]
        if set(actual) != expected or len(actual) != len(expected):
            raise RuntimeError("B1 ZIP must contain the generated results/ directory and all four task receipts")
        return {name: archive.read(f"results/{name}") for name in names}


def verified_predictions(path: Path, training_bundle: Path) -> dict[str, pd.DataFrame]:
    with ZipFile(training_bundle) as packaged:
        manifest_bytes = packaged.read("MANIFEST.json")
        manifest_hash = sha256(manifest_bytes).hexdigest()
        config = "configs/rivf_learned_baselines.json"
        protocol_bytes = packaged.read(config)
    if sha256(protocol_bytes).hexdigest() != json.loads(manifest_bytes)["sha256"][config]:
        raise RuntimeError("Training bundle protocol does not match its manifest")
    protocol = json.loads(protocol_bytes)
    cutoff = pd.Timestamp(protocol["fit_end_exclusive"])
    files = return_files(path)
    receipt = json.loads(files["RESULTS.json"])
    if receipt["bundle_manifest_sha256"] != manifest_hash or set(receipt["tasks"]) != set(TASKS):
        raise RuntimeError("Incomplete or unrelated learned-baseline run")

    result = {}
    for task in TASKS:
        name = task.lower()
        source = "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION"
        counts = {block: per_source[source] for block, per_source in EXPECTED_ROWS.items()}
        record = receipt["tasks"][task]
        if json.loads(files[f"{name}.json"]) != record:
            raise RuntimeError(f"{task}: individual and aggregate run receipts disagree")
        if (
            record["bundle_manifest_sha256"] != manifest_hash
            or record["candidate_id"] != protocol["models"][task]["candidate_id"]
            or record["seed"] != protocol["seed"]
            or record["scored_rows"] != counts
            or not (0 < record["fit_rows"] and 0 < record["head_rows"])
            or not (pd.Timestamp(record["fit_label_max"]) < pd.Timestamp(record["head_label_max"]) < cutoff)
        ):
            raise RuntimeError(f"{task}: run receipt does not satisfy the locked B1 protocol")
        payload = files[f"{name}.parquet"]
        if sha256(payload).hexdigest() != record["predictions_sha256"]:
            raise RuntimeError(f"{task}: predictions fail receipt hash")
        frame = read_table(payload)
        score_columns = SCORE_COLUMNS[task]
        if (
            set(frame) != {"block_id", "row_key", *score_columns}
            or frame["row_key"].isna().any()
            or frame["row_key"].duplicated().any()
            or frame.groupby("block_id", dropna=False).size().to_dict() != counts
        ):
            raise RuntimeError(f"{task}: invalid prediction schema, keys or cohort sizes")
        scores = frame[list(score_columns)].to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"{task}: non-finite predictions")
        if task == "CLASSIFICATION" and ((scores < 0) | (scores > 1)).any():
            raise RuntimeError("Expected classification probabilities in [0, 1]")
        if task == "POINT" and (scores <= 0).any():
            raise RuntimeError("Expected positive fare predictions")
        if task == "DISTRIBUTION" and (np.diff(scores, axis=1) < 0).any():
            raise RuntimeError("Unexpected learned quantile crossing")
        result[task] = frame
    return result
