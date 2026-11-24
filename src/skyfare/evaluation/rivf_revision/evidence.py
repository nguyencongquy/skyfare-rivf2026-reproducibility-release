"""Access verified research inputs without modifying the frozen study artifacts."""

from __future__ import annotations

import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from skyfare.core.paths import repository_root
from skyfare.models.temporal_runtime import sha256

ROOT = repository_root()
CONTRACT = ROOT / "configs/pooled_evaluation_contract.json"
CLASS_FILE = "classification_training_frame.parquet"
REG_FILE = "regression_training_frame.parquet"
FEATURE_FILES = {"CLASSIFICATION": "classification_features.parquet", "REGRESSION": "regression_features.parquet"}
LABEL_FILES = {"CLASSIFICATION": "classification_labels.parquet", "REGRESSION": "regression_labels.parquet"}
TASK_FILES = {"CLASSIFICATION": "classification", "POINT": "point", "DISTRIBUTION": "distribution", "RANKING": "ranking"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def preflight(number: int) -> dict:
    label = {1: "one", 2: "two"}[number]
    return read_json(ROOT / f"artifacts/evidence/prospective_test_{label}/preflight.json")


def verified_development(path: Path, filename: str) -> Path:
    file = path / filename
    if sha256(file) != preflight(1)["source_hashes"][f"inputs/{filename}"]:
        raise RuntimeError(f"Development frame does not match the paper: {filename}")
    return file


def test_archive(results_dir: Path, number: int) -> Path:
    archive_entry = read_json(CONTRACT)["expected_archives"][f"test_{number}"]
    archive = results_dir / archive_entry["filename"]
    if sha256(archive) != archive_entry["sha256"]:
        raise RuntimeError(f"Unexpected result archive hash: {archive.name}")
    return archive


def test_members(results_dir: Path, number: int, *, features_only: bool = False) -> dict[str, bytes]:
    archive = test_archive(results_dir, number)
    base = f"test{number}_outputs"
    required = {
        f"{base}/staging/{filename}": filename for filename in FEATURE_FILES.values()
    }
    if not features_only:
        required.update({f"{base}/staging/{filename}": filename for filename in LABEL_FILES.values()})
        required.update({
            f"{base}/frozen_predictions/{stem}_test{number}_predictions.parquet": task
            for task, stem in TASK_FILES.items()
        })
    hashes = preflight(number)["staged_hashes"]
    block = {1: "one", 2: "two"}[number]
    commit = read_json(ROOT / f"artifacts/evidence/prospective_test_{block}/prediction_commit.json")
    hashes.update({f"{base}/{item['path']}": item["sha256"] for item in commit["files"].values()})

    results = {}
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if member.name not in required:
                continue
            if not member.isfile() or required[member.name] in results:
                raise RuntimeError(f"Unexpected archive member type or duplicate: {member.name}")
            payload = handle.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != hashes[member.name]:
                raise RuntimeError(f"Research file failed SHA-256 check: {member.name}")
            results[required[member.name]] = payload
    if set(results) != set(required.values()):
        raise RuntimeError(f"Missing archived evidence for Test {number}: {set(required.values()) - results.keys()}")
    return results


def read_table(content: bytes, columns=None) -> pd.DataFrame:
    return pd.read_parquet(BytesIO(content), columns=columns)
