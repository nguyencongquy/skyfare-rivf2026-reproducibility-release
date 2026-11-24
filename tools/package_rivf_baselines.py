#!/usr/bin/env python3
"""Build a small private Vast bundle containing projected development and test inputs."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pandas as pd

from skyfare.evaluation.rivf_revision.evidence import (
    CLASS_FILE,
    FEATURE_FILES,
    REG_FILE,
    ROOT,
    read_table,
    test_members,
    verified_development,
)
from skyfare.evaluation.rivf_revision.learned_baselines import (
    DEVELOPMENT_ONLY,
    PROJECTION,
    load_inputs,
)
from skyfare.models.temporal_runtime import regression_row_keys, sha256, write_json_atomic

SOURCE_FILES = (
    "skyfare/__init__.py",
    "skyfare/core/__init__.py",
    "skyfare/core/paths.py",
    "skyfare/models/__init__.py",
    "skyfare/models/candidate_models.py",
    "skyfare/models/selection_contract.py",
    "skyfare/models/temporal_runtime.py",
    "skyfare/calibration/__init__.py",
    "skyfare/calibration/methods.py",
    "skyfare/evaluation/__init__.py",
    "skyfare/evaluation/metrics.py",
    "skyfare/evaluation/rivf_revision/__init__.py",
    "skyfare/evaluation/rivf_revision/learned_baselines.py",
)
REQUIREMENTS = ("numpy", "pandas", "scipy", "pyarrow", "scikit-learn", "lightgbm", "catboost")

RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
export SKYFARE_V24_CPU_THREADS="${SKYFARE_V24_CPU_THREADS:-6}"
exec .venv/bin/python -m skyfare.evaluation.rivf_revision.learned_baselines "$@"
"""

README = """# RIVF 2026 B1 private training bundle

Use a Linux x86-64 Vast.ai instance with Python 3.11 or 3.12 and at least 16 GB RAM.
LightGBM and CatBoost run on CPU. The package contains derived research data and
is intended for the group's authorized machine.

```bash
unzip SKYFARE_RIVF2026_B1_VAST.zip -d skyfare-rivf
cd skyfare-rivf
bash run.sh --check-only
bash run.sh
zip -r SKYFARE_RIVF2026_B1_RESULTS.zip results
```

Return the results ZIP to the paper workspace. If interrupted, rerun the same
command. Tasks with matching receipts and hashes resume without fitting again.
Each model fits once through 28 July 2026 and scores both official test cohorts
without refitting. Test labels stay on the local research machine.
"""


def project_development(directory: Path, source: str, name: str) -> pd.DataFrame:
    path = verified_development(directory, name)
    cols = PROJECTION[source] + DEVELOPMENT_ONLY[source]
    if source == "CLASSIFICATION":
        return pd.read_parquet(path, columns=list(cols))
    columns = [column for column in cols if column != "row_key"]
    frame = pd.read_parquet(path, columns=[*columns, "target_offer_id", "source_session_key"])
    frame["row_key"] = regression_row_keys(frame)
    return frame[list(cols)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing the original two test archives")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="skyfare-rivf-bundle-", dir=output.parent) as temp:
        work = Path(temp)
        inputs = work / "inputs"
        inputs.mkdir()
        for source, name in FEATURE_FILES.items():
            dev_filename = CLASS_FILE if source == "CLASSIFICATION" else REG_FILE
            development = project_development(args.development_dir, source, dev_filename)
            development.to_parquet(inputs / f"development_{source.lower()}.parquet", compression="zstd", index=False)
        for number in (1, 2):
            archived = test_members(args.results_dir, number, features_only=True)
            for source, name in FEATURE_FILES.items():
                frame = read_table(archived[name], columns=list(PROJECTION[source]))
                frame.to_parquet(inputs / f"test_{number}_{source.lower()}.parquet", compression="zstd", index=False)

        files = {str(path.relative_to(work)): path for path in sorted(inputs.iterdir())}
        files.update({f"src/{name}": ROOT / "src" / name for name in SOURCE_FILES})
        config = "configs/rivf_learned_baselines.json"
        files[config] = ROOT / config
        for name, content in (("run.sh", RUN_SCRIPT), ("README.md", README)):
            path = work / name
            path.write_text(content)
            files[name] = path
        requirements = [
            line for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.split("==", 1)[0] in REQUIREMENTS
        ]
        path = work / "requirements.txt"
        path.write_text("\n".join(requirements) + "\n")
        files[path.name] = path
        manifest = work / "MANIFEST.json"
        write_json_atomic(manifest, {"sha256": {name: sha256(path) for name, path in files.items()}})
        files[manifest.name] = manifest
        cutoff = pd.Timestamp("2026-07-29")
        for source, window in (("CLASSIFICATION", "EXPANDING"), ("REGRESSION", "EXPANDING"), ("REGRESSION", "RECENT84")):
            load_inputs(work, source, window, cutoff)
        temporary_zip = work / output.name
        with ZipFile(temporary_zip, "w", compression=ZIP_DEFLATED) as bundle:
            for name, path in files.items():
                bundle.write(path, name, compress_type=ZIP_STORED if path.suffix == ".parquet" else ZIP_DEFLATED)
        os.replace(temporary_zip, output)
    print(f"SHA-256 {sha256(output)}  {output}", flush=True)


if __name__ == "__main__":
    main()
