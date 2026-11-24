"""Bound uncalibrated interval coverage and WIS from archived member predictions."""

import argparse
import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.calibration.methods import _group_tokens
from skyfare.evaluation.pooled.metrics import distribution_row_loss
from skyfare.evaluation.rivf_revision.evidence import LABEL_FILES, read_table, test_archive, test_members
from skyfare.evaluation.rivf_revision.learned_baselines import Q_COLUMNS
from skyfare.models.selection_contract import QUANTILES

INTERVALS = {50: (2, 4), 80: (1, 5), 90: (0, 6)}


def member_artifacts(path):
    receipts, predictions = {}, {}
    with tarfile.open(path, "r|gz") as archive:
        for item in archive:
            if not item.isfile() or "/jobs/D24_" not in item.name:
                continue
            job, name = item.name.split("/")[-2:]
            if name == "done.json":
                receipts[job] = json.load(archive.extractfile(item))
            elif name == "predictions.parquet":
                predictions[job] = archive.extractfile(item).read()
    if receipts.keys() != predictions.keys() or len(receipts) != 6:
        raise RuntimeError("Expected six distribution members")
    for job, info in sorted(receipts.items()):
        payload = predictions[job]
        if (
            hashlib.sha256(payload).hexdigest() != info["prediction_sha256"]
            or info["status"] != "COMPLETE"
            or info["test_labels_read"]
        ):
            raise RuntimeError(f"Unverified distribution member {job}")
        yield info["calibration"], pd.read_parquet(BytesIO(payload)).set_index("row_key", verify_integrity=True)


def inverse_bounds(frame, state):
    if state["recipe"] != "MONDRIAN_CONFORMAL":
        raise RuntimeError("Unexpected distribution calibration")
    offsets = np.array([state["mondrian_groups"].get(k, state["global_offsets"]) for k in _group_tokens(frame)])
    observed = frame[list(Q_COLUMNS)].to_numpy()
    cap = observed - offsets
    fixed = np.column_stack([np.ones(len(frame), dtype=bool), np.diff(observed, axis=1) > 0])
    lower = cap.copy()
    # A new observed maximum identifies the input. On plateaus the lowest
    # feasible quantile is the preceding one and the upper cap comes from
    # this or any later coordinate under the monotonicity constraint.
    for i in range(1, len(Q_COLUMNS)):
        lower[:, i] = np.where(fixed[:, i], cap[:, i], lower[:, i - 1])
    upper = np.where(fixed, cap, np.minimum.accumulate(cap[:, ::-1], axis=1)[:, ::-1])
    for bound in (lower, upper):
        if (np.diff(bound, axis=1) < -1e-12).any() or not np.allclose(
            np.maximum.accumulate(bound + offsets, axis=1), observed, atol=1e-12, rtol=0,
        ):
            raise RuntimeError("Quantile bounds do not reconstruct the calibrated artifact")
    if not np.isfinite(lower).all() or (lower > upper + 1e-12).any():
        raise RuntimeError("Inconsistent inverse bounds")
    return lower, upper, observed


def row_scores(archives, block):
    assets = test_members(archives, block)
    frozen = read_table(assets["DISTRIBUTION"], ["row_key", "label_time", *Q_COLUMNS])
    frozen = frozen.set_index("row_key", verify_integrity=True)
    labels = read_table(assets[LABEL_FILES["REGRESSION"]], ["row_key", "target_anchor_relative_log"])
    labels = labels.set_index("row_key", verify_integrity=True).reindex(frozen.index)
    target = labels["target_anchor_relative_log"].to_numpy()
    if not np.isfinite(target).all():
        raise RuntimeError("Frozen target alignment failed")
    low, high, observed = [], [], []
    for state, frame in member_artifacts(test_archive(archives, block)):
        if len(frame) != len(frozen) or not frame.index.isin(frozen.index).all():
            raise RuntimeError("Member row coverage differs from frozen ensemble")
        l, u, y = inverse_bounds(frame.reindex(frozen.index), state)
        low.append(l)
        high.append(u)
        observed.append(y)
    calibrated = frozen[list(Q_COLUMNS)].to_numpy()
    if not np.allclose(np.mean(observed, axis=0), calibrated, atol=1e-12, rtol=0):
        raise RuntimeError("Archived calibrated ensemble did not reconstruct")
    lower, upper = np.mean(low, axis=0), np.mean(high, axis=0)
    levels = np.asarray(QUANTILES)[None, :]

    def pinball(prediction):
        error = target[:, None] - prediction
        return np.maximum(levels * error, (levels - 1) * error)

    scores = {
        "block": f"TEST_{block}",
        "date": frozen["label_time"].dt.normalize().to_numpy(),
        "wis_calibrated": distribution_row_loss(target, calibrated),
        "wis_without_calibration_lower": 2 * pinball(np.clip(target[:, None], lower, upper)).mean(axis=1),
        "wis_without_calibration_upper": 2 * np.maximum(pinball(lower), pinball(upper)).mean(axis=1),
        "ambiguous_prediction": ((upper - lower) > 1e-10).any(axis=1),
    }
    for level, (left, right) in INTERVALS.items():
        scores[f"coverage{level}_calibrated"] = (
            (target >= calibrated[:, left]) & (target <= calibrated[:, right])
        )
        # Guaranteed inclusion uses the largest possible lower endpoint and
        # the smallest possible upper endpoint. Possible inclusion uses the
        # opposite extrema, jointly feasible by joining a lower-bound prefix
        # to an upper-bound suffix in each monotone member vector.
        scores[f"coverage{level}_without_calibration_lower"] = (
            (target >= upper[:, left]) & (target <= lower[:, right])
        )
        scores[f"coverage{level}_without_calibration_upper"] = (
            (target >= lower[:, left]) & (target <= upper[:, right])
        )
    return pd.DataFrame(scores)


def summary(frame, period):
    dates = frame.groupby(["block", "date"], sort=True).mean(numeric_only=True)
    lower = frame["wis_without_calibration_lower"]
    upper = frame["wis_without_calibration_upper"]
    if (lower > upper + 1e-12).any():
        raise RuntimeError("WIS lower bound exceeds upper bound")
    coverage = {
        column: float(frame[column].mean()) for column in frame if column.startswith("coverage")
    }
    return {
        "period": period, "rows": len(frame), "day_blocks": len(dates),
        "uniquely_recovered_rows": int((~frame["ambiguous_prediction"]).sum()),
        "ambiguous_rows": int(frame["ambiguous_prediction"].sum()),
        "wis_calibrated": float(frame["wis_calibrated"].mean()),
        "wis_without_calibration_lower": float(lower.mean()),
        "wis_without_calibration_upper": float(upper.mean()),
        "equal_day_delta_lower": float((dates["wis_calibrated"] - dates["wis_without_calibration_upper"]).mean()),
        "equal_day_delta_upper": float((dates["wis_calibrated"] - dates["wis_without_calibration_lower"]).mean()),
        **coverage,
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--archives", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    first, second = (row_scores(args.archives, number) for number in (1, 2))
    result = pd.DataFrame([
        summary(first, "TEST_1"), summary(second, "TEST_2"), summary(pd.concat([first, second]), "POOLED"),
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
