"""Audit paired development scores, isotonic ties, and development targets."""

import argparse
import json
import tarfile
from hashlib import sha256 as digest
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.calibration.methods import sigmoid
from skyfare.evaluation.metrics import classification_metrics
from skyfare.evaluation.prospective_test_two.contract import FINALISTS
from skyfare.evaluation.rivf_revision.evidence import (
    LABEL_FILES, ROOT, read_json, read_table, test_archive, test_members, verified_development,
)
from skyfare.models.selection_contract import LATE_SEEDS, SCREEN_FOLDS, SCREEN_SEEDS
from skyfare.models.temporal_runtime import sha256


def archived_members(path, paths):
    found = {}
    with tarfile.open(path, "r|gz") as archive:
        for item in archive:
            if item.name not in paths:
                continue
            if not item.isfile() or item.name in found:
                raise RuntimeError(f"Unexpected archive entry: {item.name}")
            found[item.name] = archive.extractfile(item).read()
    if set(found) != set(paths):
        raise RuntimeError(f"Missing archive entries: {set(paths) - set(found)}")
    return found


def development_scores(path, out):
    protocol = read_json(ROOT / "configs/rivf_learned_baselines.json")
    baseline = protocol["models"]["CLASSIFICATION"]["candidate_id"]
    finalists = FINALISTS["CLASSIFICATION"]
    candidates = [*finalists, baseline]
    manifest = read_json(ROOT / "artifacts/release_manifests/development_selection_results.json")
    if sha256(path) != manifest["archive_sha256"]:
        raise RuntimeError("Unexpected development archive checksum")
    paths = {
        f"v24_refinement_outputs/jobs/{c}__{f}__S{s}/{name}"
        for c in candidates for f in SCREEN_FOLDS for s in SCREEN_SEEDS
        for name in ("done.json", "predictions.parquet")
    }
    selection_path = "v24_refinement_outputs/analysis/classification_selection_v24.json"
    paths.add(selection_path)
    assets = archived_members(path, paths)
    for name, payload in assets.items():
        if digest(payload).hexdigest() != manifest["files"][name]:
            raise RuntimeError(f"Development archive member failed hash check: {name}")
    rows, folds = [], []
    for fold in SCREEN_FOLDS:
        ensembles = []
        for seed in SCREEN_SEEDS:
            predictions = []
            for candidate in candidates:
                prefix = f"v24_refinement_outputs/jobs/{candidate}__{fold}__S{seed}"
                receipt = json.loads(assets[f"{prefix}/done.json"])
                payload = assets[f"{prefix}/predictions.parquet"]
                if receipt["status"] != "COMPLETE" or receipt["job"]["phase"] != "SCREEN" or (
                    digest(payload).hexdigest() != receipt["prediction_sha256"]
                ):
                    raise RuntimeError(f"Unverified development prediction: {prefix}")
                frame = read_table(payload, ["row_key", "target", "probability"])
                frame = frame.set_index("row_key", verify_integrity=True).sort_index()
                if len(frame) != receipt["rows"] or (
                    predictions and not frame["target"].equals(predictions[0]["target"])
                ):
                    raise RuntimeError(f"Unpaired development rows: {prefix}")
                score = classification_metrics(frame["target"], frame["probability"])["proper_score"]
                if not np.isclose(score, receipt["metrics"]["proper_score"], atol=1e-12, rtol=0):
                    raise RuntimeError(f"Archived proper score did not reproduce: {prefix}")
                rows.append({"fold": fold, "seed": seed, "model": candidate, "rows": len(frame), "proper_score": score})
                predictions.append(frame)
            first, second, _ = predictions
            probability = (first["probability"] + second["probability"]) / 2
            score = classification_metrics(first["target"], probability)["proper_score"]
            rows.append({"fold": fold, "seed": seed, "model": "V24_MEAN", "rows": len(first), "proper_score": score})
            ensembles.append(pd.DataFrame({"target": first["target"], "probability": probability}))
        first, second = ensembles
        if not first["target"].equals(second["target"]):
            raise RuntimeError(f"Unpaired development seed rows: {fold}")
        score = classification_metrics(first["target"], (first["probability"] + second["probability"]) / 2)["proper_score"]
        folds.append({"fold": fold, "model": "V24_MEAN_seed_averaged", "rows": len(first), "proper_score": score})
    detail = pd.DataFrame(rows)
    summary = detail.groupby("model", sort=False, as_index=False).agg(
        proper_score=("proper_score", "mean"), fold_seed_pairs=("proper_score", "size")
    )
    recipes = json.loads(assets[selection_path])["recipes"]
    v24_mean = next(recipe for recipe in recipes if recipe["recipe"] == "V24_MEAN")
    for row in folds:
        if not np.isclose(row["proper_score"], v24_mean["fold_metrics"][row["fold"]]["proper_score"], atol=1e-12, rtol=0):
            raise RuntimeError("Seed-averaged V24 recipe does not reproduce the archive")
    detail.to_csv(out / "B1_development_proper_score_by_fold_seed.csv", index=False)
    summary.to_csv(out / "B1_development_proper_score_summary.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "B1_development_v24_frozen_recipe.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def auc_mass(groups, score, target):
    """Credit and positive-negative pairs within equal values of groups."""
    order = np.lexsort((score, groups))
    group, score, target = groups[order], score[order], target[order]
    starts = np.flatnonzero(np.r_[True, (group[1:] != group[:-1]) | (score[1:] != score[:-1])])
    counts = np.diff(np.r_[starts, len(order)])
    positive = np.add.reduceat(target.astype(np.int64), starts)
    negative = counts - positive
    group = group[starts]
    boundaries = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    group_for_score = np.cumsum(np.r_[True, group[1:] != group[:-1]]) - 1
    before = np.cumsum(negative) - negative
    prior_in_group = before - before[boundaries][group_for_score]
    credit_twice = np.sum(positive * (2 * prior_in_group + negative))
    pairs = np.sum(np.add.reduceat(positive, boundaries) * np.add.reduceat(negative, boundaries))
    return int(credit_twice), int(pairs)


def auc_details(target, raw, calibrated):
    raw = np.clip(np.asarray(raw), 1e-7, 1 - 1e-7)
    calibrated = np.clip(np.asarray(calibrated), 1e-7, 1 - 1e-7)
    group = np.zeros(len(target), dtype=int)
    raw_mass, pairs = auc_mass(group, raw, target)
    cal_mass, _ = auc_mass(group, calibrated, target)
    raw_tie_cal_mass, raw_ties = auc_mass(raw, calibrated, target)
    cal_tie_raw_mass, cal_ties = auc_mass(calibrated, raw, target)
    tie_change = raw_tie_cal_mass - raw_ties + cal_ties - cal_tie_raw_mass
    return {
        "rows": len(target), "positive_negative_pairs": pairs,
        "auc_uncalibrated": raw_mass / (2 * pairs), "auc_calibrated": cal_mass / (2 * pairs),
        "raw_tie_pairs": raw_ties, "calibrated_tie_pairs": cal_ties,
        "tie_delta_auc": tie_change / (2 * pairs),
        "strict_order_delta_auc": (cal_mass - raw_mass - tie_change) / (2 * pairs),
        "auc_delta": (cal_mass - raw_mass) / (2 * pairs),
        "twice_auc_credit_change": cal_mass - raw_mass,
    }


def isotonic_audit(archives, out):
    results, periods = [], {}
    for block in (1, 2):
        period = f"TEST_{block}"
        jobs = [f"{c}__{period}__S{s}" for c in FINALISTS["CLASSIFICATION"] for s in LATE_SEEDS]
        prefix = f"test{block}_outputs/jobs"
        paths = {f"{prefix}/{job}/{name}" for job in jobs for name in ("done.json", "predictions.parquet")}
        inputs = archived_members(test_archive(archives, block), paths)
        verified = test_members(archives, block)
        frozen = read_table(verified["CLASSIFICATION"], ["row_key", "probability"]).set_index("row_key", verify_integrity=True)
        labels = read_table(verified[LABEL_FILES["CLASSIFICATION"]], ["row_key", "DROP_5PCT"])
        labels = labels.set_index("row_key", verify_integrity=True).reindex(frozen.index)
        target = labels["DROP_5PCT"].to_numpy()
        if len(target) != len(frozen) or set(np.unique(target)) != {0, 1}:
            raise RuntimeError(f"Invalid classification labels: {period}")
        raw_scores, calibrated_scores = [], []
        for job in jobs:
            receipt = json.loads(inputs[f"{prefix}/{job}/done.json"])
            payload = inputs[f"{prefix}/{job}/predictions.parquet"]
            if receipt["status"] != "COMPLETE" or receipt["job"]["job_id"] != job or (
                receipt["test_labels_read"] or receipt["prediction_sha256"] != digest(payload).hexdigest()
            ):
                raise RuntimeError(f"Invalid scoring receipt: {job}")
            frame = read_table(payload, ["row_key", "probability", "raw_score"]).set_index("row_key", verify_integrity=True)
            if len(frame) != len(frozen):
                raise RuntimeError(f"Unexpected row count: {job}")
            frame = frame.reindex(frozen.index)
            if not np.isfinite(frame.to_numpy()).all():
                raise RuntimeError(f"Unmatched prediction rows: {job}")
            raw, cal = sigmoid(frame["raw_score"]), frame["probability"].to_numpy()
            raw_scores.append(raw)
            calibrated_scores.append(cal)
            results.append({
                "period": period, "model": job, "recipe": receipt["calibration"]["recipe"], **auc_details(target, raw, cal),
            })
        raw, cal = np.mean(raw_scores, axis=0), np.mean(calibrated_scores, axis=0)
        if not np.allclose(cal, frozen["probability"], atol=1e-15, rtol=0):
            raise RuntimeError(f"Archived scores did not reconstruct the ensemble: {period}")
        periods[period] = (target, raw, cal)
        results.append({"period": period, "model": "ENSEMBLE", "recipe": "MEMBER_CALIBRATION", **auc_details(target, raw, cal)})
    target, raw, cal = (np.concatenate(parts) for parts in zip(*periods.values(), strict=True))
    pooled = auc_details(target, raw, cal)
    results.append({"period": "POOLED", "model": "ENSEMBLE", "recipe": "MEMBER_CALIBRATION", **pooled})
    pd.DataFrame(results).to_csv(out / "N2_auc_calibration_and_ties.csv", index=False)
    within = [row for row in results if row["model"] == "ENSEMBLE" and row["period"] != "POOLED"]
    change = pooled["twice_auc_credit_change"] - sum(row["twice_auc_credit_change"] for row in within)
    pairs = pooled["positive_negative_pairs"] - sum(row["positive_negative_pairs"] for row in within)
    pd.DataFrame([{
        "pooled_pairs": pooled["positive_negative_pairs"], "cross_block_pairs": pairs,
        "cross_block_pair_weight": pairs / pooled["positive_negative_pairs"],
        "cross_block_auc_change": change / (2 * pairs),
        "cross_block_contribution_to_pooled_delta": change / (2 * pooled["positive_negative_pairs"]),
    }]).to_csv(out / "N2_auc_cross_block_contribution.csv", index=False)
    print(pd.DataFrame([*within, results[-1]]).to_string(index=False), flush=True)


def development_targets(directory, out):
    filename = "regression_training_frame.parquet"
    frame_path = verified_development(directory, filename)
    offers_path = directory / "standard_128_day_offers.parquet"
    development = read_json(ROOT / "artifacts/evidence/development_selection/preflight.json")
    if sha256(offers_path) != development["data"]["offers_sha256"]:
        raise RuntimeError("Development offers do not match preflight")
    cols = [
        "target_offer_id", "target_session_price_vnd", "query_session_observed_fare_vnd",
        "target_observation_state", "dud_support_mode", "candidate_source",
    ]
    targets = pd.read_parquet(frame_path, columns=cols)
    offers = pd.read_parquet(offers_path, columns=["offer_id", "price_vnd"])
    target = targets.merge(
        offers, left_on="target_offer_id", right_on="offer_id", how="left", validate="many_to_one", indicator=True,
    )
    exact = (
        target["target_observation_state"].eq("OBSERVED") & target["dud_support_mode"].eq("ON_GRID")
        & target["_merge"].eq("both") & target["target_session_price_vnd"].eq(target["price_vnd"])
        & target["target_session_price_vnd"].eq(target["query_session_observed_fare_vnd"])
    )
    interpolated = target["dud_support_mode"].eq("INTERIOR_OFF_GRID_INTERPOLATED")
    if not (exact ^ interpolated).all():
        raise RuntimeError("Unclassified or contradictory target provenance")
    result = pd.DataFrame([{
        "period": "DEVELOPMENT", "rows": len(target), "exact": int(exact.sum()),
        "interpolated": int(interpolated.sum()), "interpolated_fraction": float(interpolated.mean()),
        "frame_sha256": sha256(frame_path), "observed_offers_sha256": sha256(offers_path),
    }])
    result.to_csv(out / "B2_development_exact_targets.csv", index=False)
    print(result.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", choices=["scores", "auc", "targets"], required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.check == "scores":
        development_scores(args.archives / "SKYFARE_V24_FINAL_REFINEMENT_FREEZE_RESULTS_R1.tar.gz", args.out)
    elif args.check == "auc":
        isotonic_audit(args.archives, args.out)
    else:
        development_targets(args.development, args.out)


if __name__ == "__main__":
    main()
