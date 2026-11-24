"""Compute the recommended N1 and N2 checks from frozen study predictions."""

from __future__ import annotations

import argparse
import json
import tarfile
from hashlib import sha256 as digest
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.calibration.methods import sigmoid
from skyfare.evaluation.pooled.metrics import daily_stratified_bootstrap, policy_metrics
from skyfare.evaluation.rivf_revision.analyse import PERIODS, compare, load_paper_evidence
from skyfare.evaluation.rivf_revision.baseline_return import SCORE_COLUMNS
from skyfare.evaluation.rivf_revision.evidence import (
    CONTRACT,
    FEATURE_FILES,
    ROOT,
    read_json,
    read_table,
    test_archive,
    test_members,
)
from skyfare.models.selection_contract import LATE_SEEDS, SCREEN_FOLDS, SCREEN_SEEDS
from skyfare.models.temporal_runtime import sha256

PRIMARY = {"CLASSIFICATION": "proper_score", "POINT": "mape", "DISTRIBUTION": "wis", "RANKING": "ndcg5"}


def pick_single_members(path: Path, finalists: dict) -> dict[str, dict]:
    manifest = read_json(ROOT / "artifacts/release_manifests/development_selection_results.json")
    if sha256(path) != manifest["archive_sha256"]:
        raise RuntimeError("Development selection archive hash mismatch")
    expected = {
        f"v24_refinement_outputs/jobs/{candidate}__{fold}__S{seed}/done.json": (task, candidate)
        for task, candidates in finalists.items()
        for candidate in candidates
        for fold in SCREEN_FOLDS
        for seed in SCREEN_SEEDS
    }
    scores = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name not in expected:
                continue
            data = json.load(archive.extractfile(member))
            task, candidate = expected[member.name]
            if data["status"] != "COMPLETE" or data["job"]["phase"] != "SCREEN":
                raise RuntimeError(f"Incomplete development scoring: {member.name}")
            scores[member.name] = (task, candidate, float(data["metrics"][PRIMARY[task]]))
    if scores.keys() != expected.keys():
        raise RuntimeError("Missing selection fold/seed metrics")
    values = pd.DataFrame(scores.values(), columns=["task", "candidate", "metric"])
    selected = {}
    for task in PRIMARY:
        means = values.loc[values["task"].eq(task)].groupby("candidate")["metric"].mean()
        winner = means.sort_values(ascending=task != "RANKING", kind="stable").index[0]
        selected[task] = {"candidate": winner, "development_mean": float(means[winner]), "seed": LATE_SEEDS[0]}
    return selected


def read_saved_jobs(archives: Path, number: int, finalists: dict, selected: dict) -> dict:
    block = f"TEST_{number}"
    prefix = f"test{number}_outputs/jobs"
    jobs = {
        f"{candidate}__{block}__S{seed}": task
        for task, candidates in finalists.items()
        for candidate in candidates
        for seed in LATE_SEEDS
        if task == "CLASSIFICATION"
        or (candidate == selected[task]["candidate"] and seed == selected[task]["seed"])
    }
    paths = {
        f"{prefix}/{job}/{name}": (job, name)
        for job in jobs
        for name in ("done.json", "predictions.parquet")
    }
    found = {}
    with tarfile.open(test_archive(archives, number), "r|gz") as archive:
        for member in archive:
            if member.name in paths:
                if not member.isfile():
                    raise RuntimeError(f"Invalid member: {member.name}")
                found[paths[member.name]] = archive.extractfile(member).read()
    if len(found) != len(paths):
        raise RuntimeError(f"Missing archived seed predictions in {block}")
    frames = {}
    for job, task in jobs.items():
        receipt = json.loads(found[job, "done.json"])
        payload = found[job, "predictions.parquet"]
        if (
            receipt["status"] != "COMPLETE"
            or receipt["job"]["job_id"] != job
            or receipt["test_labels_read"] is not False
            or digest(payload).hexdigest() != receipt["prediction_sha256"]
        ):
            raise RuntimeError(f"Invalid prediction receipt for {job}")
        columns = ["row_key", *SCORE_COLUMNS[task]]
        if task == "CLASSIFICATION":
            columns.append("raw_score")
        frame = read_table(payload, columns)
        if len(frame) != receipt["rows"] or frame["row_key"].duplicated().any():
            raise RuntimeError(f"Invalid archived prediction keys for {job}")
        frames[job] = frame.set_index("row_key")
    return frames


def align(frame: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    values = frame.reindex(keys).reset_index(drop=True)
    if len(frame) != len(keys) or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError("Saved job does not match the frozen evaluation cohort")
    return values


def classification_inputs(archives: Path, number: int, original: pd.DataFrame) -> pd.DataFrame:
    members = test_members(archives, number, features_only=True)
    frame = read_table(members[FEATURE_FILES["CLASSIFICATION"]])
    source_column = "current_dud" if "current_dud" in frame else "days_until_departure"
    current = frame[["row_key", source_column]].rename(columns={source_column: "current_dud"})
    merged = original.merge(current, on="row_key", validate="one_to_one")
    if len(merged) != len(original) or not merged["current_dud"].gt(merged["target_dud"]).all():
        raise RuntimeError("Invalid source and target booking windows")
    return merged


def threshold_rows(classifications: dict, contract: dict) -> list[dict]:
    rows = []
    friction = contract["policy"]["friction_vnd"]
    nominal = contract["policy"]["threshold"]
    thresholds = (round(nominal - 0.05, 2), nominal, round(nominal + 0.05, 2))
    for block, frame in classifications.items():
        for threshold in thresholds:
            groups = [("ALL", frame)]
            groups.extend(
                (f"{int(source)}→{int(target)}", cohort)
                for (source, target), cohort in frame.groupby(["current_dud", "target_dud"], sort=False)
            )
            for window, cohort in groups:
                policy, observed = policy_metrics(cohort, "FROZEN", threshold, friction)
                buy, baseline = policy_metrics(cohort, "BUY_ALL", threshold, friction)
                delta = observed - baseline
                row = {
                    "period": block, "booking_window": window, "threshold": threshold, "friction_vnd": friction,
                    "rows": len(cohort), "wait_rate": policy["wait_rate"],
                    "policy_mean_regret_vnd": policy["mean_regret_vnd"], "buy_mean_regret_vnd": buy["mean_regret_vnd"],
                    "policy_cvar95_vnd": policy["cvar95_regret_vnd"], "buy_cvar95_vnd": buy["cvar95_regret_vnd"],
                    "row_weighted_delta_vnd": float(delta.mean()),
                }
                if window == "ALL":
                    ci, _ = daily_stratified_bootstrap(delta, cohort["feature_time"], cohort["block_id"], 43005)
                    row.update(ci)
                rows.append(row)
    return rows


def paired_rows(evidence: dict, selected: dict, finalists: dict, archived_jobs: dict) -> tuple[list[dict], list[dict]]:
    historical = read_json(ROOT / "configs/baseline_metrics.json")
    ablations = {}
    for task, block_inputs in evidence.items():
        comparisons = {"SINGLE_MEMBER": {}}
        if task == "CLASSIFICATION":
            comparisons["NO_CALIBRATION"] = {}
        for block in PERIODS[:2]:
            frame = block_inputs[block]
            jobs = archived_jobs[block]
            member = selected[task]
            name = f"{member['candidate']}__{block}__S{member['seed']}"
            values = align(jobs[name], frame["row_key"])
            columns = SCORE_COLUMNS[task]
            comparisons["SINGLE_MEMBER"][block] = frame.assign(
                **{f"{column}_learned": values[column].to_numpy() for column in columns}
            )
            if task == "CLASSIFICATION":
                aligned = [
                    align(jobs[f"{candidate}__{block}__S{seed}"], frame["row_key"])
                    for candidate in finalists[task]
                    for seed in LATE_SEEDS
                ]
                calibrated = np.mean([job["probability"].to_numpy() for job in aligned], axis=0)
                if not np.allclose(calibrated, frame["probability"], atol=1e-15, rtol=0):
                    raise RuntimeError(f"{block}: failed to reconstruct the frozen calibrated ensemble")
                probabilities = np.mean([sigmoid(job["raw_score"]) for job in aligned], axis=0)
                comparisons["NO_CALIBRATION"][block] = frame.assign(probability_learned=probabilities)
        for name, periods in comparisons.items():
            ablations[task, name] = {**periods, "POOLED": pd.concat(list(periods.values()), ignore_index=True)}

    metrics, contrasts = [], []
    for (task, comparison), periods in ablations.items():
        for block, frame in periods.items():
            values, ci = compare(task, frame, historical)
            key = {"task": task, "comparison": comparison, "period": block}
            contrasts.append({**key, "metric": PRIMARY[task], **ci})
            for model, value in zip(("ENSEMBLE", comparison), values[:2], strict=True):
                metrics.append({**key, "model": model, **value})
    return metrics, contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--development-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    recipe = read_json(ROOT / "artifacts/evidence/development_selection/finalist_recipe_lock.json")
    finalists = {task: recipe["tasks"][task]["finalists"] for task in PRIMARY}
    selected = pick_single_members(args.development_results, finalists)
    evidence, _ = load_paper_evidence(args.archives)
    classification = {
        f"TEST_{i}": classification_inputs(args.archives, i, evidence["CLASSIFICATION"][f"TEST_{i}"])
        for i in (1, 2)
    }
    classification["POOLED"] = pd.concat(list(classification.values()), ignore_index=True)
    contract = read_json(CONTRACT)
    thresholds = threshold_rows(classification, contract)
    nominal = next(row for row in thresholds if
                   row["period"] == "POOLED" and row["booking_window"] == "ALL"
                   and row["threshold"] == contract["policy"]["threshold"])
    frozen = read_json(ROOT / "artifacts/evidence/pooled_evaluation_report.json")["stratified_day_bootstrap"]["policy_regret"]
    for key in ("mean_model_minus_baseline", "ci95_low", "ci95_high"):
        if not np.isclose(nominal[key], frozen[key], atol=1e-8, rtol=0):
            raise RuntimeError(f"Could not reproduce frozen threshold result: {key}")
    jobs = {f"TEST_{i}": read_saved_jobs(args.archives, i, finalists, selected) for i in (1, 2)}
    metrics, contrasts = paired_rows(evidence, selected, finalists, jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(thresholds).to_csv(args.out / "N1_threshold_sensitivity.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.out / "N2_ablation_metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.out / "N2_ablation_contrasts.csv", index=False)
    pd.DataFrame.from_dict(selected, orient="index").rename_axis("task").to_csv(args.out / "N2_development_member_selection.csv")
    print(f"Verified N1 and N2 evidence saved in {args.out}")


if __name__ == "__main__":
    main()
