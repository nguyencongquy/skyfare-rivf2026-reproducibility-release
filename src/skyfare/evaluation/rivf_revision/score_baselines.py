"""Regenerate the B1 metrics and paired CIs after the Test 2 refit."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.evaluation.rivf_revision.analyse import PERIODS, compare, load_paper_evidence
from skyfare.evaluation.rivf_revision.baseline_return import SCORE_COLUMNS, return_files, verified_predictions
from skyfare.evaluation.rivf_revision.evidence import ROOT, read_json
from skyfare.evaluation.rivf_revision.learned_baselines import TASKS
from skyfare.models.temporal_runtime import sha256

PRIMARY = {"CLASSIFICATION": "proper_score", "POINT": "mape", "DISTRIBUTION": "wis", "RANKING": "ndcg5"}


def join(frame, predictions, context):
    result = frame.merge(predictions, on=["block_id", "row_key"], suffixes=("", "_learned"), validate="one_to_one")
    if len(result) != len(frame) or len(result) != len(predictions):
        raise RuntimeError(f"{context}: prediction IDs do not match the frozen cohort")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--refit", type=Path, required=True)
    parser.add_argument("--previous-evidence", type=Path, default=ROOT / "revision_evidence/previous_b1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    originals, _ = load_paper_evidence(args.archives)
    old_predictions = verified_predictions(args.results, args.bundle)
    old_receipts = json.loads(return_files(args.results)["RESULTS.json"])["tasks"]
    old_metrics = pd.read_csv(args.previous_evidence / "B1_learned_baselines.csv").set_index(["task", "period", "model"])
    old_contrasts = pd.read_csv(args.previous_evidence / "B1_paired_contrasts.csv").set_index(["task", "period"])
    historical = read_json(ROOT / "configs/baseline_metrics.json")
    config = ROOT / "configs/rivf_learned_baselines.json"
    prior_config = ROOT / "configs/rivf_learned_baselines_development_only.json"
    protocol = read_json(config)
    prior_protocol = read_json(prior_config)
    if (
        not protocol["refit_for_test_2"] or prior_protocol["refit_for_test_2"]
        or any(protocol[key] != prior_protocol[key] for key in ("models", "seed", "fit_end_exclusive"))
    ):
        raise RuntimeError("Revised B1 must retain the original model recipes, seed and Test 1 cutoff")
    cutoff = pd.Timestamp(protocol["test_2_fit_end_exclusive"])
    accepted_config_hashes = {sha256(config), sha256(prior_config)}
    args.out.mkdir(parents=True, exist_ok=True)
    metrics, contrasts, receipts, calibration = [], [], {}, []
    for task in TASKS:
        path = args.refit / f"{task.lower()}.parquet"
        receipt = read_json(path.with_suffix(".json"))
        receipt.setdefault("prediction_source", "NEW_LOCAL_TEST_2_REFIT")
        if receipt["prediction_source"] == "VERIFIED_ORIGINAL_TEST_2_REFIT" and "versions" in receipt:
            receipt["verification_versions"] = receipt.pop("versions")
        fit_limits = [pd.Timestamp(receipt[k]) for k in ("fit_label_max", "head_label_max", "cutoff_exclusive")]
        model = protocol["models"][task]
        if (
            receipt["task"] != task or receipt["candidate_id"] != model["candidate_id"]
            or receipt["window"] != model["window"] or receipt["seed"] != protocol["seed"]
            or receipt["config_sha256"] not in accepted_config_hashes
            or not fit_limits[0] < fit_limits[1] < fit_limits[2] == cutoff
            or fit_limits[1] > pd.Timestamp(receipt["first_scoring_feature"])
            or receipt["predictions_sha256"] != sha256(path)
        ):
            raise RuntimeError(f"{task}: invalid refit receipt")
        second = pd.read_parquet(path)
        if (
            set(second.columns) != {"block_id", "row_key", *SCORE_COLUMNS[task]}
            or set(second["block_id"]) != {"TEST_2"} or len(second) != receipt["scored_rows"]
            or not np.isfinite(second[list(SCORE_COLUMNS[task])].to_numpy()).all()
        ):
            raise RuntimeError(f"{task}: invalid refit predictions")
        old = old_predictions[task]
        first = old.loc[old["block_id"].eq("TEST_1")]
        new = {"TEST_1": first, "TEST_2": second, "POOLED": pd.concat([first, second], ignore_index=True)}
        receipts[task] = receipt
        calibration.append({
            "task": task, "candidate_id": model["candidate_id"], "window": model["window"],
            "seed": receipt["seed"],
            "test_2_prediction_source": receipt["prediction_source"],
            "test_1_calibration": old_receipts[task]["model"].get("calibration", {}).get("recipe", "NONE"),
            "test_2_calibration": receipt["model"].get("calibration", {}).get("recipe", "NONE"),
            "test_2_fit_rows": receipt["fit_rows"], "test_2_head_rows": receipt["head_rows"],
            "test_2_fit_label_max": receipt["fit_label_max"], "test_2_head_label_max": receipt["head_label_max"],
        })
        primary = PRIMARY[task]
        for period in PERIODS:
            original = originals[task][period]
            prior = old if period == "POOLED" else old.loc[old.block_id.eq(period)]
            previous_metrics, previous_ci = compare(task, join(original, prior, f"{task}/{period}/original"), historical)
            expected_ci = old_contrasts.loc[(task, period)]
            for key, value in previous_ci.items():
                if not np.isclose(value, expected_ci[key], atol=1e-11, rtol=0):
                    raise RuntimeError(f"{task}/{period}: original CI did not reproduce for {key}")
            for label, values in zip(("ENSEMBLE", "LEARNED", "ORIGINAL"), previous_metrics, strict=True):
                expected = old_metrics.loc[(task, period, label), primary]
                if not np.isclose(values[primary], expected, atol=1e-11, rtol=0):
                    raise RuntimeError(f"{task}/{period}: original metric did not reproduce for {label}")
            values, ci = compare(task, join(original, new[period], f"{task}/{period}/refit"), historical)
            contrasts.append({
                "task": task, "period": period, "metric": primary,
                "ensemble": values[0][primary], "learned": values[1][primary], **ci,
            })
            for label, value in zip(("ENSEMBLE", "LEARNED", "ORIGINAL"), values, strict=True):
                metrics.append({"task": task, "period": period, "model": label, **value})
    pd.DataFrame(metrics).to_csv(args.out / "B1_learned_baselines.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.out / "B1_paired_contrasts.csv", index=False)
    pd.DataFrame(calibration).to_csv(args.out / "B1_refit_calibration_and_cutoffs.csv", index=False)
    (args.out / "B1_refit_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
