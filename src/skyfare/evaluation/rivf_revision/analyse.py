"""Produce evidence tables for the RIVF reviewer responses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from skyfare.evaluation.metrics import point_metrics
from skyfare.evaluation.pooled.finalize import join_prediction
from skyfare.evaluation.pooled.metrics import (
    classification_metrics,
    daily_stratified_bootstrap,
    distribution_metrics,
    distribution_row_loss,
    policy_metrics,
    ranking_query_metrics,
)
from skyfare.evaluation.prospective_test_two.prepare import _session_semantics
from skyfare.evaluation.rivf_revision.baseline_return import verified_predictions
from skyfare.evaluation.rivf_revision.evidence import (
    CONTRACT,
    FEATURE_FILES,
    LABEL_FILES,
    ROOT,
    read_json,
    read_table,
    test_members,
)
from skyfare.evaluation.rivf_revision.learned_baselines import Q_COLUMNS, TASKS
from skyfare.features.audit_common import completed_batch_offers
from skyfare.models.temporal_runtime import sha256, write_json_atomic
from skyfare.policy.buy_wait import FRICTIONS_VND
from skyfare.preparation import merge_daily_offers, temporal_sessions

PERIODS = ("TEST_1", "TEST_2", "POOLED")
POLICIES = ("FROZEN", "BUY_ALL", "WAIT_ALL")
RAW_KEYS = ["route", "airline", "flight_date", "departure_time", "session_date", "session_label", "price_vnd"]


def load_paper_evidence(archives: Path) -> tuple[dict, dict]:
    results = {task: {} for task in TASKS}
    regression_inputs = {}
    for number in (1, 2):
        block = f"TEST_{number}"
        members = test_members(archives, number)
        regression_inputs[block] = read_table(
            members[FEATURE_FILES["REGRESSION"]],
            ["row_key", "target_offer_id", "target_observation_state", "dud_support_mode", *RAW_KEYS],
        )
        for task in TASKS:
            source = "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION"
            labels = read_table(members[LABEL_FILES[source]])
            prediction = read_table(members[task])
            results[task][block] = join_prediction(prediction, labels).assign(block_id=block)
    for periods in results.values():
        periods["POOLED"] = pd.concat([periods[block] for block in PERIODS[:2]], ignore_index=True)
    return results, regression_inputs


def audit_exact(points: dict, regression_inputs: dict, raw_root: Path, contract: dict) -> list[dict]:
    raw_manifest = read_json(ROOT / "data/manifests/raw_collection_manifest.json")["files"]
    checks = {}
    for block in PERIODS[:2]:
        point = points[block]
        frame = point.merge(regression_inputs[block], on="row_key", validate="one_to_one", suffixes=("", "_frame"))
        if len(frame) != len(point) or not frame["target_offer_id"].eq(frame["target_offer_id_frame"]).all():
            raise RuntimeError(f"{block}: scored offer identity mismatch")
        dates = pd.date_range(*contract["expected_blocks"][block.lower()])
        daily_files = [raw_root / f"{day:%Y-%m-%d}.csv" for day in dates]
        for file in daily_files:
            if sha256(file) != raw_manifest[f"data/raw/trip_com/{file.name}"]["sha256"]:
                raise RuntimeError(f"Raw snapshot failed SHA-256 check: {file.name}")
        raw = pd.concat([pd.read_csv(file) for file in daily_files], ignore_index=True)
        raw = raw.loc[merge_daily_offers.assign_price_tier(raw).eq("standard")]
        raw = completed_batch_offers(_session_semantics(raw, temporal_sessions))
        for column in ("flight_date", "departure_time", "session_date"):
            frame[column] = frame[column].astype("datetime64[ns]")
            raw[column] = raw[column].astype("datetime64[ns]")
        matches = frame.merge(raw[RAW_KEYS].drop_duplicates().assign(raw_match=True), on=RAW_KEYS, how="left", validate="many_to_one")
        exact = (
            matches["raw_match"].eq(True)
            & matches["target_observation_state"].eq("OBSERVED")
            & matches["dud_support_mode"].eq("ON_GRID")
            & matches["price_vnd"].eq(matches["target_session_price_vnd"])
        )
        if not exact.all():
            raise RuntimeError(f"{block}: {int((~exact).sum())} targets lack exact provenance")
        checks[block] = matches
    checks["POOLED"] = pd.concat(list(checks.values()), ignore_index=True)
    return [
        {
            "period": block, "rows": len(frame), "exact": len(frame), "interpolated": 0,
            "ensemble_mape": point_metrics(frame["target_session_price_vnd"], frame["predicted_price_vnd"])["mape"],
            "prior_anchor_mape": point_metrics(frame["target_session_price_vnd"], frame["prior_anchor_vnd"])["mape"],
        }
        for block, frame in checks.items()
    ]


def policy_tables(classification: dict, contract: dict) -> tuple[list[dict], dict]:
    threshold = contract["policy"]["threshold"]
    nominal = contract["policy"]["friction_vnd"]
    report = []
    for block, frame in classification.items():
        for friction in FRICTIONS_VND:
            policy = {}
            regrets = {}
            for action in POLICIES:
                policy[action], regrets[action] = policy_metrics(frame, action, threshold, friction)
            delta = regrets["FROZEN"] - regrets["BUY_ALL"]
            interval, dates = daily_stratified_bootstrap(
                delta, frame["feature_time"], frame["block_id"], 43005,
                include_standard_error=True,
            )
            row = {"period": block, "friction_vnd": friction, "threshold": threshold, **interval}
            for action, metrics in policy.items():
                row.update({f"{action.lower()}_{key}": value for key, value in metrics.items()})
            report.append(row)
            if block == "POOLED" and friction == nominal:
                always_buy_daily = (
                    frame.assign(date=frame["feature_time"].dt.normalize(), regret=regrets["BUY_ALL"])
                    .groupby(["block_id", "date"], sort=True)["regret"].mean()
                )
                mde = (norm.ppf(0.975) + norm.ppf(0.80)) * interval["bootstrap_se"]
                detectable = {
                    **interval, "alpha_two_sided": 0.05, "power": 0.80, "friction_vnd": nominal,
                    "eligible_dates_per_test": dates.groupby("block_id").size().to_dict(),
                    "always_buy_equal_day_mean_vnd": float(always_buy_daily.mean()),
                    "mde_vnd": float(mde), "mde_relative": float(mde / always_buy_daily.mean()),
                }
    return report, detectable


def classification_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p)) + np.square(y - p)


def compare(task: str, frame: pd.DataFrame, historical: dict) -> tuple[list[dict], dict]:
    blocks = frame["block_id"].to_numpy()
    if task == "CLASSIFICATION":
        target = frame["DROP_5PCT"].to_numpy(dtype=int)
        old = frame["block_id"].map(historical["classification_constant_probability"]).to_numpy()
        predictions = (frame["probability"].to_numpy(), frame["probability_learned"].to_numpy(), old)
        metrics = [classification_metrics(target, prediction) for prediction in predictions]
        losses = [classification_loss(target, prediction) for prediction in predictions[:2]]
        dates = frame["feature_time"]
    elif task == "POINT":
        target = frame["target_session_price_vnd"].to_numpy()
        predictions = [frame[column].to_numpy() for column in ("predicted_price_vnd", "predicted_price_vnd_learned", "prior_anchor_vnd")]
        metrics = [point_metrics(target, prediction) for prediction in predictions]
        losses = [np.abs(prediction - target) / target for prediction in predictions[:2]]
        dates = frame["label_time"]
    elif task == "DISTRIBUTION":
        target = frame["target_anchor_relative_log"].to_numpy()
        old = np.asarray([historical["distribution_empirical_quantiles"][block] for block in blocks])
        predictions = (frame[list(Q_COLUMNS)].to_numpy(), frame[[f"{q}_learned" for q in Q_COLUMNS]].to_numpy(), old)
        metrics = [distribution_metrics(target, prediction) for prediction in predictions]
        losses = [distribution_row_loss(target, prediction) for prediction in predictions[:2]]
        dates = frame["label_time"]
    elif task == "RANKING":
        data = frame.rename(columns={"target_session_price_vnd": "target_price_vnd"})
        queries = [
            ranking_query_metrics(data, prediction)
            for prediction in (data["score"], data["score_learned"], -data["prior_anchor_vnd"])
        ]
        metrics = [
            {"ndcg5": float(query["ndcg5"].mean()), "concordance": float(query["pairwise_concordance"].mean()), "queries": len(query)}
            for query in queries
        ]
        losses = [-query["ndcg5"].to_numpy() for query in queries[:2]]
        dates = queries[0]["label_time"]
        blocks = queries[0]["block_id"].to_numpy()
    else:
        raise KeyError(task)
    seed = {"CLASSIFICATION": 43001, "POINT": 43002, "DISTRIBUTION": 43003, "RANKING": 43004}[task]
    ci, _ = daily_stratified_bootstrap(losses[0] - losses[1], dates, pd.Series(blocks), seed)
    if task == "RANKING":
        # Ranking results report NDCG gain, so reverse the loss contrast and its endpoints.
        ci["mean_model_minus_baseline"] *= -1
        ci["ci95_low"], ci["ci95_high"] = -ci["ci95_high"], -ci["ci95_low"]
    return metrics, ci


def learned_tables(evidence: dict, learned: Path, bundle: Path) -> tuple[list[dict], list[dict]]:
    predictions = verified_predictions(learned, bundle)
    historical = read_json(ROOT / "configs/baseline_metrics.json")
    metrics = []
    contrasts = []
    primary = {"CLASSIFICATION": "proper_score", "POINT": "mape", "DISTRIBUTION": "wis", "RANKING": "ndcg5"}
    for task in TASKS:
        prediction = predictions[task]
        for block, original in evidence[task].items():
            current = prediction if block == "POOLED" else prediction.loc[prediction["block_id"].eq(block)]
            frame = original.merge(current, on=["block_id", "row_key"], validate="one_to_one", suffixes=("", "_learned"))
            if len(frame) != len(original) or len(frame) != len(current):
                raise RuntimeError(f"{task}/{block}: baseline row coverage differs from paper")
            values, ci = compare(task, frame, historical)
            contrasts.append({"task": task, "period": block, "metric": primary[task], **ci})
            for model, value in zip(("ENSEMBLE", "LEARNED", "ORIGINAL"), values, strict=True):
                metrics.append({"task": task, "period": block, "model": model, **value})
    return metrics, contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--raw-trip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--learned-results", type=Path, help="Vast results ZIP or its extracted results directory")
    parser.add_argument("--training-bundle", type=Path)
    args = parser.parse_args()
    if (args.learned_results is None) != (args.training_bundle is None):
        parser.error("--learned-results and --training-bundle must be provided together")

    contract = read_json(CONTRACT)
    evidence, staged = load_paper_evidence(args.archives)
    exact = audit_exact(evidence["POINT"], staged, args.raw_trip, contract)
    stress, mde = policy_tables(evidence["CLASSIFICATION"], contract)
    frozen = read_json(ROOT / "artifacts/evidence/pooled_evaluation_report.json")
    for name in ("mean_model_minus_baseline", "ci95_low", "ci95_high", "day_blocks"):
        if not np.isclose(mde[name], frozen["stratified_day_bootstrap"]["policy_regret"][name], atol=1e-8, rtol=0):
            raise RuntimeError(f"Nominal policy contrast drift: {name}")
    for name, key in (("ensemble_mape", "model"), ("prior_anchor_mape", "baseline")):
        if not np.isclose(exact[-1][name], frozen["pooled"]["point"][key]["mape"], atol=1e-10, rtol=0):
            raise RuntimeError(f"Exact-only MAPE differs from the archived result: {name}")
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exact).to_csv(args.out / "B2_exact_targets.csv", index=False)
    write_json_atomic(args.out / "B4_policy_mde.json", mde)
    pd.DataFrame(stress).to_csv(args.out / "B5_policy_penalty.csv", index=False)
    if args.learned_results:
        metrics, contrasts = learned_tables(evidence, args.learned_results, args.training_bundle)
        pd.DataFrame(metrics).to_csv(args.out / "B1_learned_baselines.csv", index=False)
        pd.DataFrame(contrasts).to_csv(args.out / "B1_paired_contrasts.csv", index=False)
    print(f"Verified results saved in {args.out}")


if __name__ == "__main__":
    main()
