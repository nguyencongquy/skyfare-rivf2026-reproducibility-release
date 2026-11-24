"""Estimate route-cluster sensitivity for the original paired contrasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.evaluation.pooled.metrics import distribution_row_loss, ranking_query_metrics
from skyfare.evaluation.rivf_revision.analyse import classification_loss, load_paper_evidence
from skyfare.evaluation.rivf_revision.evidence import ROOT, read_json
from skyfare.evaluation.rivf_revision.learned_baselines import Q_COLUMNS

DRAWS = 10_000
SEED = 43006


def differences(evidence: dict, staged: dict, baselines: dict) -> dict[str, pd.DataFrame]:
    routes = pd.concat([frame[["row_key", "route"]] for frame in staged.values()], ignore_index=True)
    result = {}
    for task, data in evidence.items():
        original = data["POOLED"]
        if task == "CLASSIFICATION":
            frame = original
            target = frame["DROP_5PCT"].to_numpy()
            baseline = frame["block_id"].map(baselines["classification_constant_probability"]).to_numpy()
            delta = classification_loss(target, frame["probability"].to_numpy()) - classification_loss(target, baseline)
            date_column = "feature_time"
        else:
            frame = original.drop(columns=["route"], errors="ignore").merge(routes, on="row_key", validate="one_to_one")
            date_column = "label_time"
            if len(frame) != len(original):
                raise RuntimeError(f"{task}: missing directional routes")
            if task == "POINT":
                target = frame["target_session_price_vnd"].to_numpy()
                delta = (abs(frame["predicted_price_vnd"] - target) - abs(frame["prior_anchor_vnd"] - target)) / target
            elif task == "DISTRIBUTION":
                target = frame["target_anchor_relative_log"].to_numpy()
                baseline = np.asarray([baselines["distribution_empirical_quantiles"][block] for block in frame["block_id"]])
                delta = distribution_row_loss(target, frame[list(Q_COLUMNS)].to_numpy()) - distribution_row_loss(target, baseline)
            elif task == "RANKING":
                frame = frame.rename(columns={"target_session_price_vnd": "target_price_vnd"})
                queries = ranking_query_metrics(frame, frame["score"].to_numpy())
                baseline = ranking_query_metrics(frame, -frame["prior_anchor_vnd"].to_numpy())
                if not queries["query_id"].equals(baseline["query_id"]):
                    raise RuntimeError("Ranking query populations disagree")
                query_routes = frame[["query_id", "route"]].drop_duplicates()
                frame = queries.merge(query_routes, on="query_id", validate="one_to_one")
                # Positive values favour the ensemble on the NDCG scale.
                delta = queries["ndcg5"].to_numpy() - baseline["ndcg5"].to_numpy()
            else:
                raise KeyError(task)
        result[task] = pd.DataFrame({
            "block_id": frame["block_id"].to_numpy(),
            "date": frame[date_column].dt.normalize().to_numpy(),
            "route": frame["route"].to_numpy(),
            "delta": np.asarray(delta, dtype=float),
        })
    return result


def route_cluster_ci(frame: pd.DataFrame) -> dict:
    """Sample whole directional routes, preserving each route across dates and tests."""
    if frame[["block_id", "date", "route", "delta"]].isna().any().any():
        raise RuntimeError("Missing keys or losses in paired route sample")
    totals = frame.groupby(["block_id", "date", "route"], sort=True)["delta"].agg(["sum", "count"])
    sums = totals["sum"].unstack("route", fill_value=0)
    counts = totals["count"].unstack("route", fill_value=0)
    routes = len(sums.columns)
    weights = np.random.default_rng(SEED).multinomial(routes, np.full(routes, 1 / routes), size=DRAWS)
    draw_sums = weights @ sums.to_numpy().T
    draw_counts = weights @ counts.to_numpy().T
    if (draw_counts == 0).any():
        raise RuntimeError("A cluster draw removed all support for an evaluation day")
    estimates = (draw_sums / draw_counts).mean(axis=1)
    return {
        "mean_model_minus_baseline": float((sums.sum(axis=1) / counts.sum(axis=1)).mean()),
        "route_ci95_low": float(np.quantile(estimates, 0.025)),
        "route_ci95_high": float(np.quantile(estimates, 0.975)),
        "route_clusters": routes,
        "fixed_days": len(sums),
        "draws": DRAWS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence, staged = load_paper_evidence(args.archives)
    baselines = read_json(ROOT / "configs/baseline_metrics.json")
    frozen = read_json(ROOT / "artifacts/evidence/pooled_evaluation_report.json")["stratified_day_bootstrap"]
    metrics = {"CLASSIFICATION": "classification_proper_score", "POINT": "point_mape",
               "DISTRIBUTION": "distribution_wis", "RANKING": "ranking_ndcg5"}
    rows = []
    for task, frame in differences(evidence, staged, baselines).items():
        interval = route_cluster_ci(frame)
        day = frozen[metrics[task]]
        reverse = -1 if task == "RANKING" else 1
        if not np.isclose(interval["mean_model_minus_baseline"], reverse * day["mean_model_minus_baseline"], atol=1e-12, rtol=0):
            raise RuntimeError(f"{task}: route bootstrap changed the frozen point estimand")
        day_limits = sorted(reverse * day[key] for key in ("ci95_low", "ci95_high"))
        rows.append({
            "task": task, "metric": metrics[task], "day_ci95_low": day_limits[0], "day_ci95_high": day_limits[1], **interval,
        })
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "N3_route_cluster_sensitivity.csv", index=False)
    print(f"Verified route-cluster sensitivity saved in {args.out}")


if __name__ == "__main__":
    main()
