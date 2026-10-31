"""Published SHC ranking and risk-coverage analyses on frozen paired observations."""

import math
from statistics import mean

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.ccocr import clipped_cer
from src.evaluation.shc import critical_span_flag
from src.evaluation.shc_routing import validate_paired
from src.routing.confroute import ConfRouteRouter

COVERAGES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def routing_labels(rows, target="route_benefit"):
    if target == "route_benefit":
        return [
            int(
                clipped_cer(r["reference_full"], r["vlm"]["text"])
                < clipped_cer(r["reference_full"], r["ocr"]["text"])
            )
            for r in rows
        ]
    if target == "operational_flag":
        return [int(critical_span_flag({**r, "prediction": r["ocr"]["text"]})) for r in rows]
    raise ValueError("Choose route_benefit or operational_flag target")


def binary_ranking(labels, scores):
    if not labels or len(labels) != len(scores) or any(x not in (0, 1) for x in labels):
        raise ValueError("Require aligned binary labels and scores")
    if any(not math.isfinite(x) for x in scores):
        raise ValueError("Scores must be finite")
    both = len(set(labels)) == 2
    return {
        "samples": len(labels),
        "positive_samples": sum(labels),
        "positive_rate": mean(labels),
        "auroc": float(roc_auc_score(labels, scores)) if both else None,
        "auprc_average_precision": float(average_precision_score(labels, scores)) if both else None,
        "metric_note": (
            "Average precision, not trapezoidal PR area; undefined single-class rankings are null"
        ),
    }


def confidence_ranking(rows):
    validate_paired(rows)
    router = ConfRouteRouter("box_min")
    scores = [-router.confidence(row["ocr"]) for row in rows]
    return {
        "signal": "negative box-minimum recognition confidence",
        "targets": {
            target: binary_ranking(routing_labels(rows, target), scores)
            for target in ("route_benefit", "operational_flag")
        },
    }


def risk_coverage(rows, *, random_resamples=1000, seed=42):
    validate_paired(rows)
    if not isinstance(random_resamples, int) or random_resamples < 2:
        raise ValueError("Require at least two random resamples")
    n = len(rows)
    paddle = np.array([clipped_cer(r["reference_full"], r["ocr"]["text"]) for r in rows])
    qwen = np.array([clipped_cer(r["reference_full"], r["vlm"]["text"]) for r in rows])
    delta = paddle - qwen
    base = float(qwen.mean())
    router = ConfRouteRouter("box_min")
    order = sorted(range(n), key=lambda i: (-router.confidence(rows[i]["ocr"]), rows[i]["id"]))
    counts = np.array([math.floor(c * n) for c in COVERAGES])
    ranked = base + np.r_[0.0, np.cumsum(delta[order])][counts] / n
    rng = np.random.default_rng(seed)
    random_values = np.array(
        [
            base + np.r_[0.0, np.cumsum(delta[rng.permutation(n)])][counts] / n
            for _ in range(random_resamples)
        ]
    )
    points = []
    for index, (coverage, k) in enumerate(zip(COVERAGES, counts, strict=True)):
        confidence = float(ranked[index])
        random_mean = float(random_values[:, index].mean())
        random_std = float(random_values[:, index].std(ddof=0))
        if k in (0, n):
            confidence = random_mean = base if k == 0 else float(paddle.mean())
            random_std = 0.0
        points.append(
            {
                "requested_paddle_coverage": coverage,
                "paddle_samples": int(k),
                "actual_paddle_coverage": k / n,
                "vlm_call_rate": 1 - k / n,
                "cer_full_confidence": confidence,
                "cer_full_random_mean": random_mean,
                "cer_full_random_std": random_std,
                "cer_full_random_exact_expectation": float(base + k / n * delta.mean()),
            }
        )
    return {
        "points": points,
        "random_resamples": random_resamples,
        "seed": seed,
        "rng": "numpy.default_rng/PCG64",
        "implementation_note": (
            "Independent conventions: floor(coverage*N), confidence ties "
            "by sample ID, population std; author "
            "rounding/ties/random-trial count unavailable"
        ),
        "interpretation": (
            "Keep top-confidence Paddle fraction, use Qwen elsewhere; "
            "CER over the full set, not only retained samples"
        ),
    }
