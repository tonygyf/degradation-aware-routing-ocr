"""Published internal document-level metrics on paired, independently sourced observations."""

from statistics import mean

from src.evaluation.bootstrap import paired_sign_flip, paired_stratified
from src.evaluation.ccocr import clipped_cer, image_confidence, normalize
from src.routing.confroute import ConfRouteRouter


def evaluate(rows):
    if not rows:
        raise ValueError("Require paired Baseline observations")
    names = ("paddleocr", "qwen", "confroute")
    predictions = {name: [] for name in names}
    for row in rows:
        confidence = image_confidence(row["ocr"])
        route = (0 if confidence is None else confidence) < 0.90
        for name in names:
            called = name == "qwen" or (name == "confroute" and route)
            output = row["vlm"] if called else row["ocr"]
            latency = output["latency_ms"]
            if name == "confroute" and called:
                latency += row["ocr"]["latency_ms"]
            predictions[name].append(
                {
                    "reference": row["reference"],
                    "prediction": output["text"],
                    "called": called,
                    "latency_ms": latency,
                }
            )
    scores = {
        name: [clipped_cer(x["reference"], x["prediction"]) for x in items]
        for name, items in predictions.items()
    }
    systems = {
        name: {
            "samples": len(items),
            "cer_macro_clipped": mean(scores[name]),
            "empty_rate": mean(not normalize(x["prediction"]) for x in items),
            "vlm_call_rate": mean(x["called"] for x in items),
            "latency_ms_mean": mean(x["latency_ms"] for x in items),
        }
        for name, items in predictions.items()
    }
    differences = [q - p for q, p in zip(scores["qwen"], scores["paddleocr"], strict=True)]
    interval = paired_stratified([[difference] for difference in differences], ["all"] * len(rows))
    return {
        "threshold": 0.90,
        "signal": "image_mean",
        "systems": systems,
        "qwen_minus_paddle_cer": {
            "paired_bootstrap_10000_seed42": interval,
            "sign_flip": paired_sign_flip(differences),
        },
    }


def _require_role(rows, role):
    if not rows or any(
        r.get("role") != role or r.get("split") != "val" or "vlm" not in r for r in rows
    ):
        raise ValueError(f"Require paired {role} observations represented as split=val")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{role} sample IDs must be unique")


def _threshold_candidates(rows, thresholds, signal):
    always_qwen_cer = mean(clipped_cer(row["reference"], row["vlm"]["text"]) for row in rows)
    candidates = []
    for threshold in thresholds:
        router = ConfRouteRouter(signal, threshold)
        calls = [threshold == 1.0 or bool(router.score(row["ocr"])) for row in rows]
        cer = mean(
            clipped_cer(row["reference"], row["vlm" if call else "ocr"]["text"])
            for row, call in zip(rows, calls, strict=True)
        )
        candidates.append(
            {
                "threshold": threshold,
                "samples": len(rows),
                "vlm_calls": sum(calls),
                "vlm_call_rate": mean(calls),
                "cer_macro_clipped": cer,
                "delta_cer_vs_qwen_percentage_points": 100 * (cer - always_qwen_cer),
                "latency_ms_mean": mean(
                    row["ocr"]["latency_ms"] + call * row["vlm"]["latency_ms"]
                    for row, call in zip(rows, calls, strict=True)
                ),
            }
        )
    return candidates, always_qwen_cer


def replay_calibration(rows):
    """Replay published Cal candidates; this does not select a threshold."""
    _require_role(rows, "Cal")
    candidates, always_qwen_cer = _threshold_candidates(
        rows, (0.80, 0.85, 0.90, 0.95), "image_mean"
    )
    return {
        "role": "Cal",
        "signal": "image_mean",
        "historical_threshold": 0.90,
        "selected_threshold": None,
        "selection_note": "No recovered deterministic author Cal-only selection rule; replay only",
        "always_qwen_cer_macro_clipped": always_qwen_cer,
        "candidates": candidates,
    }


def replay_development(rows):
    _require_role(rows, "Dev")
    thresholds = tuple(i / 20 for i in range(21))
    return {
        "role": "Dev",
        "historical_threshold": 0.90,
        "selected_threshold": None,
        "signals": {
            signal: _threshold_candidates(rows, thresholds, signal)[0]
            for signal in ("image_mean", "box_min")
        },
        "always_qwen_vlm_only_latency_ms_mean": mean(r["vlm"]["latency_ms"] for r in rows),
        "endpoint_note": (
            "0/1 are forced always-OCR/always-Qwen endpoints. Grid timings include OCR "
            "as in the serial replay equation; VLM-only endpoint timing is reported separately."
        ),
        "selection_note": "Dev frontier is descriptive, never a threshold-fitting result",
    }


def lambda_sensitivity(rows):
    _require_role(rows, "Cal")
    grid, _ = _threshold_candidates(rows, tuple(i / 20 for i in range(21)), "image_mean")
    choices = []
    for weight in (0.0, 0.002, 0.005, 0.010, 0.020):
        winner = min(
            grid,
            key=lambda r: (
                r["cer_macro_clipped"] + weight * r["latency_ms_mean"] / 1000,
                r["vlm_calls"],
                r["threshold"],
            ),
        )
        choices.append(
            {
                "lambda": weight,
                "lambda_optimal_threshold": winner["threshold"],
                "objective": winner["cer_macro_clipped"]
                + weight * winner["latency_ms_mean"] / 1000,
            }
        )
    return {
        "role": "Cal",
        "historical_threshold": 0.90,
        "historical_policy_modified": False,
        "grid": grid,
        "lambda_choices": choices,
        "objective": "CER fraction + lambda * serial latency in seconds",
        "convention_note": (
            "Independent explicit cost/tie convention: objective, fewer calls, lower threshold. "
            "Author exact lambda code unavailable; this does not recover the historical rule."
        ),
    }
