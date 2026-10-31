"""SHC box-minimum routing and the paper's aggressive empty-OCR pre-filter."""

import math
from statistics import mean

from src.evaluation.ccocr import normalize
from src.evaluation.shc import evaluate
from src.routing.confroute import ConfRouteRouter


def validate_paired(rows: list[dict], fallback_key="vlm") -> None:
    if not rows:
        raise ValueError("Cannot replay an empty SHC set")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("SHC sample IDs must be unique")
    if any(not isinstance(key, str) or not key for key in ids):
        raise ValueError("Require nonempty sample IDs")
    for row in rows:
        ConfRouteRouter("box_min").confidence(row["ocr"])
        for key in ("ocr", fallback_key):
            observation = row[key]
            if not isinstance(observation["text"], str):
                raise ValueError("Predictions must be text")
            if not math.isfinite(observation["latency_ms"]) or observation["latency_ms"] < 0:
                raise ValueError("Latencies must be finite and nonnegative")


def replay(rows: list[dict], *, threshold=0.90, fallback="qwen") -> dict:
    if fallback not in {"qwen", "internvl"}:
        raise ValueError("Choose qwen or internvl fallback")
    fallback_key = "vlm" if fallback == "qwen" else "internvl"
    validate_paired(rows, fallback_key)
    router = ConfRouteRouter("box_min", threshold)
    calls = [bool(router.score(row["ocr"])) for row in rows]
    ocr_mean = mean(row["ocr"]["latency_ms"] for row in rows)
    vlm_mean = mean(row[fallback_key]["latency_ms"] for row in rows)
    report = {}
    for system in ("paddleocr", fallback, "confroute", "no_text_confroute"):
        predictions, routed = [], []
        for row, call in zip(rows, calls, strict=True):
            prefiltered = system == "no_text_confroute" and not normalize(row["ocr"]["text"])
            use_vlm = system == fallback or (
                system in {"confroute", "no_text_confroute"} and call and not prefiltered
            )
            text = "" if prefiltered else row[fallback_key if use_vlm else "ocr"]["text"]
            predictions.append({**row, "prediction": text})
            routed.append(use_vlm)
        rate = mean(routed)
        report[system] = {
            **evaluate(predictions),
            "vlm_call_rate": rate,
            "estimated_latency_ms_mean": (0 if system == fallback else ocr_mean) + rate * vlm_mean,
        }
    return {
        "threshold": threshold,
        "fallback": fallback,
        "signal": "box_min",
        "latency_kind": "paper SHC estimate: mean OCR + call rate * mean VLM; not online timing",
        "systems": report,
    }


def sensitivity(rows: list[dict], *, include_internvl=False) -> dict:
    reports = {str(t): replay(rows, threshold=t) for t in (0.85, 0.90, 0.95)}
    baseline = reports["0.9"]["systems"]
    empty = [not normalize(row["ocr"]["text"]) for row in rows]
    s6 = [row["failure_type"] == "S6" for row in rows]
    true_positive = sum(a and b for a, b in zip(empty, s6, strict=True))
    result = {
        "historical_threshold": 0.90,
        "selected_threshold": None,
        "thresholds": reports,
        "endpoints": {"always_paddleocr": baseline["paddleocr"], "always_qwen": baseline["qwen"]},
        "no_text_audit": {
            "predicted_no_text": sum(empty),
            "s6_samples": sum(s6),
            "true_positive": true_positive,
            "non_s6_filtered": sum(empty) - true_positive,
            "precision": true_positive / sum(empty) if any(empty) else None,
            "recall": true_positive / sum(s6) if any(s6) else None,
        },
    }
    if include_internvl:
        result["internvl_fallback"] = replay(rows, fallback="internvl")
    return result
