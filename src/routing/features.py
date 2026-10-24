from __future__ import annotations

FEATURE_NAMES = ("mean_confidence", "min_confidence", "std_confidence", "box_count", "text_length")


def features(ocr: dict) -> list[float]:
    scores = ocr["confidences"]
    mean = sum(scores) / len(scores) if scores else 0.0
    variance = sum((x - mean) ** 2 for x in scores) / len(scores) if scores else 0.0
    return [mean, min(scores, default=0.0), variance**0.5, len(scores), len(ocr["text"])]
