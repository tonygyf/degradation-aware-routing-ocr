"""Published historical ConfRoute; SHC box caches require an explicit signal."""

import math

from src.evaluation.ccocr import image_confidence


class ConfRouteRouter:
    def __init__(self, signal: str, threshold: float = 0.90):
        if signal not in {"image_mean", "box_min"}:
            raise ValueError("Choose image_mean (Baseline/CC-OCR) or box_min (SHC)")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be finite and in [0, 1]")
        self.signal = signal
        self.threshold = threshold

    def confidence(self, ocr: dict) -> float:
        if self.signal == "image_mean":
            value = image_confidence(ocr)
            return 0.0 if value is None else value
        # The SHC cache is separate: never fall back to the full-page scalar.
        boxes = ocr["confidence_boxes"]
        if not isinstance(boxes, list):
            raise ValueError("confidence_boxes must be a list")
        scores = [box["confidence"] for box in boxes]
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
            raise ValueError("Invalid box confidence")
        return min(scores, default=0.0)

    def score(self, ocr: dict) -> float:
        return float(self.confidence(ocr) < self.threshold)
