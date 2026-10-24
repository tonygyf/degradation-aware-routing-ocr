"""IJDAR clipped macro-CER and the English CC-OCR Eval-Trans convention.

Eval-Trans follows AlibabaResearch/AdvancedLiterateMachinery at
fb6187ebe5374d5682ac0bc05f31e3639ab6fb15, ocr_evaluator.py (MIT).
See docs/licenses/CC-OCR-MIT.txt for the upstream license.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean, median


def normalize(text: str) -> str:
    return " ".join(text.split())


def clipped_cer(reference: str, prediction: str) -> float:
    reference, prediction = normalize(reference), normalize(prediction)
    previous = list(range(len(prediction) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return min(1.0, previous[-1] / max(1, len(reference)))


def eval_trans(reference: str, prediction: str) -> float:
    def tokens(text):
        text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
        words = re.sub(r"\s+", " ", text.strip()).split(" ")
        return [word for item in words if (word := re.sub("[^A-Za-z0-9]+", "", item.lower()))]

    gt, pred = tokens(reference), tokens(prediction)
    correct = sum((Counter(gt) & Counter(pred)).values())
    recall, precision = correct / (len(gt) + 1e-9), correct / (len(pred) + 1e-9)
    return 2 * recall * precision / (recall + precision + 1e-9)


def image_confidence(ocr: dict) -> float | None:
    texts, scores = ocr["rec_texts"], ocr["confidences"]
    if len(texts) != len(scores):
        raise ValueError("Recognition texts and confidences must have matching lengths")
    if any(not isinstance(t, str) for t in texts) or any(
        not math.isfinite(s) or not 0 <= s <= 1 for s in scores
    ):
        raise ValueError("Invalid recognition text or confidence")
    selected = [s for t, s in zip(texts, scores, strict=True) if t]
    return round(mean(selected), 4) if selected else None


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot evaluate an empty set")
    latency = sorted(r["latency_ms"] for r in records)
    return {
        "samples": len(records),
        "eval_trans": sum(eval_trans(r["reference"], r["prediction"]) for r in records)
        / (len(records) + 1e-9),
        "cer_macro_clipped": mean(clipped_cer(r["reference"], r["prediction"]) for r in records),
        "empty_rate": mean(not normalize(r["prediction"]) for r in records),
        "vlm_call_rate": mean(r["vlm_called"] for r in records),
        "latency_ms_mean": mean(latency),
        "latency_ms_median": median(latency),
        "latency_ms_p95": latency[math.ceil(0.95 * len(latency)) - 1],
    }


def evaluate(rows: list[dict], threshold: float = 0.90) -> tuple[dict, list[dict]]:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    systems = (
        ("paddleocr", "qwen", "confroute") if all("vlm" in r for r in rows) else ("paddleocr",)
    )
    report, predictions = {}, []
    for system in systems:
        records = []
        for row in rows:
            confidence = image_confidence(row["ocr"])
            called = system == "qwen" or (
                system == "confroute" and (0 if confidence is None else confidence) < threshold
            )
            observation = row["vlm"] if called else row["ocr"]
            latency = observation["latency_ms"]
            if system == "confroute" and called:
                latency += row["ocr"]["latency_ms"]
            records.append(
                {
                    "id": row["id"],
                    "subset": row["subset"],
                    "system": system,
                    "reference": row["reference"],
                    "prediction": observation["text"],
                    "image_level_confidence": confidence,
                    "vlm_called": called,
                    "latency_ms": latency,
                }
            )
        by_subset = {
            name: summarize([r for r in records if r["subset"] == name])
            for name in sorted({r["subset"] for r in records})
        }
        report[system] = {"pooled": summarize(records), "subsets": by_subset}
        report[system]["source_macro"] = {
            key: mean(part[key] for part in by_subset.values())
            for key in ("eval_trans", "cer_macro_clipped", "empty_rate", "vlm_call_rate")
        }
        predictions.extend(records)
    return report, predictions
