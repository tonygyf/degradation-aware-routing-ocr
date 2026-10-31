"""Combine the paper's five CC-OCR systems using verified paired sample identities."""

from collections import Counter
from statistics import mean

from src.evaluation.ccocr import evaluate as evaluate_routing
from src.evaluation.ccocr_diagnostics import repetition_flag
from src.evaluation.external import evaluate as evaluate_external

IDENTITY = ("group_id", "image_sha256", "reference", "split", "synthetic", "subset")


def compare(qwen_rows: list[dict], others: dict[str, list[dict]]) -> dict:
    if not qwen_rows or set(others) != {"internvl", "deepseek"}:
        raise ValueError("Require Qwen, InternVL and DeepSeek observations")
    base = {row["id"]: row for row in qwen_rows}
    if len(base) != len(qwen_rows):
        raise ValueError("Duplicate Qwen sample IDs")
    for model, rows in others.items():
        indexed = {row["id"]: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(base):
            raise ValueError(f"{model} sample IDs do not match Qwen")
        if any(
            row.get("system") != model or any(row[key] != base[key_id][key] for key in IDENTITY)
            for key_id, row in indexed.items()
        ):
            raise ValueError(f"{model} sample identity or model differs")
    systems, predictions = evaluate_routing(qwen_rows)
    if set(systems) != {"paddleocr", "qwen", "confroute"}:
        raise ValueError("Missing Qwen observations")
    repetition = {}
    for model in ("paddleocr", "qwen", "confroute"):
        records = [row for row in predictions if row["system"] == model]
        repetition[model] = _repetition([(row["subset"], row["prediction"]) for row in records])
    for model, rows in others.items():
        systems[model] = evaluate_external(rows, model)
        repetition[model] = _repetition([(row["subset"], row["vlm"]["text"]) for row in rows])
    for metrics in systems.values():
        metrics["source_macro"]["latency_ms_mean"] = mean(
            subset["latency_ms_mean"] for subset in metrics["subsets"].values()
        )
    counts = dict(Counter(row["subset"] for row in qwen_rows))
    return {
        "subset_counts": counts,
        "complete_external_set": counts == {"TotalText": 300, "InverseText": 500, "Hieragent": 100},
        "systems": systems,
        "repetition": repetition,
        "repetition_tokenization": "whitespace-normalized split; case/punctuation preserved",
    }


def _repetition(rows):
    flags = [(subset, repetition_flag(text)) for subset, text in rows]
    subsets = {
        name: {
            "samples": sum(subset == name for subset, _ in flags),
            "count": sum(flag for subset, flag in flags if subset == name),
        }
        for name in sorted({subset for subset, _ in flags})
    }
    for result in subsets.values():
        result["rate"] = result["count"] / result["samples"]
    return {
        "count": sum(flag for _, flag in flags),
        "pooled_rate": mean(flag for _, flag in flags),
        "subsets": subsets,
        "source_macro_rate": mean(item["rate"] for item in subsets.values()),
    }
