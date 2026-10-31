"""Single-model CC-OCR scores without relabeling other VLM outputs as Qwen."""

from collections import Counter
from statistics import mean

from src.evaluation.ccocr import summarize


def evaluate(rows, system):
    if not rows or system not in {"internvl", "deepseek"}:
        raise ValueError("Require external InternVL or DeepSeek observations")
    if any(row.get("system") != system for row in rows):
        raise ValueError("Model identity mismatch")
    records = [
        {
            "reference": row["reference"],
            "prediction": row["vlm"]["text"],
            "latency_ms": row["vlm"]["latency_ms"],
            "vlm_called": True,
            "subset": row["subset"],
        }
        for row in rows
    ]
    subsets = {
        name: summarize([row for row in records if row["subset"] == name])
        for name in sorted({row["subset"] for row in records})
    }
    counts = dict(Counter(row["subset"] for row in rows))
    return {
        "system": system,
        "subset_counts": counts,
        "complete_external_set": counts == {"TotalText": 300, "InverseText": 500, "Hieragent": 100},
        "pooled": summarize(records),
        "subsets": subsets,
        "source_macro": {
            key: mean(part[key] for part in subsets.values())
            for key in ("eval_trans", "cer_macro_clipped", "empty_rate", "vlm_call_rate")
        },
    }
