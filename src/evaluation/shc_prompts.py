"""Paired model/prompt cache comparison; Test remains exploratory only."""

import math
from statistics import mean

from src.evaluation.shc import evaluate

PROMPTS = ("P0", "P2", "P3", "P2+P3", "P0+P3")
IDENTITY = (
    "role",
    "failure_type",
    "reference_visible",
    "reference_full",
    "critical_span",
    "source_corpus",
    "source_image",
    "family_id",
)


def compare_prompts(rows):
    if not rows or len({r["role"] for r in rows}) != 1:
        raise ValueError("Require one nonempty role per comparison")
    role = rows[0]["role"]
    if role not in {"Prompt_dev", "Test"}:
        raise ValueError("Prompt comparisons use Prompt_dev or exploratory Test")
    grouped, gold = {}, {}
    for row in rows:
        if any(not isinstance(row[k], str) or not row[k] for k in ("id", "model")):
            raise ValueError("Require nonempty sample and model IDs")
        if row["prompt"] not in PROMPTS:
            raise ValueError("Prompt family is not part of the published fixed sweep")
        latency = row["latency_ms"]
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Latency must be finite and nonnegative")
        identity = tuple(row.get(k) for k in IDENTITY)
        if row["id"] in gold and gold[row["id"]] != identity:
            raise ValueError("Paired annotations/source identities differ")
        gold[row["id"]] = identity
        entries = grouped.setdefault((row["model"], row["prompt"]), {})
        if row["id"] in entries:
            raise ValueError("Duplicate sample within model/prompt")
        entries[row["id"]] = row
    if any(set(entries) != set(gold) for entries in grouped.values()):
        raise ValueError("All model/prompt groups must contain identical sample IDs")
    models = {}
    for (model, prompt), entries in grouped.items():
        items = [entries[key] for key in sorted(entries)]
        models.setdefault(model, {})[prompt] = {
            **evaluate(items),
            "latency_ms_mean": mean(r["latency_ms"] for r in items),
        }
    if any("P0" not in prompts for prompts in models.values()):
        raise ValueError("Require the P0 baseline for each model")
    selected = None
    if role == "Prompt_dev":
        selected = {
            model: min(
                prompts,
                key=lambda p: (prompts[p]["pooled"]["cer_full_macro_clipped"], PROMPTS.index(p)),
            )
            for model, prompts in models.items()
        }
    return {
        "role": role,
        "samples_per_group": len(gold),
        "models": models,
        "selected_dev_prompts": selected,
        "main_comparison_prompt": "P0",
        "exploratory_nonconfirmatory": role == "Test",
        "selection_note": (
            "Prompt_dev CER minimum only; independent ties by fixed "
            "prompt order. No Test selection; does not rewrite the P0 "
            "main comparison."
        ),
    }
