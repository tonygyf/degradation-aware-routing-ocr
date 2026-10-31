"""Supplementary CC-OCR repetition and paired stratified comparisons."""

from statistics import mean

from src.evaluation.bootstrap import paired_stratified
from src.evaluation.ccocr import clipped_cer, eval_trans, normalize

METRICS = ("eval_trans", "cer_macro_clipped", "latency_ms_mean", "vlm_call_rate")


def repetition_flag(text: str) -> bool:
    tokens = normalize(text).split()
    return any(
        tokens[start : start + width]
        == tokens[start + width : start + 2 * width]
        == tokens[start + 2 * width : start + 3 * width]
        for width in range(2, min(20, len(tokens) // 3) + 1)
        for start in range(len(tokens) - 3 * width + 1)
    )


def diagnostics(records: list[dict], *, resamples=10_000, seed=42) -> dict:
    systems = {}
    for row in records:
        entries = systems.setdefault(row["system"], {})
        if row["id"] in entries:
            raise ValueError("Duplicate sample within system")
        entries[row["id"]] = row
    if not {"confroute", "qwen", "paddleocr"} <= systems.keys():
        raise ValueError("Require paired ConfRoute, Qwen and PaddleOCR records")
    ids = sorted(systems["confroute"])
    for entries in systems.values():
        if set(entries) != set(ids):
            raise ValueError("Systems must have identical sample IDs")
        for key in ids:
            if any(
                entries[key][field] != systems["confroute"][key][field]
                for field in ("reference", "subset")
            ):
                raise ValueError("Paired sample references and subsets must match")

    def scores(row):
        return (
            eval_trans(row["reference"], row["prediction"]),
            clipped_cer(row["reference"], row["prediction"]),
            row["latency_ms"],
            float(row["vlm_called"]),
        )

    comparisons = {}
    for other in ("qwen", "paddleocr"):
        differences = [
            [
                a - b
                for a, b in zip(
                    scores(systems["confroute"][key]), scores(systems[other][key]), strict=True
                )
            ]
            for key in ids
        ]
        interval = paired_stratified(
            differences,
            [systems["confroute"][key]["subset"] for key in ids],
            resamples=resamples,
            seed=seed,
        )
        comparisons[f"confroute_minus_{other}"] = {
            metric: {name: values[index] for name, values in interval.items()}
            for index, metric in enumerate(METRICS)
        }
    repetition = {}
    for system, entries in systems.items():
        flags = {key: repetition_flag(row["prediction"]) for key, row in entries.items()}
        subsets = {
            subset: mean(flags[key] for key in ids if entries[key]["subset"] == subset)
            for subset in sorted({row["subset"] for row in entries.values()})
        }
        repetition[system] = {
            "count": sum(flags.values()),
            "pooled_rate": mean(flags.values()),
            "subset_rates": subsets,
            "source_macro_rate": mean(subsets.values()),
        }
    return {
        "resamples": resamples,
        "seed": seed,
        "rng": "numpy.default_rng/PCG64",
        "interval": "percentile 2.5/97.5; paired within subset; A minus B",
        "repetition_tokenization": "whitespace-normalized split; case/punctuation preserved",
        "comparisons": comparisons,
        "repetition": repetition,
    }
