from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n")


def edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return edit distance, insertion count and reference length; ties prefer diagonal."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    previous = [(j, j) for j in range(len(hyp) + 1)]
    for i, a in enumerate(ref, 1):
        current = [(i, 0)]
        for j, b in enumerate(hyp, 1):
            diagonal = (previous[j - 1][0] + (a != b), previous[j - 1][1])
            deletion = (previous[j][0] + 1, previous[j][1])
            insertion = (current[j - 1][0] + 1, current[j - 1][1] + 1)
            current.append(min((diagonal, deletion, insertion), key=lambda x: x[0]))
        previous = current
    return previous[-1][0], previous[-1][1], len(ref)


def cer(reference: str, hypothesis: str) -> float:
    edits, _, n = edit_counts(reference, hypothesis)
    return edits / max(n, 1)


def summarize(records: list[dict], cost_per_call: float) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty split")
    total_edits = total_chars = insertions = 0
    regressions = blank_nonempty = blanks = disagreements = calls = 0
    latency = []
    for r in records:
        edits, ins, n = edit_counts(r["reference"], r["prediction"])
        total_edits += edits
        total_chars += n
        insertions += ins
        blanks += not normalize(r["reference"])
        blank_nonempty += not normalize(r["reference"]) and bool(normalize(r["prediction"]))
        calls += r["vlm_called"]
        regressions += r["vlm_called"] and (
            cer(r["reference"], r["prediction"]) > cer(r["reference"], r["ocr_text"])
        )
        disagreements += r["vlm_called"] and r["prediction"] != r["ocr_text"]
        latency.append(r["latency_ms"])
    ordered = sorted(latency)
    import math

    return {
        "samples": len(records),
        "cer_micro": total_edits / max(total_chars, 1),
        "edit_count": total_edits,
        "reference_characters": total_chars,
        "vlm_calls": calls,
        "vlm_call_rate": calls / len(records),
        "insertion_proxy": insertions / max(total_chars, 1),
        "blank_nonempty_rate": blank_nonempty / max(blanks, 1),
        "blank_samples": blanks,
        "vlm_regression_rate": regressions / max(calls, 1),
        "ocr_vlm_disagreement_rate": disagreements / max(calls, 1),
        "latency_ms_mean": sum(latency) / len(latency),
        "latency_ms_p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "declared_cost_units": calls * cost_per_call,
    }
