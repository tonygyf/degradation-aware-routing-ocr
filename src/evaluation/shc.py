"""Independent implementation of published SHC equations; not an author-data adapter."""

from statistics import mean

from src.evaluation.ccocr import clipped_cer, normalize

SEMANTIC_TYPES = frozenset(f"S{i}" for i in range(1, 7))
FAILURE_TYPES = SEMANTIC_TYPES | {"L1", "L2", "R1", "R2"}


def critical_span_flag(row: dict) -> bool:
    """Use preassigned annotations only; never infer spans from predictions."""
    if row["failure_type"] not in FAILURE_TYPES:
        raise ValueError("Unknown SHC failure type")
    for key in ("reference_visible", "reference_full", "prediction"):
        if not isinstance(row[key], str):
            raise ValueError(f"{key} must be text")
    visible = normalize(row["reference_visible"])
    prediction = normalize(row["prediction"])
    if row["failure_type"] == "S6":
        if visible or normalize(row["reference_full"]):
            raise ValueError("S6 requires both references to be empty")
        return bool(prediction)
    span = row["critical_span"]
    if span is None:
        return False
    if not isinstance(span, str) or not normalize(span):
        raise ValueError("Use null for an unassigned critical span")
    span = normalize(span)
    if not any(character.isalnum() for character in span):
        raise ValueError("Critical spans must be content-bearing, not punctuation-only")
    return span in prediction and span not in visible


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty set")
    flags = [critical_span_flag(row) for row in rows]
    flagged = sum(flags)
    semantic = sum(
        flag and row["failure_type"] in SEMANTIC_TYPES
        for row, flag in zip(rows, flags, strict=True)
    )
    return {
        "samples": len(rows),
        "cer_visible_macro_clipped": mean(
            clipped_cer(row["reference_visible"], row["prediction"]) for row in rows
        ),
        "cer_full_macro_clipped": mean(
            clipped_cer(row["reference_full"], row["prediction"]) for row in rows
        ),
        "empty_rate": mean(not normalize(row["prediction"]) for row in rows),
        "operational_flag_count": flagged,
        "operational_flag_rate": flagged / len(rows),
        "semantic_flag_count": semantic,
        "semantic_share": semantic / max(1, flagged),
    }


def evaluate(rows: list[dict]) -> dict:
    pooled = summarize(rows)
    groups = {"identical": [], "different": []}
    for row in rows:
        key = (
            "identical"
            if normalize(row["reference_visible"]) == normalize(row["reference_full"])
            else "different"
        )
        groups[key].append(row)
    return {
        "pooled": pooled,
        "reference_overlap": {
            key: summarize(group) if group else None for key, group in groups.items()
        },
        "failure_types": {
            kind: summarize([row for row in rows if row["failure_type"] == kind])
            for kind in sorted({row["failure_type"] for row in rows})
        },
    }
