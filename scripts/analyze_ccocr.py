"""Compute supplementary diagnostics from completed external observations."""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.datasets.manifest import load_manifest, sha256
from src.evaluation.ccocr import evaluate
from src.evaluation.ccocr_diagnostics import diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_manifest(args.observations)
    if any(row["split"] != "test" or row.get("synthetic") is not False for row in rows):
        raise ValueError("Require real external test observations")
    metrics, records = evaluate(rows)
    counts = dict(Counter(row["subset"] for row in rows))
    report = {
        "complete_external_set": counts == {"TotalText": 300, "InverseText": 500, "Hieragent": 100},
        "subset_counts": counts,
        "qwen_token_limit_hits": sum(row["vlm"]["hit_token_limit"] for row in rows),
        "observations_sha256": sha256(args.observations),
        "samples": len(rows),
        "latency_kinds": sorted({row["latency_kind"] for row in rows}),
        "latency_note": (
            "Separately measured OCR/Qwen; ConfRoute counterfactual replay. "
            "See observation latency_kind and run metadata for devices; not paper hardware."
        ),
        "systems": metrics,
        **diagnostics(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
