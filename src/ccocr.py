"""Evaluate frozen real CC-OCR observations without fitting or tuning on test."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.datasets.manifest import load_manifest, sha256
from src.evaluation.ccocr import evaluate


def run(observations: Path, output: Path) -> dict:
    rows = load_manifest(observations)
    if any(r["split"] != "test" or r.get("synthetic") is not False for r in rows):
        raise ValueError("CC-OCR reproduction expects real frozen external test rows")
    metrics, predictions = evaluate(rows)
    counts = dict(Counter(r["subset"] for r in rows))
    complete = counts == {"TotalText": 300, "InverseText": 500, "Hieragent": 100}
    report = {
        "protocol": "IJDAR 2026 public CC-OCR; author package unavailable",
        "complete_external_set": complete,
        "subset_counts": counts,
        "threshold": 0.90,
        "observations_sha256": sha256(observations),
        "latency_kind": "measured, model load excluded; ConfRoute counterfactual replay",
        "systems": metrics,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.observations, args.output), indent=2))


if __name__ == "__main__":
    main()
