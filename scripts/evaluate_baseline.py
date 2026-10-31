"""Evaluate real paired internal test observations, without asserting author identity."""

import argparse
import json
from pathlib import Path

from src.datasets.manifest import load_manifest, sha256
from src.evaluation.baseline import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_manifest(args.observations)
    if any(
        row["split"] != "test" or row.get("synthetic") is not False or "vlm" not in row
        for row in rows
    ):
        raise ValueError("Require real paired test observations")
    report = {
        "observations_sha256": sha256(args.observations),
        "author_dataset_identity_verified": False,
        "paper_test_count_matches": len(rows) == 116,
        "statistics_note": (
            "Independent implementation; author RNG and sign-flip convention not verified"
        ),
        **evaluate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
