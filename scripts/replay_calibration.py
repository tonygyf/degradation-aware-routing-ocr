"""Replay the published internal Cal candidates without selecting a new policy."""

import argparse
import json
from pathlib import Path

from src.datasets.manifest import load_manifest, sha256
from src.evaluation.baseline import lambda_sensitivity, replay_calibration, replay_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("Cal", "Dev"), default="Cal")
    parser.add_argument("--lambda-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.lambda_sensitivity and args.role != "Cal":
        parser.error("Lambda sensitivity is Cal-only")
    rows = load_manifest(args.observations)
    if any(row.get("synthetic") is not False for row in rows):
        raise ValueError("Require real Cal observations; synthetic fixtures are software checks")
    report = {
        "observations_sha256": sha256(args.observations),
        "author_dataset_identity_verified": False,
        "paper_role_count_matches": len(rows) == (87 if args.role == "Cal" else 347),
        "evaluation_source_sha256": {
            str(path): sha256(path)
            for path in (
                Path("src/evaluation/baseline.py"),
                Path("src/evaluation/ccocr.py"),
                Path("src/routing/confroute.py"),
                Path(__file__).resolve().relative_to(Path.cwd()),
            )
        },
        "latency_note": "Stored per-image OCR + routed Qwen timings; counterfactual replay",
        **(replay_calibration(rows) if args.role == "Cal" else replay_development(rows)),
    }
    if args.role == "Cal":
        report["paper_cal_count_matches"] = len(rows) == 87
    if args.lambda_sensitivity:
        report["lambda_sensitivity"] = lambda_sensitivity(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
