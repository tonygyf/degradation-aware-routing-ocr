"""Replay the published SHC policies on paired, preannotated observations."""

import argparse
from pathlib import Path

from scripts._report_io import read_real_rows, write_report
from src.datasets.manifest import sha256
from src.evaluation.shc_routing import replay, sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--include-internvl", action="store_true")
    args = parser.parse_args()
    if args.include_internvl and not args.sensitivity:
        parser.error("--include-internvl requires --sensitivity")
    rows = read_real_rows(args.observations)
    if any(row["role"] != "Test" for row in rows):
        raise ValueError("Frozen final SHC replay requires Test observations")
    report = {
        "observations_sha256": sha256(args.observations),
        "author_dataset_identity_verified": False,
        "paper_test_count_matches": len(rows) == 3897,
        "evaluation_source_sha256": {
            str(p): sha256(p)
            for p in (
                Path("scripts/_report_io.py"),
                Path("src/evaluation/shc_routing.py"),
                Path("src/evaluation/shc.py"),
                Path("src/evaluation/ccocr.py"),
                Path("src/routing/confroute.py"),
                Path(__file__).resolve().relative_to(Path.cwd()),
            )
        },
        **(
            sensitivity(rows, include_internvl=args.include_internvl)
            if args.sensitivity
            else replay(rows)
        ),
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
