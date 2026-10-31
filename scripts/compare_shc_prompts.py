"""Aggregate paired SHC prompt caches without selecting prompts on Test."""

import argparse
from pathlib import Path

from scripts._report_io import read_real_rows, write_report
from src.datasets.manifest import sha256
from src.evaluation.shc_prompts import compare_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = read_real_rows(args.observations)
    report = {
        "observations_sha256": sha256(args.observations),
        "author_dataset_identity_verified": False,
        "evaluation_source_sha256": {
            str(p): sha256(p)
            for p in (
                Path("scripts/_report_io.py"),
                Path("src/evaluation/shc_prompts.py"),
                Path("src/evaluation/shc.py"),
                Path("src/evaluation/ccocr.py"),
                Path(__file__).resolve().relative_to(Path.cwd()),
            )
        },
        **compare_prompts(rows),
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
