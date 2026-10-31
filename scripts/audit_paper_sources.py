"""Audit original Baseline/SHC source reuse without changing frozen roles."""

import argparse
from pathlib import Path

from scripts._report_io import read_real_rows, write_report
from src.datasets.manifest import sha256
from src.datasets.paper_sources import audit_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--shc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in ("baseline", "shc")}
    rows = {name: read_real_rows(path) for name, path in inputs.items()}
    report = {
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "author_dataset_identity_verified": False,
        "evaluation_source_sha256": {
            str(p): sha256(p)
            for p in (
                Path("scripts/_report_io.py"),
                Path("src/datasets/paper_sources.py"),
                Path(__file__).resolve().relative_to(Path.cwd()),
            )
        },
        **audit_sources(rows["baseline"], rows["shc"]),
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
