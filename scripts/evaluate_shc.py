"""Evaluate one system's independently prepared SHC-format JSONL observations."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts._report_io import write_report
from src.evaluation.bootstrap import cluster_flag_interval
from src.evaluation.shc import critical_span_flag, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cluster-bootstrap", action="store_true")
    args = parser.parse_args()
    content = args.observations.read_bytes()
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("Require unique, nonempty sample IDs for one system")
    report = {
        "implementation": "independent-published-equations",
        "author_dataset_identity_verified": False,
        "observations_sha256": hashlib.sha256(content).hexdigest(),
        **evaluate(rows),
    }
    if args.cluster_bootstrap:
        report["cluster_bootstrap"] = {
            "resamples": 10_000,
            "seed": 42,
            "rng": "numpy.default_rng/PCG64",
        }
        for field in ("family_id", "source_image"):
            clusters = [row[field] for row in rows]
            if any(not isinstance(key, str) or not key for key in clusters):
                raise ValueError(f"Require audited nonempty {field} identifiers")
            report["cluster_bootstrap"][field] = cluster_flag_interval(
                [critical_span_flag(row) for row in rows], clusters
            )
    write_report(args.output, report)


if __name__ == "__main__":
    main()
