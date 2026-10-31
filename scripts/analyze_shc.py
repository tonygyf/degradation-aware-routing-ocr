"""Analyze frozen SHC caches; optional paper L2 training uses Tau_dev only."""

import argparse
import importlib.metadata
import json
from pathlib import Path
from statistics import mean

from scripts._report_io import read_real_rows, write_report
from src.datasets.manifest import sha256
from src.datasets.paper_sources import canonical_source
from src.evaluation.shc import evaluate
from src.evaluation.shc_analysis import (
    binary_ranking,
    confidence_ranking,
    risk_coverage,
    routing_labels,
)
from src.evaluation.shc_routing import validate_paired
from src.routing.shc_learned import PaperSHCRouter


def read_rows(path):
    rows = read_real_rows(path)
    validate_paired(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--router-config", type=Path)
    parser.add_argument("--random-resamples", type=int, default=1000)
    args = parser.parse_args()
    if bool(args.train) != bool(args.router_config):
        parser.error("Provide --train and --router-config together")
    rows = read_rows(args.observations)
    if any(r.get("role") != "Test" or r.get("split", "test") != "test" for r in rows):
        raise ValueError("Final SHC analysis requires frozen Test observations")
    report = {
        "observations_sha256": sha256(args.observations),
        "author_dataset_identity_verified": False,
        "runtime_packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy")
        },
        "confidence_ranking": confidence_ranking(rows),
        "risk_coverage": risk_coverage(rows, random_resamples=args.random_resamples),
    }
    if args.train:
        train = read_rows(args.train)
        if {r["id"] for r in train} & {r["id"] for r in rows}:
            raise ValueError("Training and Test sample IDs must not overlap")
        config = json.loads(args.router_config.read_text())
        router = PaperSHCRouter(config).fit(train)
        probabilities = [router.score(r["ocr"]) for r in rows]
        calls = [p >= config["threshold"] for p in probabilities]
        predictions = [
            {**r, "prediction": r["vlm" if call else "ocr"]["text"]}
            for r, call in zip(rows, calls, strict=True)
        ]
        report["learned_router"] = {
            "train_sha256": sha256(args.train),
            "config_sha256": sha256(args.router_config),
            "model": router.metadata(),
            "ranking": binary_ranking(routing_labels(rows, config["target"]), probabilities),
            "test_metrics": evaluate(predictions),
            "vlm_call_rate": mean(calls),
            "estimated_latency_ms_mean": mean(r["ocr"]["latency_ms"] for r in rows)
            + mean(calls) * mean(r["vlm"]["latency_ms"] for r in rows),
            "train_test_shared_parent_sources": len(
                {canonical_source(r) for r in train} & {canonical_source(r) for r in rows}
            ),
            "source_independence_claimed": False,
        }
    paths = [
        "scripts/analyze_shc.py",
        "scripts/_report_io.py",
        "src/evaluation/shc.py",
        "src/evaluation/ccocr.py",
        "src/evaluation/shc_analysis.py",
        "src/evaluation/shc_routing.py",
        "src/routing/confroute.py",
        "src/routing/shc_learned.py",
        "src/datasets/paper_sources.py",
    ]
    report["evaluation_source_sha256"] = {p: sha256(Path(p)) for p in paths}
    report["latency_note"] = "SHC mean OCR + routed fraction * mean VLM; not online timing"
    write_report(args.output, report)


if __name__ == "__main__":
    main()
