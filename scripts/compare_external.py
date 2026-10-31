"""Compare all five external systems after full paired observations exist."""

import argparse
import json
from pathlib import Path

from scripts.collect_ccocr import cached_ocr
from scripts.preflight_ccocr import MANIFEST_SHA256
from src.datasets.manifest import load_manifest, sha256
from src.evaluation.external_comparison import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/ccocr/manifest.jsonl"))
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--internvl", type=Path, required=True)
    parser.add_argument("--deepseek", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in ("qwen", "internvl", "deepseek")}
    if sha256(args.manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen external manifest changed")
    source = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    paired = cached_ocr(args.qwen, source)
    if set(paired) != {row["id"] for row in source}:
        raise ValueError("Qwen samples do not exactly match the frozen manifest")
    rows = {name: load_manifest(path) for name, path in inputs.items()}
    if any(
        row["split"] != "test" or row.get("synthetic") is not False
        for observations in rows.values()
        for row in observations
    ):
        raise ValueError("Require real frozen external test observations")
    report = {
        "input_sha256": {
            **{name: sha256(path) for name, path in inputs.items()},
            "manifest": sha256(args.manifest),
        },
        "evaluation_source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve().relative_to(Path.cwd()),
                Path("src/evaluation/ccocr.py"),
                Path("src/evaluation/external.py"),
                Path("src/evaluation/external_comparison.py"),
                Path("src/evaluation/ccocr_diagnostics.py"),
            )
        },
        "author_dataset_identity_verified": False,
        "ocr_latency_note": "OCR and Qwen are measured separately; see input run metadata",
        **compare(rows["qwen"], {name: rows[name] for name in ("internvl", "deepseek")}),
    }
    if not report["complete_external_set"]:
        raise ValueError("Refuse a five-system scientific report for an incomplete external set")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
