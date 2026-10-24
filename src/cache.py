"""Explicit, preflight-gated real observation collection, separate from offline routing."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.manifest import load_manifest, sha256
from src.recognition.backends import PaddleOCRAdapter, QwenVLHTTPAdapter


def preflight(registry: dict) -> None:
    if registry.get("reviewed") is not True or registry.get("data_authorized") is not True:
        raise ValueError("Review model provenance/resources and data authorization first")
    required = (
        "dataset_license",
        "dataset_source",
        "split_protocol",
        "ocr_model_revision",
        "ocr_model_license",
        "qwen_model_id",
        "qwen_revision",
        "qwen_model_license",
        "serving_version",
        "resources",
    )
    for key in required:
        value = registry.get(key)
        if not isinstance(value, str) or not value.strip() or "REQUIRED" in value:
            raise ValueError(f"Fill reviewed registry field: {key}")
    if registry.get("device") != "cpu":
        raise ValueError("Stage 1 observation collection is CPU OCR only")


def collect(manifest: Path, registry_path: Path, output: Path, allow_real: bool) -> None:
    if not allow_real:
        raise ValueError("Explicit --allow-real is required; this command calls both models")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    preflight(registry)
    for package in ("paddleocr", "paddlepaddle"):
        if importlib.metadata.version(package) != registry[f"{package}_package"]:
            raise ValueError(f"Installed version mismatch: {package}")
    rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x]
    # Validate all identity/split/license fields before any network or inference calls.
    for row in rows:
        image = (manifest.parent / row["image"]).resolve()
        if row["image_sha256"] != sha256(image):
            raise ValueError(f"Image hash mismatch: {row['id']}")
        if row.get("synthetic"):
            raise ValueError("Real collection must not consume mocked fixtures")
    output.mkdir(parents=True, exist_ok=False)
    validation = output / "input-validation.jsonl"
    validation.write_text(
        "".join(
            json.dumps({**r, "ocr": {"text": "", "confidences": [], "latency_ms": 0}}) + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    load_manifest(validation)
    (output / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": sha256(manifest),
                "registry_sha256": sha256(registry_path),
                "latency_kind": "measured_serial_inference_ms_excludes_model_load",
                "counterfactual_collection": True,
                "note": "Calls every image for train labels and offline policy replay; "
                "not selective serving",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ocr = PaddleOCRAdapter(
        Path(registry["paddle_detection_dir"]), Path(registry["paddle_recognition_dir"])
    )
    vlm = QwenVLHTTPAdapter(registry["qwen_endpoint"], registry["qwen_model_id"])
    partial = output / "partial.jsonl"
    with partial.open("x", encoding="utf-8") as handle:
        for row in rows:
            image = (manifest.parent / row["image"]).resolve()
            result = {
                **row,
                "image": str(image),
                "ocr": ocr.infer(image),
                "vlm": vlm.infer(image),
                "registry_sha256": sha256(registry_path),
                "latency_kind": "measured_serial_inference_ms_excludes_model_load",
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    load_manifest(partial)
    partial.rename(output / "observations.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-real", action="store_true")
    args = parser.parse_args()
    collect(args.manifest, args.registry, args.output, args.allow_real)


if __name__ == "__main__":
    main()
