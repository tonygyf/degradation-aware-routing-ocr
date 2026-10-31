"""Verify offline CC-OCR readiness on CPU without loading model weights or CUDA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path

from scripts.collect_ccocr import cached_ocr, verify_model
from scripts.prepare_ccocr import file_hash
from src.evaluation.ccocr import image_confidence

MANIFEST_SHA256 = "22149779ff2f61d41a12c322a16e2ae90802693c5406d8639810f91f8cc59ad9"
COUNTS = {"TotalText": 300, "InverseText": 500, "Hieragent": 100}


def check_inputs(registry: Path, manifest: Path, observations: Path) -> dict:
    spec = json.loads(registry.read_text())
    if not spec.get("data_authorization"):
        raise ValueError("Missing data authorization")
    if file_hash(manifest) != MANIFEST_SHA256:
        raise ValueError("External test manifest differs from the frozen 900-image set")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if Counter(row["subset"] for row in rows) != COUNTS:
        raise ValueError("Incomplete external test set")
    for row in rows:
        if row["split"] != "test" or row.get("synthetic") is not False:
            raise ValueError("Expected real frozen external test images")
        if file_hash(manifest.parent / row["image"]) != row["image_sha256"]:
            raise ValueError(f"Image changed: {row['id']}")
    cached = cached_ocr(observations, rows)
    if set(cached) != {row["id"] for row in rows}:
        raise ValueError("OCR cache must contain exactly the frozen test IDs")
    if any(
        row.get("latency_kind") != "measured_serial_inference_ms_excludes_model_load"
        for row in cached.values()
    ):
        raise ValueError("Expected measured real OCR observations")
    for row in cached.values():
        image_confidence(row["ocr"])
    models = {model["role"]: model for model in spec["models"]}
    if set(models) != {"detector", "recognizer", "vlm"}:
        raise ValueError("Expected detector, recognizer and VLM snapshots")
    for model in models.values():
        record = verify_model(model)
        if any(record.get(key) != model[key] for key in ("repository", "license")):
            raise ValueError(f"Model identity or license differs: {model['role']}")
    return {
        "samples": len(rows),
        "manifest_sha256": file_hash(manifest),
        "ocr_observations_sha256": file_hash(observations),
        "models": {role: model["revision"] for role, model in models.items()},
    }


def check_runtime(registry: Path, manifest: Path) -> dict:
    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    assert Qwen3VLForConditionalGeneration is not None
    spec = json.loads(registry.read_text())
    model = next(model for model in spec["models"] if model["role"] == "vlm")
    root = Path(model["directory"])
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in sorted(set(index.values())):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            expected = {name for name, filename in index.items() if filename == shard}
            if set(handle.keys()) != expected:
                raise ValueError(f"Qwen tensor index differs: {shard}")
    processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    row = json.loads(manifest.read_text().splitlines()[0])
    with Image.open(manifest.parent / row["image"]) as source:
        inputs = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": source.convert("RGB")},
                        {"type": "text", "text": spec["prompt"]},
                    ],
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    return {
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("paddleocr", "paddlepaddle", "transformers", "torch", "safetensors")
        },
        "qwen_indexed_tensors": len(index),
        "processor_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "cuda_initialized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/ccocr_models_gpu.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/ccocr/manifest.jsonl"))
    parser.add_argument(
        "--ocr-observations",
        type=Path,
        default=Path("outputs/ccocr-ocr-mkldnn-full/observations.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_inputs(args.registry, args.manifest, args.ocr_observations)
    report.update(check_runtime(args.registry, args.manifest))
    report.update(ready=True, runtime_image_id=os.environ.get("CCOCR_IMAGE_ID", "unrecorded"))
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
