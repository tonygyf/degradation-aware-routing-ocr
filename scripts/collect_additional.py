"""Collect pinned InternVL/DeepSeek outputs on the frozen external CC-OCR set."""

import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from scripts.collect_ccocr import cached_ocr, verify_model
from scripts.preflight_ccocr import MANIFEST_SHA256
from scripts.prepare_ccocr import file_hash
from src.datasets.manifest import load_manifest
from src.evaluation.external import evaluate
from src.recognition.external import ExternalModel


def verify_custom_code(spec):
    root = Path(spec["directory"])
    files = spec["custom_code_files"]
    if not files:
        raise ValueError("Require pinned custom code hashes")
    expected = set()
    for item in files:
        path = Path(item["rfilename"])
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError("Invalid custom code path")
        if file_hash(root / path) != item["local_sha256"]:
            raise ValueError("Custom code hash mismatch")
        expected.add(path)
    if {path.relative_to(root) for path in root.rglob("*.py")} != expected:
        raise ValueError("Unexpected custom Python files")


def require_idle_gpu(resources):
    if resources["gpu_authorized"] is not True:
        raise ValueError("GPU authorization required")
    uuid = resources["authorization"]["host_gpu_uuid"]
    processes = subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    free = int(
        subprocess.check_output(
            ["nvidia-smi", "-i", uuid, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    if processes or free < resources["min_free_vram_mib"]:
        raise ValueError("Assigned GPU is busy or lacks free memory; leave existing work untouched")
    import torch

    if torch.cuda.device_count() != 1:
        raise ValueError("Require exactly one visible GPU")
    actual = str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if actual != uuid.removeprefix("GPU-"):
        raise ValueError("Visible GPU UUID mismatch")


def collect(args):
    registry = json.loads(args.registry.read_text())
    spec = next(model for model in registry["models"] if model["role"] == args.model)
    verify_model(spec)
    verify_custom_code(spec)
    if file_hash(args.manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen external manifest changed")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    cached = cached_ocr(args.ocr_observations, rows)
    if args.per_subset:
        rows = [
            row
            for name in sorted({row["subset"] for row in rows})
            for row in [r for r in rows if r["subset"] == name][: args.per_subset]
        ]
    for row in rows:
        if file_hash(args.manifest.parent / row["image"]) != row["image_sha256"]:
            raise ValueError("Frozen image changed")
    sources = [
        Path(__file__),
        Path("src/recognition/external.py"),
        Path("src/recognition/internvl_preprocess.py"),
        Path("scripts/collect_ccocr.py"),
    ]
    meta = {
        "registry_sha256": file_hash(args.registry),
        "manifest_sha256": MANIFEST_SHA256,
        "ocr_sha256": file_hash(args.ocr_observations),
        "model": args.model,
        "per_subset": args.per_subset,
        "seed": 42,
        "source_sha256": {
            str(path.relative_to(Path.cwd()) if path.is_absolute() else path): file_hash(path)
            for path in sources
        },
        "runtime_image_id": os.environ.get("EXTERNAL_IMAGE_ID", "unrecorded"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers")
        },
        "latency_kind": (
            "measured serial preprocessing/inference/decoding; no warmup; model load excluded"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    metadata = args.output / "metadata.json"
    if args.resume:
        if json.loads(metadata.read_text()) != meta:
            raise ValueError("Resume configuration or runtime changed")
    else:
        metadata.write_text(json.dumps(meta, indent=2))
    final = args.output / "observations.jsonl"
    partial = args.output / "partial.jsonl"
    if not final.exists():
        completed = (
            load_manifest(partial)
            if args.resume and partial.exists() and partial.stat().st_size
            else []
        )
        selected = {row["id"]: row for row in rows}
        for row in completed:
            if (
                row["id"] not in selected
                or row.get("system") != args.model
                or any(
                    row[key] != selected[row["id"]][key]
                    for key in ("reference", "image_sha256", "group_id")
                )
            ):
                raise ValueError("Resume samples differ from frozen inputs")
        done = {row["id"] for row in completed}
        require_idle_gpu(registry["resources"])
        model = ExternalModel(spec)
        with partial.open("a" if args.resume else "x", encoding="utf-8") as handle:
            for row in rows:
                if row["id"] in done:
                    continue
                image = (args.manifest.parent / row["image"]).resolve()
                observation = {
                    **row,
                    "image": str(image),
                    "ocr": cached[row["id"]]["ocr"],
                    "vlm": model.infer(image),
                    "system": args.model,
                    "latency_kind": meta["latency_kind"],
                }
                handle.write(json.dumps(observation, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                done.add(row["id"])
                print(f"{args.model}: {len(done)}/{len(rows)} {row['id']}", flush=True)
        load_manifest(partial)
        partial.rename(final)
    finished = load_manifest(final)
    selected = {row["id"]: row for row in rows}
    if {row["id"] for row in finished} != set(selected) or any(
        any(
            row[key] != selected[row["id"]][key]
            for key in ("reference", "image_sha256", "group_id")
        )
        for row in finished
    ):
        raise ValueError("Completed observations do not match selected frozen inputs")
    report = {"observations_sha256": file_hash(final), **evaluate(finished, args.model)}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("internvl", "deepseek"), required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/ccocr_additional_gpu.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/ccocr/manifest.jsonl"))
    parser.add_argument(
        "--ocr-observations",
        type=Path,
        default=Path("outputs/ccocr-ocr-mkldnn-full/observations.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-subset", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.per_subset < 0:
        parser.error("--per-subset must be nonnegative")
    collect(args)


if __name__ == "__main__":
    main()
