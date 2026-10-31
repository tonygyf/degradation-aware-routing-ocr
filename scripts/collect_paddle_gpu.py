"""Recollect frozen CC-OCR Paddle output on an isolated GPU and replay Qwen."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.collect_ccocr import cached_ocr, verify_model
from scripts.preflight_ccocr import MANIFEST_SHA256
from scripts.prepare_ccocr import file_hash
from src.datasets.manifest import load_manifest


def require_idle_gpu(resources):
    uuid = resources["authorization"]["host_gpu_uuid"]
    if resources["gpu_authorized"] is not True:
        raise ValueError("GPU authorization required")
    listing = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines()
    if len(listing) != 1 or uuid not in listing[0]:
        raise ValueError("Require exactly the authorized GPU")
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
    if processes or free < 40960:
        raise ValueError("Authorized GPU is busy or has less than 40 GiB free")


class PaddleGPU:
    def __init__(self, models):
        import paddle
        from paddleocr import PaddleOCR

        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() != 1:
            raise ValueError("Expected CUDA Paddle with exactly one visible GPU")
        self.paddle = paddle
        self.model = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_detection_model_dir=models["detector"]["directory"],
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_recognition_model_dir=models["recognizer"]["directory"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu:0",
            text_recognition_batch_size=6,
        )

    def infer(self, image):
        self.paddle.device.cuda.synchronize()
        start = time.perf_counter()
        texts, scores = [], []
        for page in self.model.predict(input=str(image)):
            texts.extend(page["rec_texts"])
            scores.extend(float(score) for score in page["rec_scores"])
        self.paddle.device.cuda.synchronize()
        return {
            "text": "\n".join(texts),
            "rec_texts": texts,
            "confidences": scores,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }


def collect(args):
    registry = json.loads(args.registry.read_text())
    models = {model["role"]: model for model in registry["models"]}
    for role in ("detector", "recognizer"):
        verify_model(models[role])
    if file_hash(args.manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen external manifest changed")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    cached = cached_ocr(args.qwen_observations, rows)
    if set(cached) != {row["id"] for row in rows} or any(
        "vlm" not in row for row in cached.values()
    ):
        raise ValueError("Require complete frozen Qwen observations")
    if args.per_subset:
        rows = [
            row
            for name in sorted({row["subset"] for row in rows})
            for row in [r for r in rows if r["subset"] == name][: args.per_subset]
        ]
    for row in rows:
        if file_hash(args.manifest.parent / row["image"]) != row["image_sha256"]:
            raise ValueError("Frozen image changed")
    meta = {
        "registry_sha256": file_hash(args.registry),
        "manifest_sha256": MANIFEST_SHA256,
        "qwen_observations_sha256": file_hash(args.qwen_observations),
        "collector_sha256": file_hash(Path(__file__)),
        "per_subset": args.per_subset,
        "runtime_image_id": os.environ.get("PADDLE_GPU_IMAGE_ID", "unrecorded"),
        "paddle_gpu": importlib.metadata.version("paddlepaddle-gpu"),
        "paddleocr": importlib.metadata.version("paddleocr"),
        "latency_note": "GPU serial measured, no warmup; Qwen cached counterfactual",
    }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    metadata, partial, final = (
        args.output / name for name in ("metadata.json", "partial.jsonl", "observations.jsonl")
    )
    if args.resume:
        if json.loads(metadata.read_text()) != meta:
            raise ValueError("Resume configuration or runtime changed")
    else:
        metadata.write_text(json.dumps(meta, indent=2))
    if final.exists():
        raise ValueError("Completed observations already exist")
    completed = (
        load_manifest(partial)
        if args.resume and partial.exists() and partial.stat().st_size
        else []
    )
    selected = {row["id"]: row for row in rows}
    if any(
        row["id"] not in selected
        or any(
            row[key] != selected[row["id"]][key]
            for key in ("reference", "image_sha256", "group_id")
        )
        for row in completed
    ):
        raise ValueError("Resume samples differ from frozen inputs")
    done = {row["id"] for row in completed}
    require_idle_gpu(registry["resources"])
    model = PaddleGPU(models)
    with partial.open("a" if args.resume else "x", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            image = (args.manifest.parent / row["image"]).resolve()
            result = {
                **row,
                "image": str(image),
                "ocr": model.infer(image),
                "vlm": cached[row["id"]]["vlm"],
                "latency_kind": "measured_gpu_paddle_reused_gpu_qwen",
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(row["id"])
            print(f"paddle-gpu: {len(done)}/{len(rows)} {row['id']}", flush=True)
    load_manifest(partial)
    partial.rename(final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/ccocr_paddle_gpu.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/ccocr/manifest.jsonl"))
    parser.add_argument(
        "--qwen-observations", type=Path, default=Path("outputs/ccocr-gpu-full/observations.jsonl")
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
