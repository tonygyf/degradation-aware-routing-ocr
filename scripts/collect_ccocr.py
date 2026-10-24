"""Real local CC-OCR observations. Frozen models; no fitting on external test images."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from scripts.prepare_ccocr import file_hash
from src.datasets.manifest import load_manifest


class Paddle:
    def __init__(self, models: dict):
        from paddleocr import PaddleOCR

        self.model = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_detection_model_dir=models["detector"]["directory"],
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_recognition_model_dir=models["recognizer"]["directory"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            cpu_threads=1,
            enable_mkldnn=False,
        )

    def infer(self, image: Path) -> dict:
        start = time.perf_counter()
        texts, scores = [], []
        for page in self.model.predict(input=str(image)):
            texts.extend(page["rec_texts"])
            scores.extend(float(x) for x in page["rec_scores"])
        return {
            "text": "\n".join(texts),
            "rec_texts": texts,
            "confidences": scores,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }


class Qwen:
    def __init__(self, model: dict, registry: dict, device: str):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        torch.manual_seed(42)
        self.device, self.torch = device, torch
        self.prompt, self.max_tokens = registry["prompt"], registry["max_new_tokens"]
        self.processor = AutoProcessor.from_pretrained(
            model["directory"], local_files_only=True, trust_remote_code=False
        )
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                model["directory"],
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .eval()
            .to(device)
        )

    def infer(self, image: Path) -> dict:
        from PIL import Image

        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        start = time.perf_counter()
        with Image.open(image) as source:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": source.convert("RGB")},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        tokens = generated[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        return {
            "text": text,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "output_tokens": tokens.shape[1],
            "hit_token_limit": tokens.shape[1] >= self.max_tokens,
        }


def init_ocr_worker(models):
    global _worker_paddle
    _worker_paddle = Paddle(models)


def infer_ocr_worker(image):
    return _worker_paddle.infer(Path(image))


def parallel_ocr(images, models, workers):
    # Recreate predictors in bounded batches to release native inference caches.
    for start in range(0, len(images), 8):
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=init_ocr_worker,
            initargs=(models,),
        ) as pool:
            yield from pool.map(infer_ocr_worker, images[start : start + 8])


def collect(args) -> None:
    if args.workers > 1 and args.mode != "ocr":
        raise ValueError("Parallel workers are only supported for CPU OCR collection")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if args.device != "cpu" and registry["resources"]["gpu_authorized"] is not True:
        raise ValueError("CPU-only guardrail: record explicit GPU authorization before use")
    models = {model["role"]: model for model in registry["models"]}
    roles = ("detector", "recognizer", "vlm") if args.mode == "both" else ("detector", "recognizer")
    for role in roles:
        model = models[role]
        root = Path(model["directory"])
        record = json.loads((root / "download-provenance.json").read_text())
        if record["status"] != "verified" or record["revision"] != model["revision"]:
            raise ValueError(f"Unverified model: {role}")
        for artifact in record["files"]:
            if file_hash(root / artifact["rfilename"]) != artifact["local_sha256"]:
                raise ValueError(f"Model artifact changed: {role}/{artifact['rfilename']}")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    if any(row["split"] != "test" or row.get("synthetic") is not False for row in rows):
        raise ValueError("Only the real frozen external test manifest is allowed")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("External manifest must have unique, nonempty sample IDs")
    if args.per_subset:
        rows = [
            row
            for name in sorted({r["subset"] for r in rows})
            for row in [r for r in rows if r["subset"] == name][: args.per_subset]
        ]
    for row in rows:
        image = args.manifest.parent / row["image"]
        if file_hash(image) != row["image_sha256"]:
            raise ValueError(f"Image changed: {row['id']}")
    meta = {
        "manifest_sha256": file_hash(args.manifest),
        "registry_sha256": file_hash(args.registry),
        "collector_sha256": file_hash(Path(__file__)),
        "mode": args.mode,
        "device": args.device,
        "per_subset": args.per_subset,
        "workers": args.workers,
        "cpu_threads_per_ocr_worker": 1,
        "max_tasks_per_ocr_pool": 8,
        "seed": 42,
        "warmup": "none; first-request overhead included, model loading excluded",
        "counterfactual_vlm_collection": args.mode == "both",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime_image_id": os.environ.get("CCOCR_IMAGE_ID", "unrecorded"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("paddleocr", "paddlepaddle", "transformers", "torch")
        },
    }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    metadata = args.output / "metadata.json"
    if args.resume:
        if json.loads(metadata.read_text()) != meta:
            raise ValueError("Resume configuration or runtime differs from the original run")
    else:
        metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    partial = args.output / "partial.jsonl"
    completed = (
        load_manifest(partial)
        if args.resume and partial.exists() and partial.stat().st_size
        else []
    )
    done = {row["id"] for row in completed}
    selected = {row["id"]: row for row in rows}
    if any(
        row["id"] not in selected
        or any(row[key] != selected[row["id"]][key] for key in ("reference", "image_sha256"))
        for row in completed
    ):
        raise ValueError("Resume observations differ from the selected frozen samples")
    paddle = Paddle(models) if args.workers == 1 else None
    qwen = Qwen(models["vlm"], registry, args.device) if args.mode == "both" else None
    pending = [row for row in rows if row["id"] not in done]
    images = [str((args.manifest.parent / row["image"]).resolve()) for row in pending]
    with partial.open("a" if args.resume else "x", encoding="utf-8") as handle:
        observations = (
            parallel_ocr(images, models, args.workers)
            if args.workers > 1
            else map(lambda image: paddle.infer(Path(image)), images)
        )
        for row, image, ocr in zip(pending, images, observations, strict=True):
            observation = {
                **row,
                "image": str(image),
                "ocr": ocr,
                "latency_kind": "measured_serial_inference_ms_excludes_model_load",
            }
            if qwen is not None:
                observation["vlm"] = qwen.infer(Path(image))
            handle.write(json.dumps(observation, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(row["id"])
            print(f"Collected {len(done)}/{len(rows)}: {row['id']}", flush=True)
    load_manifest(partial)
    partial.rename(args.output / "observations.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/ccocr/manifest.jsonl"))
    parser.add_argument("--registry", type=Path, default=Path("configs/ccocr_models.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("ocr", "both"), default="ocr")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--per-subset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 3))
    args = parser.parse_args()
    if args.per_subset < 0:
        parser.error("--per-subset must be nonnegative")
    collect(args)


if __name__ == "__main__":
    main()
