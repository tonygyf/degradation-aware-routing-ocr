"""Run the authorized GPU pilot and full external evaluation after inputs are ready."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REGISTRY = Path("configs/ccocr_models_gpu.json")
OCR = Path("outputs/ccocr-ocr-mkldnn-full/observations.jsonl")


def inputs_ready(registry: Path, observations: Path) -> bool:
    spec = json.loads(registry.read_text())
    if spec["resources"]["gpu_authorized"] is not True:
        raise ValueError("This experiment requires the recorded GPU authorization")
    if not observations.is_file():
        return False
    for model in spec["models"]:
        path = Path(model["directory"]) / "download-provenance.json"
        if not path.is_file():
            return False
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            return False  # A downloader may be updating the local status record.
        if record["status"] != "verified" or record["revision"] != model["revision"]:
            return False
    return True


def main():
    deadline = time.monotonic() + 6 * 60 * 60
    while not inputs_ready(REGISTRY, OCR):
        if time.monotonic() >= deadline:
            raise TimeoutError("Inputs were not ready within 6 hours; no GPU inference was started")
        print(
            "Waiting for verified local weights and complete frozen CPU OCR observations",
            flush=True,
        )
        time.sleep(30)
    for stage, limit in (("pilot", 1), ("full", 0)):
        output = Path(f"outputs/ccocr-gpu-{stage}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.collect_ccocr",
                "--registry",
                str(REGISTRY),
                "--mode",
                "both",
                "--device",
                "cuda:0",
                "--ocr-observations",
                str(OCR),
                "--per-subset",
                str(limit),
                "--output",
                str(output),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.ccocr",
                "--observations",
                str(output / "observations.jsonl"),
                "--output",
                str(output) + "-eval",
            ],
            check=True,
        )
        print(f"Completed {stage} and its evaluation", flush=True)


if __name__ == "__main__":
    main()
