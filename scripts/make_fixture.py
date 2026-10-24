"""Create authored synthetic SVGs and mocked predictions; never a model benchmark."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "data/fixtures/images"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, count in (("train", 32), ("val", 12), ("test", 16)):
        for i in range(count):
            sample_id = f"{split}-{i:03d}"
            mode = i % 8
            reference = "" if mode == 7 else f"实验 {split} {i:03d} OCR"
            if mode in {0, 1, 2}:
                ocr, vlm, scores = reference.replace("OCR", "0CR"), reference, [0.25 + mode * 0.12]
            elif mode == 3:
                ocr, vlm, scores = reference, reference + " guessed", [0.45]
            elif mode == 4:
                ocr, vlm, scores = reference, reference, [0.98, 0.96]
            elif mode == 5:
                ocr, vlm, scores = reference.replace("OCR", "OCX"), reference, [0.97]
            elif mode == 6:
                ocr, vlm, scores = "", reference, []
            else:
                ocr, vlm, scores = "", "幻觉文字", []
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80">'
                f"<!-- independent synthetic source {sample_id} -->"
                f'<rect width="400" height="80" fill="#{20 + mode * 5:02x}2020"/>'
                f'<text x="10" y="45" fill="#999" font-size="24">'
                f"{html.escape(reference)}</text></svg>"
            ).encode()
            path = folder / f"{sample_id}.svg"
            path.write_bytes(svg)
            rows.append(
                {
                    "id": sample_id,
                    "group_id": sample_id,
                    "image": f"images/{sample_id}.svg",
                    "image_sha256": hashlib.sha256(svg).hexdigest(),
                    "split": split,
                    "reference": reference,
                    "license": "project-authored-synthetic-fixture",
                    "synthetic": True,
                    "latency_kind": "synthetic_cached_ms_excludes_routing",
                    "ocr": {"text": ocr, "confidences": scores, "latency_ms": 12.0},
                    "vlm": {"text": vlm, "latency_ms": 120.0},
                }
            )
    manifest = ROOT / "data/fixtures/observations.jsonl"
    manifest.write_bytes(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    )
    print(manifest)


if __name__ == "__main__":
    main()
