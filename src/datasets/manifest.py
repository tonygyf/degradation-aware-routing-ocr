from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("Manifest is empty")
    ids, groups, contents = set(), {}, {}
    for row in rows:
        for field in ("id", "group_id", "image_sha256", "split", "reference", "ocr", "license"):
            if field not in row:
                raise ValueError(f"Missing field: {field}")
        if not all(
            isinstance(row[k], str) and row[k]
            for k in ("id", "group_id", "image_sha256", "license")
        ):
            raise ValueError("Identity, hash and license must be nonempty strings")
        if row["id"] in ids:
            raise ValueError("Duplicate sample ID")
        ids.add(row["id"])
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        for key, seen in (("group_id", groups), ("image_sha256", contents)):
            if row[key] in seen and seen[row[key]] != split:
                raise ValueError(f"Cross-split leakage: {key}")
            seen[row[key]] = split
        if len(row["image_sha256"]) != 64 or any(
            c not in "0123456789abcdef" for c in row["image_sha256"]
        ):
            raise ValueError("image_sha256 must be lowercase SHA-256")
        if not isinstance(row["reference"], str):
            raise ValueError("Reference must be a string, including for blank images")
        for key in ("ocr", "vlm"):
            if key not in row:
                continue
            obs = row[key]
            if not isinstance(obs["text"], str):
                raise ValueError("Prediction must be text")
            ms = obs["latency_ms"]
            if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms < 0:
                raise ValueError("Invalid latency")
        scores = row["ocr"]["confidences"]
        if not isinstance(scores, list) or any(
            not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1
            for x in scores
        ):
            raise ValueError("Invalid OCR confidence")
    return rows
