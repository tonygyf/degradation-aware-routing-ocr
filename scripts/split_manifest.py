"""Freeze source-image group assignments; never silently rewrite an existing split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def assign(rows: list[dict], seed: int = 42) -> list[dict]:
    seen, output = {}, []
    for row in rows:
        if "split" in row:
            raise ValueError("Input already has splits; do not repartition a frozen manifest")
        group = row["group_id"]
        if not isinstance(group, str) or not group:
            raise ValueError("Nonempty original source-image group_id required")
        value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) / (2**256)
        split = "train" if value < 0.7 else "val" if value < 0.85 else "test"
        digest = row["image_sha256"]
        if digest in seen and seen[digest] != split:
            raise ValueError("Duplicate image crosses groups/splits; fix source grouping first")
        seen[digest] = split
        output.append({**row, "split": split})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x]
    result = assign(rows, args.seed)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in result)


if __name__ == "__main__":
    main()
