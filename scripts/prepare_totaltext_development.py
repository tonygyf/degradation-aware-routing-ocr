"""Audit official Total-Text train images against the frozen external test set.

Run in the CPU CC-OCR image (Pillow and SciPy). Outputs are private candidates,
not ready-to-train full-image references or scientific routing results.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from scripts.prepare_ccocr import file_hash


def fingerprint(path: Path) -> dict:
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        pixels = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
        thumb = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    bits = [thumb[y * 9 + x] > thumb[y * 9 + x + 1] for y in range(8) for x in range(8)]
    return {"pixel_sha256": pixels, "dhash": sum(int(v) << i for i, v in enumerate(bits))}


def audit(rows: list[dict], external: list[dict], distance: int = 4) -> tuple[list[dict], dict]:
    if not 0 <= distance <= 64:
        raise ValueError("dHash distance must be in [0, 64]")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate development ID")
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def similar(a, b):
        return (
            a["source_id"] == b["source_id"]
            or a["image_sha256"] == b["image_sha256"]
            or a["pixel_sha256"] == b["pixel_sha256"]
            or (a["dhash"] ^ b["dhash"]).bit_count() <= distance
        )

    overlaps = []
    blocked = set()
    for i, row in enumerate(rows):
        for other in external:
            if similar(row, other):
                blocked.add(i)
                overlaps.append(
                    {
                        "candidate": row["id"],
                        "external": other["id"],
                        "source_match": row["source_id"] == other["source_id"],
                        "file_match": row["image_sha256"] == other["image_sha256"],
                        "pixel_match": row["pixel_sha256"] == other["pixel_sha256"],
                        "dhash_distance": (row["dhash"] ^ other["dhash"]).bit_count(),
                    }
                )
        for j in range(i):
            if similar(row, rows[j]):
                parents[find(i)] = find(j)
    components = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    excluded_groups = {find(i) for i in blocked}
    output = []
    for parent, members in components.items():
        if parent in excluded_groups:
            continue
        group = min(rows[i]["source_id"] for i in members)
        bucket = int(hashlib.sha256(f"42:{group}".encode()).hexdigest(), 16) / 2**256
        for i in members:
            output.append(
                {**rows[i], "group_id": group, "split": "train" if bucket < 0.8 else "val"}
            )
    output.sort(key=lambda row: row["id"])
    return output, {
        "candidates": len(rows),
        "retained": len(output),
        "excluded": len(rows) - len(output),
        "overlap_pairs": overlaps,
        "dhash_distance": distance,
        "source_groups": len(components),
        "split_counts": {s: sum(r["split"] == s for r in output) for s in ("train", "val")},
        "training_ready": False,
        "limitations": [
            "dHash screening cannot prove absence of crops or near duplicates",
            "Full-image annotation reading order requires review before CER labels",
        ],
    }


def read_annotations(data: bytes) -> list[dict]:
    from scipy.io import loadmat

    words = loadmat(io.BytesIO(data))["gt"]
    result = []
    for word in words:
        text = "".join(word[4].reshape(-1).tolist())
        orientation = "".join(word[5].reshape(-1).tolist())
        xs, ys = word[1].reshape(-1).tolist(), word[3].reshape(-1).tolist()
        if len(xs) != len(ys) or not xs:
            raise ValueError("Invalid annotation polygon")
        result.append(
            {
                "text": text,
                "polygon": list(zip(xs, ys, strict=True)),
                "ignore": orientation == "#" or text == "#",
                "orientation": orientation,
            }
        )
    return result


def prepare(config: Path, external_manifest: Path, output: Path) -> dict:
    spec = json.loads(config.read_text())
    if file_hash(external_manifest) != spec["external_test_manifest_sha256"]:
        raise ValueError("External test manifest changed")
    source = Path("data/raw/totaltext-development/source")
    artifacts = {item["role"]: source / item["filename"] for item in spec["artifacts"]}
    for item in spec["artifacts"]:
        if file_hash(source / item["filename"]) != item["sha256"]:
            raise ValueError("Downloaded source artifact changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    rows = []
    with (
        zipfile.ZipFile(artifacts["images"]) as images,
        zipfile.ZipFile(artifacts["refined_groundtruth"]) as gt,
    ):
        image_files = [
            n
            for n in images.namelist()
            if "Train" in Path(n).parts
            and "__MACOSX" not in Path(n).parts
            and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if len(image_files) != spec["official_train_count"]:
            raise ValueError("Official training image count differs")
        for name in sorted(image_files):
            filename = Path(name).name
            if ".." in Path(name).parts or "\\" in name:
                raise ValueError("Invalid archive image path")
            image = output / "images" / filename
            if image.exists():
                raise ValueError("Duplicate image filename")
            image.write_bytes(images.read(name))
            annotations = read_annotations(gt.read(f"Train/gt_{image.stem}.mat"))
            rows.append(
                {
                    "id": f"totaltext-train/{image.stem}",
                    "source_id": f"totaltext:{image.stem.lower()}",
                    "image": str(image.relative_to(output)),
                    "image_sha256": file_hash(image),
                    "official_split": "train",
                    "synthetic": False,
                    "license": spec["license"],
                    "word_annotations": annotations,
                    "reference": "\n".join(a["text"] for a in annotations if not a["ignore"]),
                    "reference_status": "annotation_order_unreviewed",
                    **fingerprint(image),
                }
            )
    external = []
    for line in external_manifest.read_text().splitlines():
        row = json.loads(line)
        path = external_manifest.parent / row["image"]
        if row["split"] != "test" or row.get("synthetic") is not False:
            raise ValueError("Expected real frozen external test rows")
        if file_hash(path) != row["image_sha256"]:
            raise ValueError("External test image changed")
        external.append(
            {
                **row,
                "source_id": f"{row['subset'].lower()}:{path.stem.lower()}",
                **fingerprint(path),
            }
        )
    retained, report = audit(rows, external)
    report.update(
        external_manifest_sha256=file_hash(external_manifest),
        external_samples=len(external),
        source_artifacts=spec["artifacts"],
        repository_revision=spec["repository_revision"],
        approved_for_model_training=False,
    )
    for filename, items in (("candidates.jsonl", rows), ("development.jsonl", retained)):
        (output / filename).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items)
        )
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/totaltext_development.json"))
    parser.add_argument(
        "--external-manifest", type=Path, default=Path("data/raw/ccocr/manifest.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/totaltext-development/prepared")
    )
    args = parser.parse_args()
    report = prepare(args.config, args.external_manifest, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "overlap_pairs"}, indent=2))


if __name__ == "__main__":
    main()
