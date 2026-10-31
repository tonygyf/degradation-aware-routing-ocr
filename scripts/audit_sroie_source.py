"""Audit raw converted SROIE archives; does not create author roles or references."""

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from scripts.prepare_ccocr import file_hash


def summarize(records, external_hashes):
    rows = list(records)
    ids, images, splits, image_splits = Counter(), Counter(), {}, {}
    info = {}
    for row in rows:
        ids[row["key"]] += 1
        images[row["image_sha256"]] += 1
        splits.setdefault(row["key"], set()).add(row["source_split"])
        if row["image_sha256"]:
            image_splits.setdefault(row["image_sha256"], set()).add(row["source_split"])
        item = info.setdefault(row["source_split"], Counter())
        item["samples"] += 1
        item["empty_image"] += int(not row["image_sha256"])
        item["empty_regions"] += int(row["region_count"] == 0)
        item["region_box_mismatch"] += int(row["region_count"] != row["box_count"])
        item["ccocr_byte_overlap"] += int(row["image_sha256"] in external_hashes)
        item["contains_paper_excluded_id"] += int(row["key"].lower() == "x51006619503")
    return {
        "samples": len(rows),
        "source_splits": {name: dict(counts) for name, counts in info.items()},
        "duplicate_id_rows": sum(count - 1 for count in ids.values()),
        "duplicate_image_byte_rows": sum(count - 1 for count in images.values()),
        "id_cross_source_splits": sum(len(groups) > 1 for groups in splits.values()),
        "image_byte_cross_source_splits": sum(len(groups) > 1 for groups in image_splits.values()),
        "author_role_manifest_available": False,
        "author_full_transcript_verified": False,
        "overlap_limit": (
            "Exact encoded bytes only; different encodings and visually similar sources can overlap"
        ),
    }


def read_archives(root, provenance):
    import pyarrow.parquet as parquet

    for item in provenance["files"]:
        path = root / item["name"]
        if file_hash(path) != item["sha256"] or item["local_sha256"] != item["sha256"]:
            raise ValueError("Source archive hash changed")
        split = path.name.split("-")[0]
        file = parquet.ParquetFile(path)
        for batch in file.iter_batches(batch_size=16, columns=["key", "image", "words", "bboxes"]):
            for row in batch.to_pylist():
                payload = row["image"]["bytes"] if row["image"] else None
                yield {
                    "key": row["key"],
                    "source_split": split,
                    "image_sha256": hashlib.sha256(payload).hexdigest() if payload else "",
                    "region_count": len(row["words"]) if row["words"] is not None else 0,
                    "box_count": len(row["bboxes"]) if row["bboxes"] is not None else 0,
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/sroie-reference/source"))
    parser.add_argument(
        "--external-manifest", type=Path, default=Path("data/raw/ccocr/manifest.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance_path = args.source / "download-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance["status"] != "verified":
        raise ValueError("Source artifacts must be verified first")
    external = [
        json.loads(line) for line in args.external_manifest.read_text().splitlines() if line
    ]
    report = {
        "audit_script_sha256": file_hash(Path(__file__)),
        "source_provenance_sha256": file_hash(provenance_path),
        "external_manifest_sha256": file_hash(args.external_manifest),
        "pyarrow_version": importlib.metadata.version("pyarrow"),
        **summarize(
            read_archives(args.source, provenance), {row["image_sha256"] for row in external}
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
