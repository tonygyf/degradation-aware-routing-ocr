"""Download the pinned public CC-OCR external test set; never create training rows."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected_sha256: str, size: int | None = None) -> None:
    if path.exists():
        if file_hash(path) != expected_sha256:
            raise ValueError(f"Existing file hash mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if size is None:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(block)
    else:
        position = partial.stat().st_size if partial.exists() else 0
        if position > size:
            raise ValueError("Partial download exceeds the reviewed artifact size")
        while position < size:
            end = min(position + 64 * 1024 * 1024, size) - 1
            request = urllib.request.Request(url, headers={"Range": f"bytes={position}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                )
                if (
                    response.status != 206
                    or not match
                    or int(match[1]) != position
                    or int(match[3]) != size
                ):
                    raise ValueError("Server returned an invalid artifact byte range")
                start = position
                with partial.open("ab") as handle:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        position += len(block)
                        if position > size:
                            raise ValueError("Download exceeds reviewed artifact size")
                        handle.write(block)
                if position == start:
                    raise ValueError("Download made no progress; partial file retained")
    if file_hash(partial) != expected_sha256:
        raise ValueError(f"Downloaded file hash mismatch: {partial}")
    partial.rename(path)


def extract_tsv(path: Path, subset: str, root: Path, license_text: str) -> list[dict]:
    csv.field_size_limit(100 * 1024 * 1024)
    rows = []
    image_dir = root / "images" / subset
    image_dir.mkdir(parents=True, exist_ok=True)
    with path.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            name = item["image_name"]
            if not name or Path(name).name != name or "\\" in name:
                raise ValueError("Image name must be a filename")
            image = image_dir / name
            content = base64.b64decode(item["image"], validate=True)
            digest = hashlib.sha256(content).hexdigest()
            if image.exists() and file_hash(image) != digest:
                raise ValueError(f"Existing image differs: {image}")
            if not image.exists():
                image.write_bytes(content)
            rows.append(
                {
                    "id": f"{subset}/{name}",
                    "group_id": f"ccocr:{subset}/{name}",
                    "image": str(image.relative_to(root)),
                    "image_sha256": digest,
                    "subset": subset,
                    "split": "test",
                    "reference": item["answer"],
                    "license": license_text,
                    "synthetic": False,
                }
            )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate CC-OCR image ID")
    return rows


def prepare(config: Path, output: Path) -> Path:
    spec = json.loads(config.read_text(encoding="utf-8"))
    rows = []
    for source in spec["files"]:
        path = output / "source" / Path(source["path"]).name
        url = (
            f"https://huggingface.co/datasets/{spec['repository']}/resolve/"
            f"{spec['revision']}/{source['path']}"
        )
        print(f"Downloading and verifying {source['subset']}", flush=True)
        download(url, path, source["sha256"])
        subset = extract_tsv(path, source["subset"], output, spec["license"])
        if len(subset) != source["count"]:
            raise ValueError(f"Unexpected subset size: {source['subset']}")
        rows.extend(subset)
    manifest = output / "manifest.jsonl"
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if manifest.exists() and manifest.read_text(encoding="utf-8") != text:
        raise ValueError("Refusing to replace a different frozen manifest")
    manifest.write_text(text, encoding="utf-8")
    provenance = {**spec, "manifest_sha256": file_hash(manifest), "samples": len(rows)}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Frozen {len(rows)} external test images: {manifest}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ccocr_public.json"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/ccocr"))
    args = parser.parse_args()
    prepare(args.config, args.output)


if __name__ == "__main__":
    main()
