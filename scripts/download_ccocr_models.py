"""Explicit pinned model download, with license, size and artifact hash records."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.prepare_ccocr import download, file_hash


def snapshot(spec: dict) -> None:
    repository, revision = spec["repository"], spec["revision"]
    url = f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if (
        metadata["sha"] != revision
        or metadata.get("cardData", {}).get("license") != spec["license"]
    ):
        raise ValueError(f"Model revision or license changed: {repository}")
    files = [
        f
        for f in metadata["siblings"]
        if f["rfilename"] != ".gitattributes" and not f["rfilename"].endswith((".py", ".bin"))
    ]
    total = sum(f["size"] for f in files)
    root = Path(spec["directory"])
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < total + 10 * 1024**3:
        raise ValueError("Insufficient free disk for the reviewed snapshot")
    record = {**spec, "total_bytes": total, "files": files, "status": "downloading"}
    provenance = root / "download-provenance.json"
    provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def fetch_once(item):
        name = item["rfilename"]
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid remote artifact path")
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}?download=true"
        print(f"{repository}: {name} ({item['size']} bytes)", flush=True)
        if "lfs" in item:
            if spec.get("weight_mirror") == "modelscope" and name.endswith(
                (".safetensors", ".pdiparams")
            ):
                url = f"https://www.modelscope.cn/models/{repository}/resolve/master/{name}"
            item["transport_url"] = url
            download(url, path, item["lfs"]["sha256"], item["size"])
        else:
            if path.exists():
                data = path.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
            git_blob = b"blob " + str(len(data)).encode() + b"\0" + data
            if hashlib.sha1(git_blob).hexdigest() != item["blobId"]:
                raise ValueError(f"Git blob hash mismatch: {repository}/{name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
        item["local_sha256"] = file_hash(path)

    def fetch(item):
        for attempt in range(4):
            try:
                fetch_once(item)
                return
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.IncompleteRead,
            ) as error:
                print(
                    f"Transport retry {attempt + 1}/4: {repository}/{item['rfilename']} "
                    f"({type(error).__name__})",
                    flush=True,
                )
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
            except Exception as error:
                print(f"Artifact rejected: {repository}/{item['rfilename']}: {error}", flush=True)
                raise

    # Separate shards can progress independently; each is still checked before acceptance.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))
    record["status"] = "verified"
    provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Verified model snapshot: {repository}@{revision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/ccocr_models.json"))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--weight-mirror", choices=("modelscope",))
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("Model download requires explicit --allow-download")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(
            pool.map(
                snapshot,
                [{**spec, "weight_mirror": args.weight_mirror} for spec in registry["models"]],
            )
        )


if __name__ == "__main__":
    main()
