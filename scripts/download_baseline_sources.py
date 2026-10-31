"""Explicitly acquire pinned source archives; no author split or training data creation."""

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.prepare_ccocr import download, file_hash


def acquire(registry, *, allow_download=False):
    if not allow_download:
        raise ValueError("Explicit --allow-download is required")
    spec = json.loads(registry.read_text())
    if (
        spec["training_permitted_by_project"] is not False
        or spec["author_manifest_available"] is not False
    ):
        raise ValueError("Raw source cannot be represented as author dataset or training-ready")
    url = f"https://huggingface.co/api/datasets/{spec['repository']}/revision/{spec['revision']}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != spec["revision"]:
        raise ValueError("Source revision changed")
    siblings = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    root = Path(spec["directory"])
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / "download-provenance.json"
    report = {
        "source": spec["repository"],
        "revision": spec["revision"],
        "status": "downloading",
        "registry_sha256": file_hash(registry),
        "files": [],
    }
    provenance.write_text(json.dumps(report, indent=2))
    for item in spec["files"]:
        name = Path(item["name"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Invalid source artifact path")
        upstream = siblings[item["name"]]
        if upstream["size"] != item["size"] or upstream["lfs"]["sha256"] != item["sha256"]:
            raise ValueError("Source artifact identity changed")
        artifact_url = (
            f"https://huggingface.co/datasets/{spec['repository']}/resolve/"
            f"{spec['revision']}/{item['name']}?download=true"
        )
        for attempt in range(20):
            try:
                download(artifact_url, root / name, item["sha256"], item["size"])
                break
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.IncompleteRead,
            ):
                if attempt == 19:
                    raise
                time.sleep(min(5 + attempt, 30))
        report["files"].append({**item, "local_sha256": file_hash(root / name)})
        print("Verified source:", item["name"], flush=True)
    report["status"] = "verified"
    provenance.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/baseline_sources.json"))
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    acquire(args.registry, allow_download=args.allow_download)


if __name__ == "__main__":
    main()
