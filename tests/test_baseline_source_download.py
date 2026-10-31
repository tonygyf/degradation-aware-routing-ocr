import io
import json

import pytest

from scripts import download_baseline_sources
from scripts.prepare_ccocr import file_hash


def test_source_acquisition_requires_explicit_download_and_rejects_changed_upstream(
    tmp_path, monkeypatch
):
    payload = b"fixture data"
    artifact = tmp_path / "fixture"
    artifact.write_bytes(payload)
    digest = file_hash(artifact)
    spec = {
        "repository": "owner/data",
        "revision": "fixed",
        "directory": str(tmp_path / "raw"),
        "author_manifest_available": False,
        "training_permitted_by_project": False,
        "files": [{"name": "data/train.parquet", "size": len(payload), "sha256": digest}],
    }
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Explicit"):
        download_baseline_sources.acquire(registry)
    metadata = {
        "sha": "fixed",
        "siblings": [
            {"rfilename": "data/train.parquet", "size": len(payload), "lfs": {"sha256": "0" * 64}}
        ],
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, timeout: io.BytesIO(json.dumps(metadata).encode())
    )
    with pytest.raises(ValueError, match="identity changed"):
        download_baseline_sources.acquire(registry, allow_download=True)
    metadata["siblings"][0]["lfs"]["sha256"] = digest

    def fake_download(url, path, expected_sha, size):
        assert expected_sha == digest and size == len(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    monkeypatch.setattr(download_baseline_sources, "download", fake_download)
    download_baseline_sources.acquire(registry, allow_download=True)
    report = json.loads((tmp_path / "raw/download-provenance.json").read_text())
    assert report["status"] == "verified"
    assert report["files"][0]["local_sha256"] == digest
    spec["training_permitted_by_project"] = True
    registry.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="training-ready"):
        download_baseline_sources.acquire(registry, allow_download=True)
