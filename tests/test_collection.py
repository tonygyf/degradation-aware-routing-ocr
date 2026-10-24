import hashlib
import json

import pytest

from src.cache import collect
from src.datasets.manifest import load_manifest


def test_collection_requires_explicit_real_flag(tmp_path):
    with pytest.raises(ValueError, match="allow-real"):
        collect(tmp_path / "absent", tmp_path / "absent", tmp_path / "out", False)
    assert not (tmp_path / "out").exists()


def test_collection_contract_and_hash_gate(tmp_path, monkeypatch):
    import src.cache as cache

    registry = {
        "reviewed": True,
        "data_authorized": True,
        "device": "cpu",
        "paddleocr_package": "3.2.0",
        "paddlepaddle_package": "3.2.0",
        "paddle_detection_dir": "unused",
        "paddle_recognition_dir": "unused",
        "qwen_endpoint": "http://127.0.0.1:8000/v1/chat/completions",
    }
    for field in (
        "dataset_license",
        "dataset_source",
        "split_protocol",
        "ocr_model_revision",
        "ocr_model_license",
        "qwen_model_id",
        "qwen_revision",
        "qwen_model_license",
        "serving_version",
        "resources",
    ):
        registry[field] = "mock-contract-only"
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    image = tmp_path / "image.png"
    image.write_bytes(b"contract-image")
    row = {
        "id": "s1",
        "group_id": "s1",
        "split": "train",
        "reference": "OCR",
        "license": "mock-contract",
        "image": "image.png",
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(cache.importlib.metadata, "version", lambda name: "3.2.0")
    calls = []

    class Backend:
        def __init__(self, *args):
            calls.append("init")

        def infer(self, path):
            assert path == image
            calls.append("infer")
            return {"text": "OCR", "confidences": [0.9], "latency_ms": 1}

    monkeypatch.setattr(cache, "PaddleOCRAdapter", Backend)
    monkeypatch.setattr(cache, "QwenVLHTTPAdapter", Backend)
    collect(manifest, registry_path, tmp_path / "out", True)
    result = load_manifest(tmp_path / "out/observations.jsonl")
    assert len(result) == 1
    assert result[0]["registry_sha256"]
    assert result[0]["latency_kind"] == "measured_serial_inference_ms_excludes_model_load"
    assert calls == ["init", "init", "infer", "infer"]
    calls.clear()
    image.write_bytes(b"changed-image")
    with pytest.raises(ValueError, match="hash mismatch"):
        collect(manifest, registry_path, tmp_path / "invalid", True)
    assert calls == []
