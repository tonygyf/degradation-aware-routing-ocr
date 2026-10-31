import json

import pytest

from scripts import preflight_ccocr
from scripts.prepare_ccocr import file_hash


def test_preflight_checks_real_inputs_and_rejects_corruption(tmp_path, monkeypatch):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"image bytes")
    row = {
        "id": "a",
        "group_id": "original:a",
        "image": image.name,
        "image_sha256": file_hash(image),
        "reference": "A",
        "license": "research",
        "subset": "TotalText",
        "split": "test",
        "synthetic": False,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    observations = tmp_path / "observations.jsonl"
    cached = {
        **row,
        "ocr": {"text": "A", "rec_texts": ["A"], "confidences": [0.95], "latency_ms": 1},
        "latency_kind": "measured_serial_inference_ms_excludes_model_load",
    }
    observations.write_text(json.dumps(cached) + "\n")
    models = []
    for role in ("detector", "recognizer", "vlm"):
        root = tmp_path / role
        root.mkdir()
        artifact = root / "weights"
        artifact.write_bytes(b"model bytes")
        model = {
            "role": role,
            "directory": str(root),
            "revision": "pinned",
            "repository": f"owner/{role}",
            "license": "apache-2.0",
        }
        (root / "download-provenance.json").write_text(
            json.dumps(
                {
                    **model,
                    "status": "verified",
                    "files": [{"rfilename": "weights", "local_sha256": file_hash(artifact)}],
                }
            )
        )
        models.append(model)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"data_authorization": "user requested", "models": models}))
    monkeypatch.setattr(preflight_ccocr, "MANIFEST_SHA256", file_hash(manifest))
    monkeypatch.setattr(preflight_ccocr, "COUNTS", {"TotalText": 1})
    assert preflight_ccocr.check_inputs(registry, manifest, observations)["samples"] == 1
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image changed"):
        preflight_ccocr.check_inputs(registry, manifest, observations)
    image.write_bytes(b"image bytes")
    observations.write_text(json.dumps({**cached, "reference": "changed"}) + "\n")
    with pytest.raises(ValueError, match="OCR cache does not match"):
        preflight_ccocr.check_inputs(registry, manifest, observations)
    observations.write_text(json.dumps(cached) + "\n")
    (tmp_path / "vlm" / "weights").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Model artifact changed"):
        preflight_ccocr.check_inputs(registry, manifest, observations)
