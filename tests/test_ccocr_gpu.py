import json
import subprocess

import pytest

from scripts.collect_ccocr import Qwen, cached_ocr
from scripts.run_ccocr_gpu import inputs_ready


def test_cached_ocr_requires_matching_frozen_identity(tmp_path):
    row = {
        "id": "a",
        "group_id": "original:a",
        "image_sha256": "a" * 64,
        "reference": "A",
        "split": "test",
        "synthetic": False,
        "license": "research",
        "ocr": {"text": "B", "confidences": [0.9], "latency_ms": 1},
    }
    path = tmp_path / "ocr.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert cached_ocr(path, [row])["a"]["ocr"]["text"] == "B"
    for field, changed in (
        ("image_sha256", "b" * 64),
        ("reference", "C"),
        ("group_id", "other"),
        ("synthetic", True),
        ("split", "val"),
    ):
        with pytest.raises(ValueError, match="does not match frozen"):
            cached_ocr(path, [{**row, field: changed}])
    with pytest.raises(ValueError, match="does not match frozen"):
        cached_ocr(path, [{**row, "id": "missing"}])


def test_gpu_workflow_waits_for_complete_inputs_and_requires_authorization(tmp_path):
    registry, observations = tmp_path / "models.json", tmp_path / "observations.jsonl"
    spec = {
        "resources": {"gpu_authorized": True},
        "models": [{"directory": str(tmp_path), "revision": "pinned"}],
    }
    registry.write_text(json.dumps(spec))
    assert not inputs_ready(registry, observations)
    observations.write_text("complete output is validated again by the collector")
    provenance = tmp_path / "download-provenance.json"
    provenance.write_text('{"status":"downloading","revision":"pinned"}')
    assert not inputs_ready(registry, observations)
    provenance.write_text('{"status":"verified","revision":"different"}')
    assert not inputs_ready(registry, observations)
    provenance.write_text('{"status":"verified","revision":"pinned"}')
    assert inputs_ready(registry, observations)
    spec["resources"]["gpu_authorized"] = False
    registry.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="recorded GPU authorization"):
        inputs_ready(registry, observations)


def test_qwen_refuses_busy_gpu_before_loading_weights(monkeypatch):
    def query(command, text):
        return "1234\n" if "--query-compute-apps=pid" in command else "80000\n"

    monkeypatch.setattr("scripts.collect_ccocr.subprocess.check_output", query)
    with pytest.raises(ValueError, match="GPU is busy"):
        Qwen({}, {"resources": {"authorization": {"host_gpu_uuid": "GPU-authorized"}}}, "cuda:0")


def test_gpu_workflow_stops_after_failed_pilot(monkeypatch):
    from scripts import run_ccocr_gpu

    calls = []
    monkeypatch.setattr(run_ccocr_gpu, "inputs_ready", lambda *args: True)

    def fail(command, check):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run_ccocr_gpu.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_ccocr_gpu.main()
    assert len(calls) == 1
    assert "outputs/ccocr-gpu-pilot" in calls[0]
