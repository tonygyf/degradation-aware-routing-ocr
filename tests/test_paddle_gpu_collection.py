import json
from types import SimpleNamespace

import pytest

from scripts import collect_paddle_gpu
from scripts.prepare_ccocr import file_hash


def test_reject_gpu_when_another_process_is_running(monkeypatch):
    def query(command, text):
        if "-L" in command:
            return "GPU 0: NVIDIA (UUID: GPU-assigned)\n"
        if "--query-compute-apps=pid" in command:
            return "1234\n"
        return "80000\n"

    monkeypatch.setattr(collect_paddle_gpu.subprocess, "check_output", query)
    with pytest.raises(ValueError, match="busy"):
        collect_paddle_gpu.require_idle_gpu(
            {"gpu_authorized": True, "authorization": {"host_gpu_uuid": "GPU-assigned"}}
        )


def test_paddle_gpu_resumes_and_retains_paired_qwen(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    model_root.mkdir()
    artifact = model_root / "weight"
    artifact.write_bytes(b"fixture")
    (model_root / "download-provenance.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "revision": "fixed",
                "files": [{"rfilename": "weight", "local_sha256": file_hash(artifact)}],
            }
        )
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "resources": {},
                "models": [
                    {"role": role, "revision": "fixed", "directory": str(model_root)}
                    for role in ("detector", "recognizer")
                ],
            }
        )
    )
    rows = []
    for index in range(2):
        image = tmp_path / f"{index}.jpg"
        image.write_bytes(str(index).encode())
        rows.append(
            {
                "id": str(index),
                "group_id": str(index),
                "image": image.name,
                "image_sha256": file_hash(image),
                "reference": "A",
                "split": "test",
                "synthetic": False,
                "license": "fixture",
                "subset": "TotalText",
                "ocr": {"text": "A", "rec_texts": ["A"], "confidences": [0.8], "latency_ms": 1},
                "vlm": {"text": "B", "latency_ms": 5},
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(collect_paddle_gpu, "MANIFEST_SHA256", file_hash(manifest))
    monkeypatch.setattr(collect_paddle_gpu, "require_idle_gpu", lambda resources: None)
    monkeypatch.setattr(collect_paddle_gpu.importlib.metadata, "version", lambda name: "fixture")
    calls = []

    class FakePaddle:
        fail = True

        def __init__(self, models):
            pass

        def infer(self, image):
            calls.append(image.name)
            if self.fail and image.name == "1.jpg":
                raise RuntimeError("interrupted")
            return {"text": "A", "rec_texts": ["A"], "confidences": [0.9], "latency_ms": 2}

    monkeypatch.setattr(collect_paddle_gpu, "PaddleGPU", FakePaddle)
    args = SimpleNamespace(
        registry=registry,
        manifest=manifest,
        qwen_observations=manifest,
        output=tmp_path / "run",
        per_subset=0,
        resume=False,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        collect_paddle_gpu.collect(args)
    FakePaddle.fail = False
    args.resume = True
    collect_paddle_gpu.collect(args)
    result = [
        json.loads(line) for line in (args.output / "observations.jsonl").read_text().splitlines()
    ]
    assert calls == ["0.jpg", "1.jpg", "1.jpg"]
    assert [row["vlm"]["text"] for row in result] == ["B", "B"]
    assert all(row["ocr"]["confidences"] == [0.9] for row in result)
