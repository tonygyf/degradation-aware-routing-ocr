import json
from types import SimpleNamespace

import pytest

from scripts import collect_additional
from scripts.prepare_ccocr import file_hash
from src.evaluation.external import evaluate


def test_external_model_identity_cannot_be_silently_relabeled():
    row = {
        "system": "internvl",
        "reference": "A",
        "subset": "TotalText",
        "vlm": {"text": "A", "latency_ms": 5},
    }
    report = evaluate([row], "internvl")
    assert report["system"] == "internvl"
    assert not report["complete_external_set"]
    assert report["pooled"]["cer_macro_clipped"] == 0
    with pytest.raises(ValueError, match="identity"):
        evaluate([row], "deepseek")


def test_busy_gpu_is_rejected_before_torch_or_model_load(monkeypatch):
    def query(command, text):
        return "1234" if "--query-compute-apps=pid" in command else "80000"

    monkeypatch.setattr(collect_additional.subprocess, "check_output", query)
    with pytest.raises(ValueError, match="busy"):
        collect_additional.require_idle_gpu(
            {
                "gpu_authorized": True,
                "authorization": {"host_gpu_uuid": "GPU-test"},
                "min_free_vram_mib": 40000,
            }
        )


def test_custom_code_hashes_reject_changes_and_extra_modules(tmp_path):
    code = tmp_path / "model.py"
    code.write_text("# pinned code")
    spec = {
        "directory": str(tmp_path),
        "custom_code_files": [{"rfilename": "model.py", "local_sha256": file_hash(code)}],
    }
    collect_additional.verify_custom_code(spec)
    (tmp_path / "unexpected.py").write_text("# unknown")
    with pytest.raises(ValueError, match="Unexpected"):
        collect_additional.verify_custom_code(spec)
    code.write_text("# changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        collect_additional.verify_custom_code(spec)


def test_collection_resumes_without_repeating_completed_images(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    model_root.mkdir()
    code = model_root / "model.py"
    code.write_text("# fixture only")
    artifact = {"rfilename": "model.py", "local_sha256": file_hash(code)}
    (model_root / "download-provenance.json").write_text(
        json.dumps({"status": "verified", "revision": "fixed", "files": [artifact]})
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "resources": {},
                "models": [
                    {
                        "role": "internvl",
                        "directory": str(model_root),
                        "revision": "fixed",
                        "custom_code_files": [artifact],
                    }
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
                "license": "software-fixture",
                "subset": "TotalText",
                "ocr": {"text": "A", "rec_texts": ["A"], "confidences": [0.99], "latency_ms": 1},
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(collect_additional, "MANIFEST_SHA256", file_hash(manifest))
    monkeypatch.setattr(collect_additional, "require_idle_gpu", lambda resources: None)
    monkeypatch.setattr(collect_additional.importlib.metadata, "version", lambda name: "fixture")
    calls = []

    class Backend:
        fail = True

        def __init__(self, spec):
            pass

        def infer(self, image):
            calls.append(image.name)
            if image.name == "1.jpg" and self.fail:
                raise RuntimeError("interrupted")
            return {"text": "A", "latency_ms": 2}

    monkeypatch.setattr(collect_additional, "ExternalModel", Backend)
    args = SimpleNamespace(
        registry=registry,
        model="internvl",
        manifest=manifest,
        ocr_observations=manifest,
        per_subset=0,
        output=tmp_path / "run",
        resume=False,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        collect_additional.collect(args)
    assert len((args.output / "partial.jsonl").read_text().splitlines()) == 1
    Backend.fail = False
    args.resume = True
    collect_additional.collect(args)
    assert calls == ["0.jpg", "1.jpg", "1.jpg"]
    assert len((args.output / "observations.jsonl").read_text().splitlines()) == 2
    report = json.loads((args.output / "metrics.json").read_text())
    assert report["system"] == "internvl"
    assert report["pooled"]["samples"] == 2
