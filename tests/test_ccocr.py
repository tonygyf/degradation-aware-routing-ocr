import base64
import csv
import hashlib
import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from scripts.prepare_ccocr import download, extract_tsv, file_hash
from src.evaluation.ccocr import clipped_cer, eval_trans, evaluate, image_confidence


def test_extract_ccocr_preserves_frozen_test_and_source_text(tmp_path):
    source = tmp_path / "sample.tsv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_name", "image", "answer"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerow(
            {"image_name": "a.jpg", "image": base64.b64encode(b"image").decode(), "answer": "A\nB"}
        )
    rows = extract_tsv(source, "TotalText", tmp_path, "research")
    assert rows[0]["reference"] == "A\nB"
    assert rows[0]["split"] == "test"
    assert rows[0]["image_sha256"] == file_hash(tmp_path / rows[0]["image"])
    source.write_text("image_name\timage\tanswer\n../escape\taW1hZ2U=\ttext\n")
    with pytest.raises(ValueError, match="filename"):
        extract_tsv(source, "TotalText", tmp_path, "research")


def test_download_rejects_bad_existing_hash_without_network(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Existing file hash mismatch"):
        download("https://example.invalid/never-requested", path, "0" * 64)


def test_download_resumes_verified_ranges_and_rejects_wrong_offset(tmp_path, monkeypatch):
    payload = b"reviewed model bytes"
    path = tmp_path / "weights"
    path.with_suffix(".partial").write_bytes(payload[:5])
    requests = []

    def respond(request, timeout):
        requests.append(request.get_header("Range"))
        response = io.BytesIO(payload[5:])
        response.status = 206
        response.headers = {"Content-Range": f"bytes 5-{len(payload) - 1}/{len(payload)}"}
        return response

    monkeypatch.setattr("urllib.request.urlopen", respond)
    download(
        "https://example.invalid/model", path, hashlib.sha256(payload).hexdigest(), len(payload)
    )
    assert path.read_bytes() == payload
    assert requests == [f"bytes=5-{len(payload) - 1}"]
    path.unlink()
    with pytest.raises(ValueError, match="invalid artifact byte range"):
        download("https://example.invalid/model", path, "0" * 64, len(payload))


def test_collector_rejects_gpu_before_loading_models(tmp_path):
    from scripts.collect_ccocr import collect

    registry = tmp_path / "registry.json"
    registry.write_text('{"resources": {"gpu_authorized": false}}')
    with pytest.raises(ValueError, match="CPU-only guardrail"):
        collect(SimpleNamespace(registry=registry, device="cuda:0", workers=1))


def test_model_snapshot_checks_identity_and_artifact_bytes(tmp_path, monkeypatch):
    from scripts.download_ccocr_models import snapshot

    content = b'{"model": "reviewed"}'
    weights = b"reviewed weights"
    metadata = {
        "sha": "fixed-revision",
        "cardData": {"license": "apache-2.0"},
        "siblings": [
            {
                "rfilename": "config.json",
                "size": len(content),
                "blobId": hashlib.sha1(
                    b"blob " + str(len(content)).encode() + b"\0" + content
                ).hexdigest(),
            },
            {
                "rfilename": "weights.safetensors",
                "size": len(weights),
                "lfs": {"sha256": hashlib.sha256(weights).hexdigest()},
            },
        ],
    }
    transports = []
    failed_once = False

    def respond(url, timeout):
        nonlocal failed_once
        if not isinstance(url, str):
            transports.append(url.full_url)
            response = io.BytesIO(weights)
            response.status = 206
            response.headers = {"Content-Range": f"bytes 0-{len(weights) - 1}/{len(weights)}"}
            return response
        if "/resolve/" in url and not failed_once:
            failed_once = True
            raise urllib.error.URLError("temporary transport failure")
        return io.BytesIO(json.dumps(metadata).encode() if "/api/" in url else content)

    monkeypatch.setattr("urllib.request.urlopen", respond)
    monkeypatch.setattr("scripts.download_ccocr_models.time.sleep", lambda seconds: None)
    spec = {
        "repository": "owner/model",
        "revision": "fixed-revision",
        "license": "apache-2.0",
        "directory": str(tmp_path),
        "weight_mirror": "modelscope",
    }
    snapshot(spec)
    assert json.loads((tmp_path / "download-provenance.json").read_text())["status"] == "verified"
    assert transports == [
        "https://www.modelscope.cn/models/owner/model/resolve/master/weights.safetensors"
    ]
    assert (tmp_path / "weights.safetensors").read_bytes() == weights
    (tmp_path / "config.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Git blob hash mismatch"):
        snapshot(spec)
    metadata["sha"] = "different-revision"
    with pytest.raises(ValueError, match="revision or license changed"):
        snapshot(spec)


def test_collector_resumes_empty_partial_and_checks_local_artifacts(tmp_path, monkeypatch):
    from scripts import collect_ccocr

    image = tmp_path / "image.jpg"
    image.write_bytes(b"local image bytes")
    artifact = tmp_path / "weights"
    artifact.write_bytes(b"reviewed weights")
    (tmp_path / "download-provenance.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "revision": "pinned",
                "files": [{"rfilename": "weights", "local_sha256": file_hash(artifact)}],
            }
        )
    )
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "models": [
                    {"role": role, "revision": "pinned", "directory": str(tmp_path)}
                    for role in ("detector", "recognizer")
                ]
            }
        )
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
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
        )
        + "\n"
    )
    args = SimpleNamespace(
        registry=registry,
        manifest=manifest,
        device="cpu",
        workers=1,
        mode="ocr",
        per_subset=0,
        output=tmp_path / "run",
        resume=False,
    )
    monkeypatch.setattr(collect_ccocr.importlib.metadata, "version", lambda name: "test")

    class FakePaddle:
        fail = True

        def __init__(self, models):
            pass

        def infer(self, image):
            if self.fail:
                raise RuntimeError("interrupted before first prediction")
            return {"text": "A", "rec_texts": ["A"], "confidences": [0.95], "latency_ms": 1}

    monkeypatch.setattr(collect_ccocr, "Paddle", FakePaddle)
    with pytest.raises(RuntimeError, match="interrupted"):
        collect_ccocr.collect(args)
    assert (args.output / "partial.jsonl").stat().st_size == 0
    FakePaddle.fail = False
    args.resume = True
    collect_ccocr.collect(args)
    assert json.loads((args.output / "observations.jsonl").read_text())["ocr"]["text"] == "A"
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Model artifact changed"):
        collect_ccocr.collect(args)
    artifact.write_bytes(b"reviewed weights")
    spec = json.loads(registry.read_text())
    spec["models"].append({"role": "vlm", "revision": "pinned", "directory": str(tmp_path)})
    registry.write_text(json.dumps(spec))
    args.ocr_observations = args.output / "observations.jsonl"
    args.output, args.resume, args.mode = tmp_path / "combined", False, "both"
    FakePaddle.fail = True

    class FakeQwen:
        def __init__(self, model, registry, device):
            pass

        def infer(self, image):
            return {"text": "Qwen result", "latency_ms": 2}

    monkeypatch.setattr(collect_ccocr, "Qwen", FakeQwen)
    collect_ccocr.collect(args)
    combined = json.loads((args.output / "observations.jsonl").read_text())
    assert combined["ocr"]["text"] == "A"
    assert combined["vlm"]["text"] == "Qwen result"


def test_paper_metrics_use_clipped_whitespace_normalized_macro_cer():
    assert clipped_cer(" A\n B ", "A B") == 0
    assert clipped_cer("a", "aaaa") == 1
    assert clipped_cer("", "text") == 1
    assert clipped_cer("", " \n") == 0
    assert clipped_cer("é", "e\u0301") == 1
    assert eval_trans("Hello, world! hello", "WORLD hello") == pytest.approx(0.8)
    assert eval_trans("### A ***", "a") == pytest.approx(1)


def test_confroute_rounding_empty_boxes_boundary_and_replay_latency():
    row = {
        "id": "a",
        "subset": "TotalText",
        "reference": "A",
        "ocr": {
            "text": "B",
            "rec_texts": ["A", ""],
            "confidences": [0.89996, 0.1],
            "latency_ms": 2,
        },
        "vlm": {"text": "A", "latency_ms": 10},
    }
    assert image_confidence(row["ocr"]) == 0.9
    report, records = evaluate([row])
    assert report["confroute"]["pooled"]["vlm_call_rate"] == 0
    assert report["confroute"]["pooled"]["cer_macro_clipped"] == 1
    assert report["qwen"]["pooled"]["latency_ms_mean"] == 10
    assert len(records) == 3
    row["ocr"].update(rec_texts=[], confidences=[])
    assert image_confidence(row["ocr"]) is None
    report, _ = evaluate([row])
    assert report["confroute"]["pooled"]["latency_ms_mean"] == 12
    assert report["confroute"]["pooled"]["vlm_call_rate"] == 1
    row["ocr"].update(rec_texts=["mismatch"])
    with pytest.raises(ValueError, match="matching lengths"):
        image_confidence(row["ocr"])
