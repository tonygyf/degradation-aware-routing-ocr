import copy
import json
from pathlib import Path

import pytest

from src.cache import preflight
from src.datasets.manifest import load_manifest
from src.evaluation.metrics import cer, edit_counts, summarize
from src.recognition.backends import QwenVLHTTPAdapter
from src.routing.features import features
from src.routing.policies import FixedThresholdRouter, LearnedRouter
from src.run import evaluate, run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rows():
    return load_manifest(ROOT / "data/fixtures/observations.jsonl")


def test_cer_unicode_blanks_and_insertions():
    assert cer("你好", "你号") == 0.5
    assert cer("é", "e\u0301") == 0
    assert edit_counts("a", "abc") == (2, 2, 1)
    assert cer("", "abc") == 3
    assert cer("", "") == 0
    assert cer("ABC", "abc") == 1


def test_fixed_threshold_boundary_and_empty_detection():
    router = FixedThresholdRouter(0.7)
    assert router.score({"text": "a", "confidences": [0.7]}) == 0
    assert router.score({"text": "a", "confidences": [0.69]}) == 1
    assert router.score({"text": "", "confidences": []}) == 1
    with pytest.raises(ValueError):
        FixedThresholdRouter(float("nan"))


def test_group_and_content_leakage_rejected(rows, tmp_path):
    for field in ("group_id", "image_sha256"):
        bad = copy.deepcopy(rows)
        bad[-1][field] = bad[0][field]
        path = tmp_path / "bad.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in bad), encoding="utf-8")
        with pytest.raises(ValueError, match="leakage"):
            load_manifest(path)


def test_duplicate_and_nonfinite_rejected(rows, tmp_path):
    for mutation in (
        lambda r: r.append(r[0]),
        lambda r: r[0]["ocr"].update(confidences=[float("nan")]),
    ):
        bad = copy.deepcopy(rows)
        mutation(bad)
        path = tmp_path / "bad.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in bad), encoding="utf-8")
        with pytest.raises(ValueError):
            load_manifest(path)


@pytest.mark.parametrize("kind", ["logreg", "mlp"])
def test_learned_no_test_or_reference_features(rows, kind):
    train = [r for r in rows if r["split"] == "train"]
    model = LearnedRouter(42, kind)
    with pytest.raises(ValueError, match="training split"):
        model.fit(rows)
    model.fit(train)
    test = [r for r in rows if r["split"] == "test"]
    before = [model.score(r["ocr"]) for r in test]
    altered = copy.deepcopy(test)
    for r in altered:
        r["reference"] = "LEAK"
        r["vlm"] = {"text": "LEAK"}
    assert before == [model.score(r["ocr"]) for r in altered]
    second = LearnedRouter(42, kind)
    second.fit(train)
    assert before == [second.score(r["ocr"]) for r in test]
    assert features(test[0]["ocr"]) == features(altered[0]["ocr"])


def test_budget_and_lazy_vlm(rows):
    class Spy:
        def __init__(self):
            self.calls = 0

        def infer(self, row):
            self.calls += 1
            return row["vlm"]

    backend = Spy()
    test = [r for r in rows if r["split"] == "test"]
    records = evaluate(test, FixedThresholdRouter(1), 0.5, 9, 4.0, 2.0, backend)
    assert backend.calls == 2
    assert sum(r["vlm_called"] for r in records) == 2
    metrics = summarize(records, 2)
    assert metrics["declared_cost_units"] == 4
    assert metrics["latency_ms_mean"] == 27
    backend = Spy()
    records = evaluate(test, FixedThresholdRouter(1), 0.5, 0, 100, 1, backend)
    assert backend.calls == 0
    assert all(r["prediction"] == r["ocr_text"] for r in records)


def test_invalid_budget(rows):
    with pytest.raises(ValueError):
        evaluate(rows, FixedThresholdRouter(0.7), 0.5, 2, float("nan"), 1)


def test_hallucination_proxies():
    records = [
        {"reference": "", "prediction": "假", "ocr_text": "", "vlm_called": True, "latency_ms": 10}
    ]
    metrics = summarize(records, 1)
    assert metrics["insertion_proxy"] == 1
    assert metrics["blank_nonempty_rate"] == 1
    assert metrics["vlm_regression_rate"] == 1


def test_real_inference_preflight_and_endpoint():
    with pytest.raises(ValueError, match="authorization"):
        preflight(json.loads((ROOT / "configs/models.example.json").read_text()))
    for url in ("https://example.com/v1", "file:///etc/passwd", "http://localhost.evil/v1"):
        with pytest.raises(ValueError):
            QwenVLHTTPAdapter(url, "model")


@pytest.mark.parametrize("name", ["baseline_a", "baseline_b", "baseline_b_mlp"])
def test_run_outputs_and_reproducibility(name, tmp_path):
    first = run(ROOT / f"configs/{name}.json", tmp_path)
    second = run(ROOT / f"configs/{name}.json", tmp_path)
    assert first != second
    one = json.loads((first / "metrics.json").read_text())
    two = json.loads((second / "metrics.json").read_text())
    for key in ("cer_micro", "vlm_calls", "declared_cost_units", "insertion_proxy"):
        assert one[key] == two[key]
    assert one["samples"] == 16
    if name == "baseline_a":
        assert one["vlm_calls"] == 12
        assert one["edit_count"] == 26
        assert one["reference_characters"] == 210
    assert json.loads((first / "metadata.json").read_text())["manifest_sha256"]
    assert (first / "run.log").is_file()
