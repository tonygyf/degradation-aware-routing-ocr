import copy
import json
import sys

import pytest

from src.evaluation.external_comparison import compare


def test_five_system_comparison_keeps_model_identity_and_rejects_misalignment():
    base = [
        {
            "id": str(i),
            "group_id": str(i),
            "image_sha256": str(i) * 64,
            "reference": "A",
            "split": "test",
            "synthetic": False,
            "subset": "TotalText",
            "ocr": {"text": "B", "rec_texts": ["B"], "confidences": [0.8], "latency_ms": 2},
            "vlm": {"text": "A", "latency_ms": 10},
        }
        for i in (1, 2)
    ]
    others = {
        name: [
            {
                **copy.deepcopy(row),
                "system": name,
                "vlm": {"text": "A B A B A B" if name == "internvl" else "A", "latency_ms": 4},
            }
            for row in reversed(base)
        ]
        for name in ("deepseek", "internvl")
    }
    result = compare(base, others)
    assert set(result["systems"]) == {"paddleocr", "qwen", "confroute", "internvl", "deepseek"}
    assert result["repetition"]["internvl"]["count"] == 2
    assert result["repetition"]["deepseek"]["count"] == 0
    assert result["systems"]["confroute"]["pooled"]["vlm_call_rate"] == 1
    assert result["complete_external_set"] is False
    others["internvl"][0]["reference"] = "other"
    with pytest.raises(ValueError, match="identity"):
        compare(base, others)
    others["internvl"][0]["reference"] = "A"
    others["internvl"].pop()
    with pytest.raises(ValueError, match="sample IDs"):
        compare(base, others)


def test_report_rejects_incomplete_data_even_if_identities_match(tmp_path, monkeypatch):
    from scripts import compare_external
    from src.datasets.manifest import sha256

    row = {
        "id": "a",
        "group_id": "a",
        "image_sha256": "a" * 64,
        "reference": "A",
        "split": "test",
        "synthetic": False,
        "subset": "TotalText",
        "license": "fixture",
        "ocr": {"text": "A", "rec_texts": ["A"], "confidences": [0.9], "latency_ms": 1},
        "vlm": {"text": "A", "latency_ms": 2},
    }
    paths = {
        name: tmp_path / f"{name}.jsonl" for name in ("manifest", "qwen", "internvl", "deepseek")
    }
    for name, path in paths.items():
        data = row if name in {"manifest", "qwen"} else {**row, "system": name}
        path.write_text(json.dumps(data) + "\n")
    monkeypatch.setattr(compare_external, "MANIFEST_SHA256", sha256(paths["manifest"]))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_external",
            "--manifest",
            str(paths["manifest"]),
            "--qwen",
            str(paths["qwen"]),
            "--internvl",
            str(paths["internvl"]),
            "--deepseek",
            str(paths["deepseek"]),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="incomplete"):
        compare_external.main()
    assert not output.exists()
    paths["manifest"].write_text("changed")
    with pytest.raises(ValueError, match="manifest changed"):
        compare_external.main()


def test_source_macro_latency_gives_each_source_equal_weight():
    rows = [
        {
            "id": str(i),
            "group_id": str(i),
            "image_sha256": str(i) * 64,
            "reference": "A",
            "split": "test",
            "synthetic": False,
            "subset": "large" if i < 3 else "small",
            "ocr": {"text": "A", "rec_texts": ["A"], "confidences": [0.8], "latency_ms": 2},
            "vlm": {"text": "A", "latency_ms": 1 if i < 3 else 100},
        }
        for i in (1, 2, 3)
    ]
    others = {name: [{**row, "system": name} for row in rows] for name in ("internvl", "deepseek")}
    systems = compare(rows, others)["systems"]
    for name in ("qwen", "internvl", "deepseek"):
        assert systems[name]["source_macro"]["latency_ms_mean"] == 50.5
        assert systems[name]["pooled"]["latency_ms_mean"] == 34
    assert systems["paddleocr"]["source_macro"]["latency_ms_mean"] == 2
    assert systems["confroute"]["source_macro"]["latency_ms_mean"] == 52.5
