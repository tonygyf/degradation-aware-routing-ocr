import copy
import json
import sys

import pytest

from src.evaluation.baseline import lambda_sensitivity, replay_calibration, replay_development


def paired_cal():
    return [
        {
            "id": str(i),
            "group_id": str(i),
            "image_sha256": str(i) * 64,
            "license": "synthetic software fixture",
            "split": "val",
            "role": "Cal",
            "synthetic": True,
            "reference": "A" if i == 1 else "",
            "ocr": {
                "text": "bad" if i == 1 else "",
                "rec_texts": ["bad"] if i == 1 else [],
                "confidences": [0.84996] if i == 1 else [],
                "latency_ms": 2 * i,
            },
            "vlm": {"text": "A" if i == 1 else "No text", "latency_ms": 10 * i},
        }
        for i in (1, 2)
    ]


def test_cal_candidates_use_rounded_mean_strict_boundary_and_serial_latency():
    rows = paired_cal()
    result = replay_calibration(rows)
    assert result["selected_threshold"] is None
    assert result["historical_threshold"] == 0.90
    candidates = result["candidates"]
    assert [r["threshold"] for r in candidates] == [0.8, 0.85, 0.9, 0.95]
    assert [r["vlm_calls"] for r in candidates] == [1, 1, 2, 2]
    assert [r["latency_ms_mean"] for r in candidates] == [13, 13, 18, 18]
    assert [r["cer_macro_clipped"] for r in candidates] == [1, 1, 0.5, 0.5]
    assert [r["delta_cer_vs_qwen_percentage_points"] for r in candidates] == [50, 50, 0, 0]
    changed = copy.deepcopy(rows)
    for row in changed:
        row["reference"], row["vlm"]["text"] = "other gold", "other prediction"
    assert [r["vlm_calls"] for r in replay_calibration(changed)["candidates"]] == [1, 1, 2, 2]


def test_cal_replay_refuses_test_roles_empty_or_duplicate_data():
    with pytest.raises(ValueError, match="paired Cal"):
        replay_calibration([])
    rows = paired_cal()
    rows[0]["role"] = "Test"
    with pytest.raises(ValueError, match="paired Cal"):
        replay_calibration(rows)
    rows = paired_cal()
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="paired Cal"):
        replay_calibration(rows)
    with pytest.raises(ValueError, match="unique"):
        replay_calibration([rows[1], rows[1]])


def test_cli_refuses_synthetic_calibration_results(tmp_path, monkeypatch):
    from scripts.replay_calibration import main

    path, output = tmp_path / "fixture.jsonl", tmp_path / "metrics.json"
    path.write_text("".join(json.dumps(row) + "\n" for row in paired_cal()))
    monkeypatch.setattr(
        sys, "argv", ["replay_calibration", "--observations", str(path), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="real Cal"):
        main()
    assert not output.exists()


def test_dev_frontier_uses_separate_signals_and_always_qwen_endpoint():
    rows = paired_cal()
    for row in rows:
        row["role"] = "Dev"
        row["ocr"]["confidence_boxes"] = [{"confidence": 0.1}] if row["id"] == "1" else []
    rows[0]["ocr"]["confidences"] = [1.0]
    result = replay_development(rows)
    assert result["selected_threshold"] is None
    assert len(result["signals"]["image_mean"]) == 21
    assert result["signals"]["image_mean"][18]["vlm_calls"] == 1
    assert result["signals"]["box_min"][18]["vlm_calls"] == 2
    assert result["signals"]["image_mean"][-1]["vlm_calls"] == 2
    assert result["signals"]["image_mean"][-1]["latency_ms_mean"] == 18
    assert result["always_qwen_vlm_only_latency_ms_mean"] == 15


def test_lambda_sensitivity_is_cal_only_and_never_modifies_historical_policy():
    rows = paired_cal()
    rows[1]["vlm"]["text"] = ""
    rows[0]["vlm"]["latency_ms"] = 100000
    result = lambda_sensitivity(rows)
    assert result["lambda_choices"][0]["lambda_optimal_threshold"] == 0.9
    assert result["lambda_choices"][-1]["lambda_optimal_threshold"] == 0
    assert result["historical_threshold"] == 0.9
    assert result["historical_policy_modified"] is False
    rows[0]["role"] = "Test"
    with pytest.raises(ValueError, match="paired Cal"):
        lambda_sensitivity(rows)
