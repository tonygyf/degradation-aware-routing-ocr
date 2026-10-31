import pytest

from src.evaluation.shc_routing import replay, sensitivity


def sample(key, kind, visible, full, ocr_text, vlm_text, boxes):
    return {
        "id": key,
        "failure_type": kind,
        "reference_visible": visible,
        "reference_full": full,
        "critical_span": None,
        "ocr": {"text": ocr_text, "latency_ms": 2, "confidence_boxes": boxes},
        "vlm": {"text": vlm_text, "latency_ms": 10},
    }


def test_no_text_filter_uses_only_ocr_not_reference_or_failure_label():
    rows = [
        sample("blank", "S6", "", "", " ", "No text", []),
        sample("missed", "L1", "A", "A", "", "A", []),
        sample("retained", "S1", "B", "B", "B", "wrong", [{"confidence": 0.9}]),
    ]
    systems = replay(rows)["systems"]
    assert systems["confroute"]["vlm_call_rate"] == pytest.approx(2 / 3)
    assert systems["confroute"]["pooled"]["operational_flag_rate"] == pytest.approx(1 / 3)
    filtered = systems["no_text_confroute"]
    assert filtered["vlm_call_rate"] == 0
    assert filtered["pooled"]["empty_rate"] == pytest.approx(2 / 3)
    assert filtered["pooled"]["cer_full_macro_clipped"] == pytest.approx(1 / 3)
    assert filtered["pooled"]["operational_flag_rate"] == 0
    assert filtered["estimated_latency_ms_mean"] == 2
    assert systems["qwen"]["estimated_latency_ms_mean"] == 10
    assert systems["confroute"]["estimated_latency_ms_mean"] == pytest.approx(2 + 20 / 3)


def test_shc_replay_rejects_duplicate_ids_and_missing_box_cache():
    row = sample("a", "S6", "", "", "", "", [])
    with pytest.raises(ValueError, match="unique"):
        replay([row, row])
    del row["ocr"]["confidence_boxes"]
    with pytest.raises(KeyError, match="confidence_boxes"):
        replay([row])


def test_sensitivity_has_true_endpoints_and_separate_internvl_fallback():
    rows = [
        sample("a", "L1", "A", "A", "B", "A", [{"confidence": 0.90}]),
        sample("b", "S6", "", "", " ", "No text", []),
    ]
    for row in rows:
        row["internvl"] = {"text": "", "latency_ms": 30}
    r = sensitivity(rows, include_internvl=True)
    assert r["selected_threshold"] is None
    assert r["thresholds"]["0.85"]["systems"]["confroute"]["vlm_call_rate"] == 0.5
    assert r["thresholds"]["0.9"]["systems"]["confroute"]["vlm_call_rate"] == 0.5
    assert r["thresholds"]["0.95"]["systems"]["confroute"]["vlm_call_rate"] == 1
    assert r["endpoints"]["always_qwen"]["vlm_call_rate"] == 1
    assert r["internvl_fallback"]["fallback"] == "internvl"
    assert "qwen" not in r["internvl_fallback"]["systems"]
    assert r["internvl_fallback"]["systems"]["confroute"]["vlm_call_rate"] == 0.5
    assert r["internvl_fallback"]["systems"]["confroute"]["estimated_latency_ms_mean"] == 17
    assert r["no_text_audit"]["precision"] == r["no_text_audit"]["recall"] == 1
    with pytest.raises(ValueError, match="fallback"):
        replay(rows, fallback="unknown")
    with pytest.raises(ValueError, match="Threshold"):
        replay(rows, threshold=float("nan"))
