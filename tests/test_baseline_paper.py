import pytest

from src.evaluation.baseline import evaluate


def test_internal_baseline_uses_clipped_macro_cer_and_frozen_policy():
    rows = [
        {
            "reference": "A",
            "ocr": {
                "text": "AAAA",
                "rec_texts": ["AAAA"],
                "confidences": [0.89996],
                "latency_ms": 2,
            },
            "vlm": {"text": "A", "latency_ms": 10},
        },
        {
            "reference": "",
            "ocr": {"text": "", "rec_texts": [], "confidences": [], "latency_ms": 4},
            "vlm": {"text": "No text", "latency_ms": 20},
        },
    ]
    result = evaluate(rows)
    assert result["systems"]["paddleocr"]["cer_macro_clipped"] == 0.5
    assert result["systems"]["qwen"]["cer_macro_clipped"] == 0.5
    assert result["systems"]["confroute"]["cer_macro_clipped"] == 1
    assert result["systems"]["confroute"]["vlm_call_rate"] == 0.5
    assert result["systems"]["confroute"]["latency_ms_mean"] == 13
    assert result["qwen_minus_paddle_cer"]["sign_flip"]["estimate"] == 0
    with pytest.raises(ValueError, match="paired"):
        evaluate([])
