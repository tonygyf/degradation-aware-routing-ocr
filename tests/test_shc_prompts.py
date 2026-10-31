import copy

import pytest

from src.evaluation.shc_prompts import compare_prompts


def observations(role="Test"):
    return [
        {
            "id": str(i),
            "model": "qwen",
            "prompt": prompt,
            "role": role,
            "failure_type": "L1",
            "reference_visible": "A",
            "reference_full": "A",
            "critical_span": None,
            "prediction": "A" if prompt == "P2" else "B",
            "latency_ms": 2,
        }
        for prompt in ("P0", "P2")
        for i in range(2)
    ]


def test_only_prompt_dev_can_report_best_prompt_and_main_stays_p0():
    test = compare_prompts(observations())
    assert test["selected_dev_prompts"] is None
    assert test["exploratory_nonconfirmatory"] is True
    dev = compare_prompts(observations("Prompt_dev"))
    assert dev["selected_dev_prompts"] == {"qwen": "P2"}
    assert dev["main_comparison_prompt"] == "P0"
    assert dev["models"]["qwen"]["P2"]["failure_types"]["L1"]["cer_full_macro_clipped"] == 0


def test_prompt_pairing_checks_ids_annotations_and_duplicates():
    rows = observations()
    with pytest.raises(ValueError, match="identical sample IDs"):
        compare_prompts(rows[:-1])
    bad = copy.deepcopy(rows)
    bad[-1]["reference_full"] = "different"
    with pytest.raises(ValueError, match="identities differ"):
        compare_prompts(bad)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_prompts(rows + [rows[0]])
    bad = copy.deepcopy(rows)
    bad[0]["prompt"] = "P4"
    with pytest.raises(ValueError, match="fixed sweep"):
        compare_prompts(bad)
