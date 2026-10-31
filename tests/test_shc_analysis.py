import pytest

from src.evaluation.shc_analysis import (
    binary_ranking,
    confidence_ranking,
    risk_coverage,
    routing_labels,
)


def rows():
    return [
        {
            "id": str(i),
            "failure_type": "L1",
            "reference_visible": "A",
            "reference_full": "A",
            "critical_span": None,
            "ocr": {
                "text": "A" if i < 2 else "B",
                "confidence_boxes": [{"confidence": score}],
                "latency_ms": 2,
            },
            "vlm": {"text": "A", "latency_ms": 10},
        }
        for i, score in enumerate((0.99, 0.8, 0.2, 0.0))
    ]


def test_risk_coverage_is_full_set_replacement_not_selected_subset_error():
    r = risk_coverage(rows(), random_resamples=2000)
    points = {p["requested_paddle_coverage"]: p for p in r["points"]}
    assert points[0]["cer_full_confidence"] == 0
    assert points[1]["cer_full_confidence"] == 0.5
    assert points[0.5]["cer_full_confidence"] == 0
    assert points[0.5]["cer_full_random_exact_expectation"] == 0.25
    assert points[0.5]["cer_full_random_mean"] == pytest.approx(0.25, abs=0.03)
    assert points[0]["cer_full_random_std"] == points[1]["cer_full_random_std"] == 0
    assert r == risk_coverage(rows(), random_resamples=2000)
    with pytest.raises(ValueError, match="two"):
        risk_coverage(rows(), random_resamples=1)


def test_confidence_rankings_keep_route_benefit_and_flag_targets_distinct():
    r = confidence_ranking(rows())
    assert r["targets"]["route_benefit"]["auroc"] == 1
    assert r["targets"]["operational_flag"]["auroc"] is None
    assert routing_labels(rows()) == [0, 0, 1, 1]
    assert binary_ranking([0, 1], [0.5, 0.5])["auroc"] == 0.5
    assert binary_ranking([0, 1], [0.5, 0.5])["auprc_average_precision"] == 0.5
    with pytest.raises(ValueError, match="finite"):
        binary_ranking([0, 1], [0, float("nan")])
    with pytest.raises(ValueError, match="target"):
        routing_labels(rows(), "not_a_target")


def test_coverage_confidence_ties_use_id_not_reference_text():
    observations = rows()[:2]
    for r in observations:
        r["ocr"]["confidence_boxes"] = [{"confidence": 0.5}]
    observations[1]["ocr"]["text"] = "B"
    a = risk_coverage(observations, random_resamples=2)["points"]
    b = risk_coverage(list(reversed(observations)), random_resamples=2)["points"]
    assert [p["cer_full_confidence"] for p in a] == [p["cer_full_confidence"] for p in b]
    assert a[6]["cer_full_confidence"] == 0
