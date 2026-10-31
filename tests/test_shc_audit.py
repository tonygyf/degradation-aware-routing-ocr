import pytest

from src.evaluation.shc_audit import reweighted_flag_audit


def test_published_qwen_stratified_counts_reproduce_reweighted_rates():
    r = reweighted_flag_audit(
        559,
        3338,
        {"true_positive": 89, "false_positive": 11},
        {"false_negative": 1, "true_negative": 99},
    )
    assert r["population"] == 3897
    assert r["flag_precision"] == 0.89
    assert r["estimated_true_positive_count"] == pytest.approx(497.51)
    assert r["estimated_false_negative_count"] == pytest.approx(33.38)
    assert r["population_reweighted_recall"] * 100 == pytest.approx(93.7, abs=0.05)
    assert r["precision_corrected_flag_rate"] * 100 == pytest.approx(12.77, abs=0.005)
    assert r["audit_adjusted_true_rate"] * 100 == pytest.approx(13.62, abs=0.005)
    assert r["population_reweighted_recall"] != pytest.approx(89 / 90)


def test_positive_only_audit_never_infers_recall_and_counts_are_validated():
    r = reweighted_flag_audit(500, 3500, {"true_positive": 99, "false_positive": 1})
    assert r["flag_precision"] == 0.99
    assert r["population_reweighted_recall"] is None
    assert r["audit_adjusted_true_rate"] is None
    with pytest.raises(ValueError, match="integers"):
        reweighted_flag_audit(500.5, 3500, {"true_positive": 99, "false_positive": 1})
    with pytest.raises(ValueError, match="auto-positive"):
        reweighted_flag_audit(50, 3500, {"true_positive": 99, "false_positive": 1})
    with pytest.raises(ValueError, match="auto-negative"):
        reweighted_flag_audit(
            500,
            2,
            {"true_positive": 99, "false_positive": 1},
            {"false_negative": 1, "true_negative": 99},
        )
