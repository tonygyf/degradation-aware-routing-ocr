import json
import sys

import pytest

from scripts import evaluate_shc
from src.evaluation.shc import critical_span_flag, evaluate, summarize
from src.routing.confroute import ConfRouteRouter


def sample(kind="S1", visible="A", full="A B", prediction="A B", span="B"):
    return {
        "id": "one",
        "failure_type": kind,
        "reference_visible": visible,
        "reference_full": full,
        "prediction": prediction,
        "critical_span": span,
    }


def test_shc_dual_reference_semantic_share_and_macro_cer():
    rows = [sample(), sample("L1"), sample("S6", "", "", "There is no text.", None)]
    result = evaluate(rows)
    metrics = result["pooled"]
    assert metrics["cer_visible_macro_clipped"] == 1
    assert metrics["cer_full_macro_clipped"] == pytest.approx(1 / 3)
    assert metrics["operational_flag_count"] == 3
    assert metrics["semantic_flag_count"] == 2
    assert metrics["semantic_share"] == pytest.approx(2 / 3)
    assert result["failure_types"]["L1"]["semantic_share"] == 0


def test_critical_span_case_punctuation_visibility_and_normalization():
    assert critical_span_flag(sample(prediction=" A\n B "))
    assert not critical_span_flag(sample(prediction="a b"))
    assert not critical_span_flag(sample(visible="A B"))
    assert not critical_span_flag(sample(span=None))
    assert not critical_span_flag(sample(prediction="A B", span="B!"))
    assert critical_span_flag(sample(prediction="A B!", span="B!"))
    with pytest.raises(ValueError, match="punctuation-only"):
        critical_span_flag(sample(span="!"))
    with pytest.raises(ValueError, match="unassigned"):
        critical_span_flag(sample(span=" "))
    with pytest.raises(ValueError, match="Unknown"):
        critical_span_flag(sample("unknown"))


def test_no_text_refusal_is_flagged_and_zero_denominator_is_defined():
    assert critical_span_flag(sample("S6", "", "", "No text", None))
    empty = sample("S6", "", "", " \n", None)
    assert not critical_span_flag(empty)
    metrics = summarize([empty])
    assert metrics["semantic_share"] == 0
    assert metrics["empty_rate"] == 1
    assert metrics["cer_full_macro_clipped"] == 0
    with pytest.raises(ValueError, match="both references"):
        critical_span_flag(sample("S6"))
    with pytest.raises(ValueError, match="empty set"):
        evaluate([])


def test_confroute_signals_rounding_strict_boundary_and_no_scalar_fallback():
    ocr = {
        "rec_texts": ["A", "B", ""],
        "confidences": [0.89996, 0.89996, 0.1],
        "confidence_boxes": [{"confidence": 0.89}, {"confidence": 0.95}],
        "confidence": 0.99,
    }
    mean_router = ConfRouteRouter("image_mean")
    box_router = ConfRouteRouter("box_min")
    assert mean_router.confidence(ocr) == 0.9
    assert mean_router.score(ocr) == 0
    assert box_router.score(ocr) == 1
    assert box_router.score({"confidence_boxes": [{"confidence": 0.9}]}) == 0
    assert box_router.score({"confidence_boxes": [], "confidence": 0.99}) == 1
    assert mean_router.score({"rec_texts": [], "confidences": []}) == 1
    with pytest.raises(KeyError, match="confidence_boxes"):
        box_router.score({"confidence": 0.99})
    with pytest.raises(ValueError, match="Invalid"):
        box_router.score({"confidence_boxes": [{"confidence": float("nan")}]})
    with pytest.raises(ValueError, match="Choose"):
        ConfRouteRouter("auto")
    with pytest.raises(ValueError, match="Threshold"):
        ConfRouteRouter("image_mean", float("nan"))


def test_shc_cli_writes_provenance_and_rejects_duplicate_ids(tmp_path, monkeypatch):
    observations, output = tmp_path / "rows.jsonl", tmp_path / "metrics.json"
    row = json.dumps(sample()) + "\n"
    observations.write_text(row)
    monkeypatch.setattr(sys, "argv", ["evaluate_shc", str(observations), "--output", str(output)])
    evaluate_shc.main()
    report = json.loads(output.read_text())
    assert report["author_dataset_identity_verified"] is False
    assert report["pooled"]["samples"] == 1
    assert len(report["observations_sha256"]) == 64
    observations.write_text(row * 2)
    with pytest.raises(ValueError, match="unique"):
        evaluate_shc.main()


def test_shc_cli_cluster_bootstrap_uses_supplied_source_identity(tmp_path, monkeypatch):
    observations, output = tmp_path / "rows.jsonl", tmp_path / "metrics.json"
    row = {**sample(), "family_id": "family-a", "source_image": "source-a"}
    observations.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_shc", str(observations), "--output", str(output), "--cluster-bootstrap"],
    )
    evaluate_shc.main()
    report = json.loads(output.read_text())
    assert report["cluster_bootstrap"]["source_image"]["ci95_low"] == 1
    row["source_image"] = ""
    observations.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="audited nonempty"):
        evaluate_shc.main()


def test_reference_overlap_normalizes_whitespace_but_preserves_case_and_punctuation():
    rows = [
        sample(visible="A\nB", full=" A B ", prediction="A B", span=None),
        sample(visible="A", full="A.", prediction="A", span=None),
        sample(visible="a", full="A", prediction="A", span=None),
    ]
    result = evaluate(rows)["reference_overlap"]
    assert result["identical"]["samples"] == 1
    assert result["identical"]["cer_full_macro_clipped"] == 0
    assert result["different"]["samples"] == 2
    assert result["different"]["cer_visible_macro_clipped"] == pytest.approx(0.5)
    assert evaluate(rows[:1])["reference_overlap"]["different"] is None
