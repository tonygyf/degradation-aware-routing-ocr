import copy
import json
from pathlib import Path

import pytest

from src.routing.shc_learned import PaperSHCRouter


def train_rows():
    return [
        {
            "id": f"train-{i}",
            "role": "Tau_dev",
            "split": "train",
            "failure_type": "L1",
            "reference_visible": "A",
            "reference_full": "A",
            "critical_span": None,
            "ocr": {
                "text": "B" if i % 2 == 0 else "A",
                "latency_ms": 2,
                "confidence_boxes": [{"confidence": 0.2 if i % 2 == 0 else 0.9}],
                "paper_features": {"text_density": 0.1 if i % 2 == 0 else 0.8},
                "paper_feature_provenance": "synthetic software fixture",
            },
            "vlm": {"text": "A" if i % 2 == 0 else "B", "latency_ms": 10},
        }
        for i in range(12)
    ]


def config():
    return json.loads(Path("configs/paper_shc_router.json").read_text())


def test_l2_router_fits_only_tau_train_and_exposes_unknown_author_settings():
    rows = train_rows()
    model = PaperSHCRouter(config()).fit(rows)
    meta = model.metadata()
    assert meta["author_optimizer_and_scaler_verified"] is False
    assert meta["fit_ids"] == [r["id"] for r in rows]
    assert meta["scaler_mean"] == pytest.approx([0.55, 0.55, 0.45])
    assert model.score(rows[0]["ocr"]) > model.score(rows[1]["ocr"])
    before = model.metadata()
    ocr = {**rows[0]["ocr"], "reference_full": "poison", "vlm": {"text": "poison"}}
    assert model.score(ocr) == model.score(rows[0]["ocr"])
    assert model.metadata() == before
    bad = copy.deepcopy(rows)
    bad[0]["role"] = "Test"
    with pytest.raises(ValueError, match="Tau_dev"):
        PaperSHCRouter(config()).fit(bad)
    bad = copy.deepcopy(rows)
    bad[0]["split"] = "test"
    with pytest.raises(ValueError, match="Tau_dev"):
        PaperSHCRouter(config()).fit(bad)


def test_unknown_density_is_required_and_no_target_class_is_fabricated():
    rows = train_rows()
    del rows[0]["ocr"]["paper_feature_provenance"]
    with pytest.raises(ValueError, match="provenance"):
        PaperSHCRouter(config()).fit(rows)
    rows = train_rows()
    for r in rows:
        r["ocr"]["text"] = r["vlm"]["text"]
    with pytest.raises(ValueError, match="Both target classes"):
        PaperSHCRouter(config()).fit(rows)
    with pytest.raises(ValueError, match="Fit before"):
        PaperSHCRouter(config()).score(train_rows()[0]["ocr"])


def test_zero_confidence_geometric_mean_and_flag_target():
    cfg = config()
    cfg.update(
        feature_set="geom",
        target="operational_flag",
        threshold_target="operational_flag",
        threshold=0.5,
        threshold_source="independent software fixture",
    )
    rows = train_rows()
    for i, r in enumerate(rows):
        if i % 2 == 0:
            r.update(failure_type="S6", reference_visible="", reference_full="")
    model = PaperSHCRouter(cfg).fit(rows)
    assert model.features({"text": "x", "confidence_boxes": [{"confidence": 0}]}) == [0]
    assert model.features({"text": "", "confidence_boxes": []}) == [0]
    assert model.train_positive_rate == 0.5


def test_threshold_from_another_learning_target_is_rejected():
    cfg = config()
    cfg["target"] = "operational_flag"
    with pytest.raises(ValueError, match="same learning target"):
        PaperSHCRouter(cfg)
