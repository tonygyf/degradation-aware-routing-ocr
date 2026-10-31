import copy
from pathlib import Path

import numpy as np
import pytest

from src.datasets.manifest import load_manifest
from src.routing.risk import RiskAwareRouter, degradation_features


def test_image_statistics_and_invalid_pixels():
    dark = degradation_features(np.zeros((4, 4, 3), dtype=np.uint8))
    bright = degradation_features(np.full((4, 4, 3), 255, dtype=np.uint8))
    assert dark == {"brightness_mean": 0, "contrast_std": 0, "gradient_rms": 0}
    assert bright["brightness_mean"] == pytest.approx(1)
    assert bright["contrast_std"] == pytest.approx(0)
    for pixels in (
        np.zeros((1, 4), dtype=np.uint8),
        np.zeros((4, 4)),
        np.zeros((4, 4, 4), dtype=np.uint8),
    ):
        with pytest.raises(ValueError):
            degradation_features(pixels)


def test_risk_fit_is_train_only_and_prediction_has_no_gt_or_vlm_input():
    rows = load_manifest(Path("data/fixtures/observations.jsonl"))
    for index, row in enumerate(rows):
        row["degradation"] = degradation_features(np.full((4, 4), index * 3, dtype=np.uint8))
    train = [row for row in rows if row["split"] == "train"]
    router = RiskAwareRouter()
    with pytest.raises(ValueError, match="training split"):
        router.fit(rows)
    unreviewed = copy.deepcopy(train)
    unreviewed[0]["reference_status"] = "annotation_order_unreviewed"
    with pytest.raises(ValueError, match="reference order"):
        router.fit(unreviewed)
    real_unreviewed = copy.deepcopy(train)
    for row in real_unreviewed:
        row["synthetic"] = False
    with pytest.raises(ValueError, match="reference order"):
        router.fit(real_unreviewed)
    router.fit(train)
    heldout = rows[-1]
    prediction = router.predict(heldout["ocr"], heldout["degradation"])
    heldout.update(reference="changed", vlm={"text": "changed"})
    assert prediction == router.predict(heldout["ocr"], heldout["degradation"])
    assert -1 <= prediction["expected_cer_gain"] <= 1
    assert 0 <= prediction["regression_probability"] <= 1


def test_budget_selection_rejects_negative_utility_and_is_deterministic():
    router = RiskAwareRouter()
    router.predict = lambda ocr, degradation: ocr

    def row(name, gain, probability):
        return {
            "id": name,
            "degradation": {},
            "ocr": {"expected_cer_gain": gain, "regression_probability": probability},
        }

    rows = [row("b", 0.8, 0.1), row("a", 0.8, 0.1), row("c", 0.2, 0.9)]
    assert router.select(rows, 1, 1) == ["a"]
    assert router.select(rows, 10, 1) == ["a", "b"]
    assert router.select(rows, 0, 1) == []
    with pytest.raises(ValueError):
        router.select(rows, 1, float("nan"))
