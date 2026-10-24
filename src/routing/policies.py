from __future__ import annotations

from typing import Protocol

from src.detection.types import Region
from src.evaluation.metrics import cer
from src.routing.features import FEATURE_NAMES, features


class FixedThresholdRouter:
    def __init__(self, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        self.threshold = threshold

    def score(self, ocr: dict) -> float:
        return float(features(ocr)[0] < self.threshold)


class LearnedRouter:
    def __init__(self, seed: int, kind: str = "logreg"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if kind == "logreg":
            model = LogisticRegression(random_state=seed, max_iter=1000, solver="lbfgs")
        elif kind == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(8,), random_state=seed, max_iter=2000, solver="lbfgs"
            )
        else:
            raise ValueError("Choose logreg or mlp")
        self.pipeline = make_pipeline(StandardScaler(), model)

    def fit(self, rows: list[dict]) -> None:
        if not rows or any(row["split"] != "train" for row in rows):
            raise ValueError("Learned router can fit only the training split")
        labels = [
            int(cer(r["reference"], r["vlm"]["text"]) < cer(r["reference"], r["ocr"]["text"]))
            for r in rows
        ]
        if len(set(labels)) < 2:
            raise ValueError("Training requires both benefit and no-benefit labels")
        self.pipeline.fit([features(r["ocr"]) for r in rows], labels)

    def score(self, ocr: dict) -> float:
        return float(self.pipeline.predict_proba([features(ocr)])[0, 1])

    def state(self) -> dict:
        scaler, model = self.pipeline.steps[0][1], self.pipeline.steps[1][1]
        state = {
            "features": FEATURE_NAMES,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "classes": model.classes_.tolist(),
        }
        if hasattr(model, "coef_"):
            state.update(coef=model.coef_.tolist(), intercept=model.intercept_.tolist())
        else:
            state.update(
                coefs=[x.tolist() for x in model.coefs_],
                intercepts=[x.tolist() for x in model.intercepts_],
            )
        return state


class DegradationRouter(Protocol):
    """C: add image-only brightness/noise/blur features, train-only calibration."""

    def score(self, ocr: dict, degradation: dict[str, float]) -> float: ...


class RegionBudgetRouter(Protocol):
    """D: select region IDs under a declared cost budget; implementation deferred."""

    def select(self, regions: list[Region], budget_units: float) -> list[str]: ...
