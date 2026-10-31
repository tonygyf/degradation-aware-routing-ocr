"""CPU prototype for expected CER benefit and post-upgrade regression risk.

This is a testable candidate, not a validated new algorithm. Select the risk
weight and budget on independent validation data before external evaluation.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.ccocr import clipped_cer
from src.routing.features import FEATURE_NAMES, features

DEGRADATION_NAMES = ("brightness_mean", "contrast_std", "gradient_rms")


def degradation_features(pixels: np.ndarray) -> dict[str, float]:
    """Summarize uint8 grayscale/RGB pixels; these are image statistics, not severity labels."""
    if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3) or min(pixels.shape[:2]) < 2:
        raise ValueError("Expected a nonempty uint8 grayscale or RGB image of at least 2x2")
    if pixels.ndim == 3 and pixels.shape[2] != 3:
        raise ValueError("Expected three RGB channels")
    gray = pixels.astype(float) / 255
    if pixels.ndim == 3:
        gray = gray @ np.array([0.2126, 0.7152, 0.0722])
    horizontal = np.diff(gray, axis=1)
    vertical = np.diff(gray, axis=0)
    return dict(
        zip(
            DEGRADATION_NAMES,
            (
                float(gray.mean()),
                float(gray.std()),
                float(np.sqrt((np.mean(horizontal**2) + np.mean(vertical**2)) / 2)),
            ),
            strict=True,
        )
    )


def risk_features(ocr: dict, degradation: dict) -> list[float]:
    values = [degradation[name] for name in DEGRADATION_NAMES]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Image statistics must be finite and in [0, 1]")
    result = features(ocr) + values
    if any(not math.isfinite(value) for value in result):
        raise ValueError("Invalid OCR features")
    return result


class RiskAwareRouter:
    """Estimate signed clipped-CER improvement and probability of strictly worse CER."""

    feature_names = FEATURE_NAMES + DEGRADATION_NAMES

    def __init__(self, seed: int = 42):
        self.benefit = make_pipeline(StandardScaler(), Ridge(alpha=1))
        self.regression = make_pipeline(
            StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000, solver="lbfgs")
        )

    def fit(self, rows: list[dict]) -> None:
        if not rows or any(row["split"] != "train" for row in rows):
            raise ValueError("Risk router can fit only the training split")
        if any(
            row.get("reference_status", "reviewed" if row.get("synthetic") is True else None)
            != "reviewed"
            for row in rows
        ):
            raise ValueError("Full-image reference order must be reviewed before training")
        gains = [
            clipped_cer(row["reference"], row["ocr"]["text"])
            - clipped_cer(row["reference"], row["vlm"]["text"])
            for row in rows
        ]
        labels = [int(gain < 0) for gain in gains]
        if len(set(labels)) != 2:
            raise ValueError("Training requires both regression and no-regression examples")
        inputs = [risk_features(row["ocr"], row["degradation"]) for row in rows]
        self.benefit.fit(inputs, gains)
        self.regression.fit(inputs, labels)

    def predict(self, ocr: dict, degradation: dict) -> dict[str, float]:
        inputs = [risk_features(ocr, degradation)]
        return {
            "expected_cer_gain": float(np.clip(self.benefit.predict(inputs)[0], -1, 1)),
            "regression_probability": float(self.regression.predict_proba(inputs)[0, 1]),
        }

    def select(self, candidates: list[dict], max_calls: int, risk_weight: float) -> list[str]:
        """Offline batch budget; positive utility only, with a validation-frozen risk weight."""
        if not isinstance(max_calls, int) or max_calls < 0:
            raise ValueError("max_calls must be a nonnegative integer")
        if not math.isfinite(risk_weight) or risk_weight < 0:
            raise ValueError("risk_weight must be finite and nonnegative")
        if len({row["id"] for row in candidates}) != len(candidates):
            raise ValueError("Duplicate candidate ID")
        ranked = []
        for row in candidates:
            prediction = self.predict(row["ocr"], row["degradation"])
            utility = (
                prediction["expected_cer_gain"] - risk_weight * prediction["regression_probability"]
            )
            if utility > 0:
                ranked.append((-utility, row["id"]))
        return [name for _, name in sorted(ranked)[:max_calls]]
