"""Paper L2 route-benefit/flag predictor; unreported settings remain explicit."""

import math
import warnings
from statistics import mean

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.ccocr import normalize
from src.evaluation.shc_analysis import routing_labels
from src.evaluation.shc_routing import validate_paired
from src.routing.confroute import ConfRouteRouter

FEATURE_SETS = {
    "mean": ("conf_mean",),
    "min": ("conf_min",),
    "geom": ("conf_geom_mean",),
    "density": ("text_density",),
    "requested": ("conf_min", "conf_geom_mean", "text_density"),
    "full": (
        "conf_min",
        "conf_geom_mean",
        "conf_mean",
        "text_density",
        "character_density",
        "box_count",
        "characters_per_box",
        "empty_prediction",
    ),
}


class PaperSHCRouter:
    def __init__(self, config):
        self.config = dict(config)
        if config["feature_set"] not in FEATURE_SETS:
            raise ValueError("Unknown published feature set")
        if config["target"] not in {"route_benefit", "operational_flag"}:
            raise ValueError("Unknown learning target")
        if not math.isfinite(config["regularization_C"]) or config["regularization_C"] <= 0:
            raise ValueError("L2 C must be finite and positive")
        if not isinstance(config["standardize"], bool):
            raise ValueError("Declare standardize explicitly")
        if not math.isfinite(config["threshold"]) or not 0 <= config["threshold"] <= 1:
            raise ValueError("Declare a finite frozen decision threshold")
        if config["max_iter"] < 1 or config["solver"] not in {"lbfgs", "liblinear"}:
            raise ValueError("Invalid optimizer configuration")
        if config["threshold_target"] != config["target"]:
            raise ValueError("Frozen threshold must belong to the same learning target")
        if not config.get("threshold_source"):
            raise ValueError("Declare threshold provenance; never choose it on Test")
        self.feature_names = FEATURE_SETS[config["feature_set"]]
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler() if config["standardize"] else "passthrough"),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=config["regularization_C"],
                        solver=config["solver"],
                        max_iter=config["max_iter"],
                        random_state=config["seed"],
                        class_weight=None,
                    ),
                ),
            ]
        )
        self.fitted = False

    def features(self, ocr):
        minimum = ConfRouteRouter("box_min").confidence(ocr)
        scores = [b["confidence"] for b in ocr["confidence_boxes"]]
        known = {
            "conf_min": minimum,
            "conf_mean": mean(scores) if scores else 0.0,
            "conf_geom_mean": math.exp(mean(math.log(s) for s in scores))
            if scores and min(scores) > 0
            else 0.0,
            "box_count": len(scores),
            "empty_prediction": float(not normalize(ocr["text"])),
        }
        values = []
        for name in self.feature_names:
            if name in known:
                value = known[name]
            else:
                # Density and character-count definitions were not released; do not invent them.
                if (
                    not isinstance(ocr.get("paper_feature_provenance"), str)
                    or not ocr["paper_feature_provenance"].strip()
                ):
                    raise ValueError("Provide audited paper feature definitions/provenance")
                value = ocr["paper_features"][name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Cheap features must be finite and nonnegative")
            values.append(value)
        return values

    def fit(self, rows):
        if not rows or any(r.get("role") != "Tau_dev" or r.get("split") != "train" for r in rows):
            raise ValueError("Fit only Tau_dev observations represented as train")
        validate_paired(rows)
        labels = routing_labels(rows, self.config["target"])
        if len(set(labels)) != 2:
            raise ValueError("Both target classes are required")
        features = [self.features(r["ocr"]) for r in rows]
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.pipeline.fit(features, labels)
        self.fitted = True
        self.fit_ids = [r["id"] for r in rows]
        self.train_positive_rate = mean(labels)
        return self

    def score(self, ocr):
        if not self.fitted:
            raise ValueError("Fit before scoring")
        return float(self.pipeline.predict_proba(np.array([self.features(ocr)]))[0, 1])

    def metadata(self):
        if not self.fitted:
            raise ValueError("Fit before exporting")
        scale, model = self.pipeline.named_steps["scale"], self.pipeline.named_steps["model"]
        return {
            "config": self.config,
            "author_optimizer_and_scaler_verified": False,
            "feature_names": self.feature_names,
            "fit_ids": self.fit_ids,
            "train_positive_rate": self.train_positive_rate,
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "scaler_mean": scale.mean_.tolist() if self.config["standardize"] else None,
            "scaler_scale": scale.scale_.tolist() if self.config["standardize"] else None,
            "threshold_note": (
                "Frozen externally supplied threshold; no Test threshold optimization"
            ),
        }
