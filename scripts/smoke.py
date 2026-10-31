from __future__ import annotations

import json
from pathlib import Path

from src.datasets.manifest import load_manifest
from src.datasets.paper_sources import audit_sources
from src.evaluation.baseline import lambda_sensitivity, replay_calibration, replay_development
from src.evaluation.ccocr import evaluate
from src.evaluation.shc import evaluate as evaluate_shc
from src.evaluation.shc_analysis import confidence_ranking, risk_coverage
from src.evaluation.shc_audit import reweighted_flag_audit
from src.evaluation.shc_prompts import compare_prompts
from src.evaluation.shc_routing import sensitivity
from src.routing.confroute import ConfRouteRouter
from src.routing.risk import RiskAwareRouter
from src.routing.shc_learned import PaperSHCRouter
from src.run import run


def smoke_ccocr():
    """Exercise the paper replay with synthetic observations, never research evidence."""
    rows = [
        {
            "id": f"synthetic-{index}",
            "subset": subset,
            "reference": "A",
            "ocr": {
                "text": "B" if scores else "",
                "rec_texts": ["B"] if scores else [],
                "confidences": scores,
                "latency_ms": 2,
            },
            "vlm": {"text": "A", "latency_ms": 10},
        }
        for index, (subset, scores) in enumerate(
            zip(("TotalText", "InverseText", "Hieragent"), ([0.89996], [0.89], []), strict=True)
        )
    ]
    report, predictions = evaluate(rows)
    assert set(report) == {"paddleocr", "qwen", "confroute"}
    assert len(predictions) == 9
    routed = [row for row in predictions if row["system"] == "confroute"]
    assert [row["vlm_called"] for row in routed] == [False, True, True]
    assert [row["latency_ms"] for row in routed] == [2, 12, 12]
    assert report["paddleocr"]["pooled"]["cer_macro_clipped"] == 1
    assert report["qwen"]["pooled"]["cer_macro_clipped"] == 0
    assert report["confroute"]["pooled"]["cer_macro_clipped"] == 1 / 3
    assert set(report["confroute"]["subsets"]) == {"TotalText", "InverseText", "Hieragent"}
    print("ccocr synthetic software check (mock latency; not scientific evidence): PASS")


def smoke_paper_analyses():
    """All observations below are synthetic software fixtures, not SHC evidence."""
    train = [
        {
            "id": f"train-{i}",
            "role": "Tau_dev",
            "split": "train",
            "synthetic": True,
            "source_corpus": "sroie",
            "source_image": f"source-{i}.jpg",
            "failure_type": "L1",
            "reference_visible": "A",
            "reference_full": "A",
            "critical_span": None,
            "ocr": {
                "text": "B" if i % 2 == 0 else "A",
                "latency_ms": 2,
                "confidence_boxes": [{"confidence": 0.2 if i % 2 == 0 else 0.9}],
                "paper_features": {"text_density": 0.1 if i % 2 == 0 else 0.8},
                "paper_feature_provenance": "synthetic smoke fixture",
            },
            "vlm": {"text": "A" if i % 2 == 0 else "B", "latency_ms": 10},
            "internvl": {"text": "", "latency_ms": 20},
        }
        for i in range(4)
    ]
    model = PaperSHCRouter(json.loads(Path("configs/paper_shc_router.json").read_text())).fit(train)
    test = [{**r, "id": f"test-{i}", "role": "Test", "split": "test"} for i, r in enumerate(train)]
    assert model.score(test[0]["ocr"]) > model.score(test[1]["ocr"])
    assert confidence_ranking(test)["targets"]["route_benefit"]["auroc"] == 1
    assert len(risk_coverage(test, random_resamples=20)["points"]) == 12
    assert sensitivity(test, include_internvl=True)["selected_threshold"] is None
    prompts = [
        {**r, "model": "qwen", "prompt": p, "prediction": r["vlm"]["text"], "latency_ms": 10}
        for r in test
        for p in ("P0", "P2")
    ]
    assert compare_prompts(prompts)["selected_dev_prompts"] is None
    sources = audit_sources([{**train[0], "id": "baseline-test", "role": "Test"}], train + test)
    assert sources["shc_parents_in_multiple_roles"] == 4
    cal = [
        {
            **r,
            "role": "Cal",
            "split": "val",
            "reference": r["reference_full"],
            "ocr": {
                **r["ocr"],
                "rec_texts": [r["ocr"]["text"]],
                "confidences": [ConfRouteRouter("box_min").confidence(r["ocr"])],
            },
        }
        for r in train
    ]
    assert replay_calibration(cal)["selected_threshold"] is None
    assert lambda_sensitivity(cal)["historical_policy_modified"] is False
    dev = [{**r, "role": "Dev"} for r in cal]
    assert len(replay_development(dev)["signals"]["box_min"]) == 21
    json.dumps(
        {
            "model": model.metadata(),
            "curve": risk_coverage(test, random_resamples=2),
            "sources": sources,
            "sensitivity": sensitivity(test, include_internvl=True),
        }
    )
    audit = reweighted_flag_audit(10, 10, {"true_positive": 4, "false_positive": 1})
    assert audit["population_reweighted_recall"] is None
    print("paper analyses synthetic software check (not scientific evidence): PASS")


def main():
    root = Path(__file__).resolve().parents[1]
    for name in ("baseline_a", "baseline_b", "baseline_b_mlp"):
        output = run(root / f"configs/{name}.json", root / "outputs/smoke")
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["samples"] == 16
        assert 0 <= metrics["vlm_calls"] <= 16
        print(name, json.dumps(metrics, ensure_ascii=False))
    smoke_ccocr()
    smoke_paper_analyses()
    shc = evaluate_shc(
        [
            {
                "failure_type": "S6",
                "reference_visible": "",
                "reference_full": "",
                "critical_span": None,
                "prediction": "There is no text.",
            }
        ]
    )
    assert shc["pooled"]["semantic_share"] == 1
    assert shc["pooled"]["operational_flag_rate"] == 1
    assert ConfRouteRouter("box_min").score({"confidence_boxes": []}) == 1
    print("SHC synthetic software check (not scientific evidence): PASS")
    rows = load_manifest(root / "data/fixtures/observations.jsonl")
    for row in rows:
        row["degradation"] = {"brightness_mean": 0.5, "contrast_std": 0, "gradient_rms": 0}
    router = RiskAwareRouter()
    router.fit([row for row in rows if row["split"] == "train"])
    selected = router.select([row for row in rows if row["split"] == "test"], 4, 1)
    assert len(selected) <= 4
    print("risk router synthetic check (mock image statistics; not scientific evidence): PASS")


if __name__ == "__main__":
    main()
