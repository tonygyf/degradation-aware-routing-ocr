from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import platform
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.manifest import load_manifest, sha256
from src.evaluation.metrics import summarize
from src.recognition.backends import CachedVLM
from src.routing.policies import FixedThresholdRouter, LearnedRouter


def evaluate(
    rows: list[dict],
    router,
    decision_threshold: float,
    max_calls: int,
    max_cost_units: float,
    cost_per_call: float,
    vlm=None,
) -> list[dict]:
    if not 0 <= decision_threshold <= 1 or max_calls < 0:
        raise ValueError("Invalid decision threshold or call budget")
    if not all(math.isfinite(x) and x >= 0 for x in (max_cost_units, cost_per_call)):
        raise ValueError("Cost and budget must be finite and nonnegative")
    backend, records, calls = vlm or CachedVLM(), [], 0
    for row in sorted(rows, key=lambda x: x["id"]):
        ocr = row["ocr"]
        start = time.perf_counter()
        score = router.score(ocr)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid routing score")
        route = score >= decision_threshold
        allowed = calls < max_calls and (calls + 1) * cost_per_call <= max_cost_units
        called = route and allowed
        routing_ms = (time.perf_counter() - start) * 1000
        result = backend.infer(row) if called else ocr
        calls += called
        records.append(
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "reference": row["reference"],
                "ocr_text": ocr["text"],
                "prediction": result["text"],
                "score": score,
                "vlm_called": called,
                "budget_denied": route and not allowed,
                "routing_ms": routing_ms,
                "latency_ms": ocr["latency_ms"] + (result["latency_ms"] if called else 0),
            }
        )
    return records


def run(config_path: Path, output_root: Path) -> Path:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["backend"] != "cached":
        raise ValueError("Use src.cache for explicit real inference, then evaluate cached results")
    root = config_path.resolve().parent.parent
    manifest = root / cfg["manifest"]
    rows = load_manifest(manifest)
    if any(r.get("latency_kind") != cfg["latency_kind"] for r in rows):
        raise ValueError("Configured latency kind must match every cached observation")
    if any(not any(r["split"] == split for r in rows) for split in ("train", "val", "test")):
        raise ValueError("Manifest must contain train, val and test")
    if cfg["eval_split"] not in {"val", "test"}:
        raise ValueError("Evaluate val or test only")
    random.seed(cfg["seed"])
    import numpy as np
    from threadpoolctl import threadpool_limits

    np.random.seed(cfg["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"{stamp}-{cfg['name']}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(str(output))
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    logger.addHandler(handler)
    start = time.perf_counter()
    try:
        logger.info("start name=%s seed=%s split=%s", cfg["name"], cfg["seed"], cfg["eval_split"])
        with threadpool_limits(limits=1):
            if cfg["router"] == "fixed":
                router = FixedThresholdRouter(cfg["confidence_threshold"])
                state = {"confidence_threshold": cfg["confidence_threshold"]}
            else:
                router = LearnedRouter(cfg["seed"], cfg["router"])
                router.fit([r for r in rows if r["split"] == "train"])
                state = router.state()
            selected = [r for r in rows if r["split"] == cfg["eval_split"]]
            records = evaluate(
                selected,
                router,
                cfg["decision_threshold"],
                cfg["max_calls"],
                cfg["max_cost_units"],
                cfg["cost_per_call"],
            )
        metrics = summarize(records, cfg["cost_per_call"])
        metrics["budget_denied"] = sum(r["budget_denied"] for r in records)
        metrics["routing_ms_mean"] = sum(r["routing_ms"] for r in records) / len(records)
        metrics["latency_kind"] = cfg["latency_kind"]
        metrics["wall_seconds_including_fit"] = time.perf_counter() - start
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
        )
        metadata = {
            "created_utc": stamp,
            "config_sha256": sha256(config_path),
            "manifest_sha256": sha256(manifest),
            "git_commit": git.stdout.strip() or None,
            "git_dirty": bool(status.stdout),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": 1,
            "packages": {
                k: importlib.metadata.version(k)
                for k in ("numpy", "scikit-learn", "scipy", "threadpoolctl")
            },
            "split_ids": {
                s: sorted(r["id"] for r in rows if r["split"] == s)
                for s in ("train", "val", "test")
            },
        }
        for name, obj in (
            ("config", cfg),
            ("metrics", metrics),
            ("metadata", metadata),
            ("router", state),
        ):
            (output / f"{name}.json").write_text(
                json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
        logger.info("complete metrics=%s", json.dumps(metrics))
    except Exception:
        logger.exception("run failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args()
    print(run(args.config, args.output_root))


if __name__ == "__main__":
    main()
