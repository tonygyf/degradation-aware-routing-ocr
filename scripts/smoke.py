from __future__ import annotations

import json
from pathlib import Path

from src.run import run


def main():
    root = Path(__file__).resolve().parents[1]
    for name in ("baseline_a", "baseline_b", "baseline_b_mlp"):
        output = run(root / f"configs/{name}.json", root / "outputs/smoke")
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["samples"] == 16
        assert 0 <= metrics["vlm_calls"] <= 16
        print(name, json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
