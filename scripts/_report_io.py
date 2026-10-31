"""Shared I/O for paper reports; preserve input checks and existing evidence."""

import json
from pathlib import Path


def read_real_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or any(row.get("synthetic") is not False for row in rows):
        raise ValueError(
            "Require nonempty real observations; synthetic fixtures are software checks"
        )
    return rows


def write_report(path: Path, report: dict) -> None:
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(content)
