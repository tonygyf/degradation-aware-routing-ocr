"""Read-only environment inspection. Does not import ML runtimes or initialize CUDA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    if not shutil.which(args[0]):
        return {"available": False}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def inspect():
    packages = {}
    for name in (
        "numpy",
        "scikit-learn",
        "paddleocr",
        "paddlepaddle",
        "torch",
        "transformers",
        "pytest",
        "ruff",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "packages": packages,
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv",
            ]
        ),
        "gpu_processes": command(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"]
        ),
        "processes": command(["ps", "-eo", "pid,comm,pcpu,pmem", "--sort=-pcpu"])
        if platform.system() != "Windows"
        else command(["tasklist", "/fi", "IMAGENAME eq python.exe"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(inspect(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
