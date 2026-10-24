# Project guardrails

- This is an independent research project. Do not modify sibling projects or existing containers.
- Stage 1 uses CPU only. Never stop or reuse running training processes. Recheck resources before real inference.
- Do not download model weights implicitly. Record model revisions, licenses, data authorization and resource needs before real inference.
- Synthetic fixtures and cached mock latency are software checks, never scientific evidence.
- Fit routing labels/scalers only on train; choose thresholds on val; freeze test and original-image groups.
- Keep secrets, private images, checkpoints and experiment predictions out of Git.
- Run `python -m pytest -q`, `ruff check .`, `ruff format --check .`, and `python -m scripts.smoke` for baseline changes.
