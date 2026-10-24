# Project guardrails

- Apply the installed `ponytail` and `grill-me` skills throughout this project's work. Inspect available evidence first, keep implementations minimal, and resolve material unknowns one question at a time while continuing independent work.
- Use `docker compose run --rm dev ...` for all development, checks and experiments. Rebuild with `docker compose build dev` after dependency changes; do not create host virtual environments or install project packages on the host.
- On Linux, export `LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)` before Compose commands so generated files remain owned by the developer. Do not mount the Docker socket, SSH keys or host credentials into the container.
- This is an independent research project. Do not modify sibling projects or existing containers.
- Stage 1 uses CPU only. Never stop or reuse running training processes. Recheck resources before real inference.
- User exception (2026-10-24): this public CC-OCR experiment may run Qwen3-VL-8B in a new isolated container on host GPU 1 only, with 8 CPU cores and 64 GiB RAM. This does not authorize other GPUs or changes to existing training.
- Do not download model weights implicitly. Record model revisions, licenses, data authorization and resource needs before real inference.
- Synthetic fixtures and cached mock latency are software checks, never scientific evidence.
- Fit routing labels/scalers only on train; choose thresholds on val; freeze test and original-image groups.
- Keep secrets, private images, checkpoints and experiment predictions out of Git.
- Run `python -m pytest -q`, `ruff check .`, `ruff format --check .`, and `python -m scripts.smoke` for baseline changes.
