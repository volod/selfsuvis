# AGENTS.md project rules

## Project guardrails

- Do not create git commits unless explicitly asked.
- Do not revert user changes or unrelated dirty files while fixing an issue.
- Do not add `from __future__ import annotations`; use normal annotations and `TYPE_CHECKING` imports when needed.
- Keep top-level `scripts/` as shell entrypoints. Put production Python implementations under `src/selfsuvis/...`, local-pipeline implementations under `src/ssv_vdp/...`.
- Reuse `scripts/shared/common.sh` for shared shell root/env/bootstrap behavior.
- Runtime data belongs under `.data/<src_module>/` (e.g. `.data/nanochat/`, `.data/sslm/`). Never write to a module-local `.data/` inside `src/`. 
- The shared `.data/wheels/` directory is reserved exclusively for compiled wheel artifacts.
- Use ASCII in logs, docs, comments, and generated shell output.
- Dependencies and library versions, such as PyTorch and Flash Attention, must be installed, taking into account the host hardware configuration (RAM, GPU) and the current host OS, GPU, and CUDA versions, which must match PyTorch's supported versions.   
- Docker images must be able to manage the target build for GPU and CUDA versions; the configuration of the system on which the Docker image is built is used as the default.

## Heavy compilation (ninja / cmake / CUDA)

Any installation that compiles C++/CUDA from source (git+, --no-binary, --no-build-isolation) MUST cap
parallelism by querying the canonical helper for this project:
- Main project / sslm / xformers: `MAX_JOBS=$(.venv/bin/python src/selfsuvis/scripts/shell_helpers.py max-jobs)`
- nanochat: `MAX_JOBS=$(python3 scripts/detect_hw.py max_jobs)`

Do not inline the formula — the helpers are the single source of truth.

Compiled wheels (flash-attn, vllm forks, xformers, etc.) MUST be cached under `.data/wheels/<package-name>_<key>/`
where `<key>` encodes the ABI-relevant dimensions (e.g. `torch2.9.1_cu128`, `sslm_zaya-vllm_latest`).
Never cache wheels under a package-specific subdirectory such as `.data/sslm/wheels/` — the shared
`.data/wheels/` root is the single source of truth for all heavy build artifacts across every sub-package.

## Current layout

- API: `src/selfsuvis/app/`
- Worker: `src/selfsuvis/worker/`
- UI: `src/selfsuvis/ui/`
- Shared pipeline: `src/selfsuvis/pipeline/` (core, vision, mapping, fusion, training, media, storage, realtime)
- Local VDP pipeline: `src/ssv_vdp/` (standalone package; pyproject.toml lives inside — `pip install -e ./src/ssv_vdp`)
- Sencoop/IoT: `src/sencoop/` (standalone sensor-mesh package)
- Runtime config: `src/selfsuvis/pipeline/core/config/`
- Docker and shell ops: `docker/`, `scripts/`

## Usual commands

- `make venv` — installs selfsuvis + ssv_vdp (both editable)
- `make test-unit`, `make lint`
- `make up`, `make down`, `make logs`
- `python -m selfsuvis.scripts.migrate_postgres`
- `scripts/ssv/ssv-reset-qdrant.sh`
- `scripts/sencoop/sencoop-bootstrap.sh`
- `ssv --mode local --videos-dir .data/videos` — run local VDP pipeline
- `ssv --mode local --cosmos3 --cosmos3-api-url http://localhost:8000 ...` — enable Cosmos3 step 15 via vLLM-Omni sidecar
- `ssv --mode local --cosmos3 ...` — enable Cosmos3 with local diffusers (hardware-selected: Nano with offload if 18-40 GB free VRAM, Nano full if >= 40 GB)
