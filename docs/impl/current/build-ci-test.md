# Build, CI, and Test

## Python environments

- `make venv` -- `uv venv .venv` + `scripts/install/install_requirements.sh vision,dev .venv`;
  installs `selfsuvis` and `src/ssv_vdp` editable. Prompts recreate/update when
  `.venv` exists. `make venv-cuda` forces CUDA wheels when `nvidia-smi` is absent.
- Torch is intentionally **not** a pyproject dependency: `install_requirements.sh`
  selects the CUDA/CPU wheel index against detected host hardware (AGENTS.md rule).
- Extras in `pyproject.toml`: `vision` (full CUDA stack: torchvision, open-clip,
  ultralytics, xformers, SAM2/3, rfdetr, gsplat, pycolmap), `mapper` (CPU ICP
  service), `sencoop` (aiomqtt + analytics), `dev` (pytest, ruff).
- Separate venvs per playground: `.venv` (main + ssv_vdp), `.venv-sslm`
  (`scripts/sslm/setup-venv.sh`), `src/nanochat/.venv` (its own Makefile).

## Heavy native builds

- Parallelism is always capped via the canonical helpers -- main/sslm/xformers:
  `MAX_JOBS=$(.venv/bin/python src/selfsuvis/scripts/shell_helpers.py max-jobs)`;
  nanochat: `MAX_JOBS=$(python3 scripts/detect_hw.py max_jobs)`. Never inline the formula.
- Compiled wheels (flash-attn, vllm forks, xformers) are cached under
  `.data/wheels/<package>_<abi-key>/` -- the single shared cache root.
- `make venv-rebuild-xformers` rebuilds xformers for the detected compute capability.

## Docker composition

| Compose file | Stack |
| --- | --- |
| `docker/core/docker-compose.yml` (+ `.override`, `.no-gpu`) | api, worker, ui, qdrant, mapper |
| `docker/sencoop/docker-compose.sencoop.yml` | IoT edge stack (profiles: lorawan, video, metrics) |
| `docker/realtime/*.yml` | MediaMTX, SLAM engines, bridge runtimes |
| `docker/cvat/docker-compose.cvat.yml` | Annotation service |
| `docker/vllm/docker-compose.vllm.yml` | Reasoning/vision sidecar |
| `docker/test/docker-compose.test.yml` | Integration test harness (`Dockerfile.tests`) |

Docker images select GPU/CUDA targets from the build host configuration
(AGENTS.md rule). `UID`/`GID` are injected so bind mounts stay user-owned.

## Tests

- `make test-unit` -- host unit tests, no services (`tests/unit/` mirrors
  `src/selfsuvis/`; fakes and factories in `tests/support/`).
- `make test` / `make test-no-gpu` -- full integration in Docker (api + worker +
  qdrant + tests container, `INDEX_DIR_PATH=/app/tests/assets`).
- Markers: `slow`, `integration`, `load` (locust file in sencoop tests).
- `make lint` -- `ruff check` + `ruff format --check` (line length 100,
  rules E/F/W/I/UP). Pyright configured `basic` against `.venv`.

## CI (GitHub Actions)

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `.github/workflows/openapi.yml` | PR touching `src/selfsuvis/app/**` | OpenAPI spec diff gate (`make export-openapi` artifact `docs/api/v1-openapi.json`) |
| `.github/workflows/claude-code-review.yml` | PR open/sync | Automated review |
| `.github/workflows/claude.yml` | issue/PR comments | Interactive agent |

## Known gaps (drive the forward plan)

- No unified `make ci` aggregate target; the de-facto agent gate is
  `make lint` + `make test-unit`.
- No CI jobs run lint/unit tests on push/PR (only review + OpenAPI gates).
- No firmware, Go, or FPGA toolchain anywhere in the build system yet; the
  cross-stack build/CI work is specified in [`../plan.md`](../plan.md)
  (`ci-cross-stack`, `firmware-workspace`, `sencoop-agent-go`).
