# CLAUDE.md

Compact guidance for Claude Code in this repository.

## Project guardrails

- Do not create git commits unless explicitly asked.
- Do not revert user changes or unrelated dirty files while fixing an issue.
- Do not add `from __future__ import annotations`; use normal annotations and `TYPE_CHECKING` imports when needed.
- Keep top-level `scripts/` as shell entrypoints. Put production Python implementations under `src/selfsuvis/...`, local-pipeline implementations under `src/ssv_vdp/...`.
- Reuse `scripts/shared/common.sh` for shared shell root/env/bootstrap behavior.
- Runtime data belongs under `.data/`; avoid recreating root `data/` unless a file explicitly still requires it.
- Use ASCII in logs, docs, comments, and generated shell output.

## Heavy compilation (ninja / cmake / CUDA)

Any installation that compiles C++/CUDA from source (git+, --no-binary, --no-build-isolation) MUST cap
parallelism via `ARG MAX_JOBS=4` (safe default) overridden by the caller with:
`MAX_JOBS = min(max(1, (nproc-2)//2), max(1, available_ram_gb//12))`.

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

## GStack

- Use GStack slash skills when the user asks for that workflow or the task clearly needs it.
- Use `/browse` from GStack for browser-backed web work; do not use `mcp__claude-in-chrome__*` tools.
- Useful routing: `/investigate` for bugs, `/review` for diff review, `/qa` or `/qa-only` for browser QA, `/plan-eng-review` for architecture, `/plan-ceo-review` for scope, `/design-review` for visual polish, `/ship` or `/land-and-deploy` for release flow, `/context-save` and `/context-restore` for handoff.
- Speckit slash commands live under `.claude/commands/`; use them only when the user invokes the Speckit flow.
