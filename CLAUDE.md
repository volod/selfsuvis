# CLAUDE.md

This file provides compact repository guidance for coding agents.

## Rules

- Never create a git commit unless the user explicitly asks for one.
- Never add `from __future__ import annotations`, and replace those cases with explicit imports TYPE_CHECKING.
- Keep `scripts/` flat; do not introduce new script subdirectories.
- Top-level `scripts/` must be shell entrypoints only. Put Python implementations under `src/selfsuvis/...` and call them from shell wrappers when needed.
- Reuse `scripts/common.sh` for shared shell behavior instead of duplicating root/env/bootstrap logic.

## Project

Outdoor autonomy perception stack with:

- FastAPI API
- async worker pipeline
- Streamlit UI
- local media / mapping / fusion / training workflows
- optional CoOP IoT integration for MQTT, Frigate, RTSP, and edge site state

## Repo Map

- `src/selfsuvis/app/` — FastAPI API
- `src/selfsuvis/worker/` — background worker
- `src/selfsuvis/ui/` — Streamlit UI
- `src/selfsuvis/pipeline/` — media, vision, mapping, fusion, storage, training
- `src/selfsuvis/pipeline/workflows/local/` — local and LangGraph orchestration
- `src/selfsuvis/coop_pilot/` — edge IoT integration and analytics
- `docker/` — container configs
- `docs/` — reference docs
- `scripts/` — operational shell entrypoints

## Preferred Commands

- `make venv`
- `make test-unit`
- `make lint`
- `make up`
- `make down`
- `make logs`
- `python -m selfsuvis.scripts.migrate_postgres`
- `scripts/reset_qdrant.sh`

## Config

- Runtime settings live in `src/selfsuvis/pipeline/core/config.py`.
- Python target is 3.10+.
- For deeper architecture or workflow details, use `docs/` and the source tree instead of expanding this file.
