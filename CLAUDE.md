# CLAUDE.md

This file provides compact repository guidance for coding agents.

## Rules

- Never create a git commit unless the user explicitly asks for one.
- Never add `from __future__ import annotations`, and replace those cases with explicit imports TYPE_CHECKING.
- Top-level `scripts/` must be shell entrypoints only. Put Python implementations under `src/selfsuvis/...` and call them from shell wrappers when needed.
- Reuse `scripts/shared/common.sh` for shared shell behavior instead of duplicating root/env/bootstrap logic.
- Use ASCII-only characters in all log messages, docstrings, comments, and documentation. No emoji, no Unicode box-drawing or symbol characters (no ✓ ▷ ═ ─ ℹ ⚠ ● or similar). Use plain ASCII equivalents: `[ok]`, `->`, `=`, `-`, `[info]`, `[warn]`, `*`.

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
- `scripts/ssv/ssv-reset-qdrant.sh`

## Config

- Runtime settings live in `src/selfsuvis/pipeline/core/config.py`.
- Python target is 3.10+.
- For deeper architecture or workflow details, use `docs/` and the source tree instead of expanding this file.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
