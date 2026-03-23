# ADR-0001: PostgreSQL as Single SQL Store (Replace SQLite)

Date: 2026-03-23
Status: Accepted
Deciders: @vola

---

## Context

The existing codebase uses SQLite for two purposes:
- `jobs.db` — background job queue (`pipeline/job_db.py`)
- `processed.db` — SHA-256 dedup registry (`pipeline/processed_db.py`)

The new architecture adds dataset metadata (`frames`, `missions` tables) and requires a
database that can be shared by multiple containers (worker, API, future CVAT annotation
service). SQLite does not support concurrent multi-process writes reliably in a Docker
Compose environment.

FiftyOne (evaluated for dataset management) was rejected because it is MongoDB-only and
cannot use PostgreSQL.

## Decision

Use **PostgreSQL 16** (`postgres:16` Docker Compose service) as the single SQL store for
all relational data. SQLite is removed entirely.

Tables migrated from SQLite:
- `jobs` — replaces `jobs.db`
- `processed_files` — replaces `processed.db`

New tables:
- `missions` — mission-level status and metadata
- `frames` — per-frame dataset metadata, captions, active learning tags, pose, GPS

`pipeline/job_db.py` and `pipeline/processed_db.py` are rewritten to use PostgreSQL
(`psycopg2-binary` or `asyncpg`). All other code interacting with these modules is
unchanged at the call-site level.

## Consequences

**Good:**
- Single database for all SQL-based activity — simpler ops, one backup target
- Multi-container safe (API + worker can both write jobs concurrently)
- CVAT (v2 annotation service) uses PostgreSQL natively — no extra database needed when
  annotation is added
- Richer query capabilities for dataset exploration (e.g., `SELECT * FROM frames WHERE
  active_learning_score > 0.7 AND mission_id = $1`)

**Bad / Tradeoffs:**
- PostgreSQL requires a running container; SQLite had zero infrastructure overhead for
  development. Mitigated by Docker Compose making `postgres:16` trivial to run locally.
- Existing `jobs.db` and `processed.db` files are not automatically migrated —
  first-time setup requires re-indexing or a one-time migration script.
