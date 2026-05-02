# ADR-0001: PostgreSQL as the Relational System of Record

Date: 2026-03-23  
Status: Accepted

## Context

The system needs one relational store that can be shared safely by the API, worker,
realtime services, analytics, and optional annotation / coop integrations. SQLite was
insufficient for concurrent multi-process/container writes.

## Decision

Use PostgreSQL as the single SQL backend for operational and metadata tables.

Current examples:
- `jobs` — async job queue
- `processed_files` — dedup registry
- `missions`, `frames`, and related metadata tables
- realtime / automation / admin state written by the app and worker

Implementation is async-first and centered on `asyncpg`:
- `src/selfsuvis/pipeline/storage/jobs.py`
- `src/selfsuvis/pipeline/storage/processed.py`

## Consequences

Positive:
- One relational system of record for app, worker, and integrations
- Safe concurrent writes with standard PostgreSQL locking patterns
- Better querying for mission, frame, and annotation metadata

Trade-offs:
- Requires a running PostgreSQL service in local and container workflows
- Schema changes require explicit migration discipline
