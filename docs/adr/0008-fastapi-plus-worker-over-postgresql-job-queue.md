# ADR-0008: Separate API Control Plane from Background Execution with a PostgreSQL Job Queue

Date: 2026-05-02  
Status: Accepted

## Context

Indexing, mapping, replay processing, training, and re-embedding are too slow
and resource-heavy to run inline in FastAPI request handlers. The product still
needs a simple local and container-friendly execution model without
introducing a separate workflow platform.

## Decision

Keep the system split into:
- FastAPI for request handling, status, and orchestration
- a separate worker process for long-running jobs
- PostgreSQL `jobs` as the queue and coordination mechanism

Current implementation:
- API entry point: `src/selfsuvis/app/main.py`
- worker entry point: `src/selfsuvis/worker/main.py`
- queue implementation: `src/selfsuvis/pipeline/storage/jobs.py`
- realtime post-flight chaining from `src/selfsuvis/app/services/realtime.py`

Workers claim pending jobs with transactional `SELECT ... FOR UPDATE SKIP LOCKED`
semantics. This keeps concurrency control in PostgreSQL instead of adding Redis,
Celery, or an external scheduler.

The worker now treats long post-flight stages as explicit job types rather than
burying them inside a single coarse `index` payload. Current first-class job
types include:
- `index`
- `supervised_finetune`
- `reembed`
- `postflight_mapping`
- `postflight_semantic_graph`

Chained post-flight jobs are enqueued after successful indexing instead of
running invisibly inside the main index stage.

## Consequences

Positive:
- API latency stays decoupled from indexing and training workloads
- One database coordinates both operational metadata and job execution
- Local deployment remains simpler than a multi-service task queue stack
- Operators can observe, retry, and reason about post-flight stages
  independently

Trade-offs:
- Worker lifecycle, retries, and observability are more custom than a dedicated
  workflow platform
- Long-running jobs still require careful GPU and process isolation
- Queue throughput scales only as far as the current worker design and database
  polling pattern allow
- Job chaining and mission-status transitions are now part of application code,
  not a separate orchestration product
