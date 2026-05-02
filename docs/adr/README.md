# Architecture Decision Records

This directory holds compact ADRs for decisions that still shape the current system.
Each record captures:

- the problem being solved
- the decision that remains in force
- the current practical consequences

Historical exploration details are intentionally omitted unless they still affect the
present architecture.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-postgresql-as-single-sql-store.md) | PostgreSQL as the Relational System of Record | Accepted |
| [0002](0002-dual-embedding-vectors-clip-and-dino.md) | Dual Vector Strategy — CLIP for Cross-Modal Search, DINO for Visual Similarity | Accepted |
| [0003](0003-florence2-for-image-captioning.md) | Florence-2 as the Default Frame Captioner | Accepted |
| [0004](0004-pycolmap-plus-nerfstudio-splatfacto-for-3d-mapping.md) | pycolmap for Poses, Gaussian-Splat Mapping for Dense 3D Output | Accepted |
| [0005](0005-active-tagging-pipeline-postgresql-not-fiftyone.md) | Active Learning Tags Live in Core Storage, Not a Separate Dataset Platform | Accepted |
| [0006](0006-mediamtx-for-video-streaming.md) | MediaMTX as the Streaming Edge for Live Video Ingest | Accepted |
| [0007](0007-qdrant-as-vector-store-with-named-vectors.md) | Qdrant as the Vector Store with Named Vector Spaces | Accepted |
| [0008](0008-fastapi-plus-worker-over-postgresql-job-queue.md) | Separate API Control Plane from Background Execution with a PostgreSQL Job Queue | Accepted |
| [0009](0009-coop-pilot-as-optional-lazy-integration.md) | Keep coop_pilot as an Optional, Lazy-Started Integration Layer | Accepted |
| [0010](0010-graceful-degradation-for-optional-analysis-and-realtime.md) | Prefer Graceful Degradation over Hard Failure for Optional Analysis Paths | Accepted |
| [0011](0011-realtime-mapping-as-optional-sidecars-plus-bridge-runtimes.md) | Realtime Mapping Uses Optional Sidecars, Owned Bridge Runtimes, and Chained Post-Flight Jobs | Accepted |

## Related

- [Overview](../overview.md)
- [Architecture](../architecture.md)
