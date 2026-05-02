# ADR-0007: Qdrant as the Vector Store with Named Vector Spaces

Date: 2026-05-02  
Status: Accepted

## Context

The system needs low-latency vector retrieval for:
- text-to-image search
- image similarity search
- frame and tile retrieval with metadata filters
- multiple embedding spaces per asset

This retrieval layer must sit alongside, not inside, the relational system of
record.

## Decision

Use Qdrant as the dedicated vector store and keep multiple named vector spaces in
the same collection.

Current implementation:
- `src/selfsuvis/pipeline/storage/qdrant.py`
- `src/selfsuvis/app/services/search.py`

Current vector layout:
- `clip` for cross-modal retrieval
- `dino` for visual similarity and reranking

Metadata stays in payloads and PostgreSQL-backed mission/frame records rather
than being modeled as a second relational index inside Qdrant.

## Consequences

Positive:
- Vector search stays separate from transactional SQL storage
- Named vectors support the current CLIP + DINO retrieval design cleanly
- Payload filters keep frame/tile search practical without a second search stack

Trade-offs:
- Operators must run and maintain a separate Qdrant service
- Collection resets or re-embedding sweeps are operationally significant events
- Retrieval behavior depends on payload discipline and vector-version hygiene
