# ADR-0005: Active Learning Tags Live in Core Storage, Not a Separate Dataset Platform

Date: 2026-03-23  
Status: Accepted

## Context

The pipeline needs to score frames for annotation priority and persist those decisions in
the same operational system used by the rest of the product.

An external dataset-management platform would add another persistence model and operator
surface for something the core stack already stores.

## Decision

Keep active-learning state in the product’s own storage model:
- frame-level tags and scores in SQL
- mirrored search/query payloads where useful

Current implementation:
- `src/selfsuvis/pipeline/analysis/active_learning.py`
- downstream storage and analytics in the main app / worker stack

The scoring model is mission-oriented and combines visual distance with caption-derived
signals, plus RSSM surprise when enabled.

## Consequences

Positive:
- No extra database or dataset-management runtime
- Annotation priority stays close to the rest of frame metadata
- Easier to query from the app, worker, and analytics layers

Trade-offs:
- Less out-of-the-box dataset visualization than dedicated annotation platforms
- Scoring and clustering behavior must be maintained in-house
