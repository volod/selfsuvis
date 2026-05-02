# ADR-0002: Dual Vector Strategy — CLIP for Cross-Modal Search, DINO for Visual Similarity

Date: 2026-03-23  
Status: Accepted

## Context

The product needs both:
- text-to-image and image-to-text retrieval
- image-only similarity and active-learning signals suited to robot / mission footage

No single embedding space serves both goals well.

## Decision

Keep two named vector spaces in Qdrant:
- `clip` — OpenCLIP for cross-modal text/image retrieval
- `dino` — DINO-family embeddings for visual similarity, reranking, and active learning

Current behavior:
- text queries search `clip`
- image queries search `clip` and may rerank with `dino`
- active learning uses DINO-distance-derived novelty / uncertainty signals

## Consequences

Positive:
- Text search remains valid because CLIP stays the text-aligned space
- Visual similarity and active learning use a stronger image-only representation
- Fits the current Qdrant named-vector design without a schema redesign

Trade-offs:
- Two embedders increase memory and indexing cost
- Qdrant reindexing is required when vector-space assumptions change
