# ADR-0002: Dual Embedding Vectors — OpenCLIP (clip) + DINOv3 (dino)

Date: 2026-03-23
Status: Accepted
Deciders: @vola

---

## Context

The existing pipeline uses OpenCLIP for both image and text embeddings, stored in Qdrant
as a single `clip` named vector. This enables cross-modal text↔image search.

Two pressures pushed toward changing this:
1. Generic CLIP (trained on web images) is suboptimal for egocentric robot camera views.
   R3M and DINOv3 are better priors for robot POV video perception.
2. The active learning tagging system requires a visual similarity score that is
   independent of text-alignment — DINOv3 embeddings cluster more meaningfully for
   uncertainty scoring on outdoor scenes.

A naive replacement (swap CLIP for DINOv3 everywhere) would break text search: CLIP and
DINOv3 embeddings live in different vector spaces. Text queries produce OpenCLIP text
embeddings; comparing those against DINOv3 image embeddings produces semantically
meaningless results.

## Decision

Retain **OpenCLIP** for the `clip` named vector (cross-modal text↔image search).
Add **DINOv3** as a second named vector `dino` (visual similarity reranking + active
learning uncertainty scoring). The `dino_model.py` module already supports dinov3 —
upgrade pretrained weights.

Search paths:
- **Text query:** OpenCLIP text encoder → `clip` Qdrant search → results; optionally
  reranked by `dino` score (70/30 blend, existing pattern in `app/services/search.py`)
- **Image query:** OpenCLIP image encoder → `clip` search + DINOv3 → `dino` reranking
- **Active learning:** DINOv3 embedding distance from cluster centroids (k-means, k=20)
  combined with Florence-2 caption confidence

Switching the primary image embedding to DINOv3 requires wiping Qdrant and re-indexing
(see `scripts/reset_qdrant.sh`). The `dino` named vector already exists in the schema.

## Consequences

**Good:**
- Text search continues to work correctly (OpenCLIP text↔image cross-modal)
- Visual similarity and active learning use a model better suited to robot camera views
- Matches existing dual-vector pattern already in the codebase
- No architectural change to Qdrant schema (named vectors already defined)

**Bad / Tradeoffs:**
- Two models loaded in the worker (OpenCLIP + DINOv3) increases GPU memory requirements
- Existing Qdrant index must be wiped and re-indexed when upgrading to DINOv3 weights
- Dual-vector reranking adds latency to image queries (~10-20ms on GPU)
