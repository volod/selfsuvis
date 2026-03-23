# ADR-0005: Active Tagging via PostgreSQL (FiftyOne Rejected)

Date: 2026-03-23
Status: Accepted
Deciders: @vola

---

## Context

The self-improvement moat depends on accumulating high-quality labeled training data from
mission video. To make annotation tractable, the system must automatically identify which
frames are most worth annotating (uncertain, novel, or representative).

FiftyOne was initially proposed as the dataset management layer. It provides:
- Per-frame metadata storage with ML-specific tooling
- Built-in active learning hooks and brain methods
- CVAT integration for annotation

FiftyOne was rejected because **it is MongoDB-only**. It cannot use PostgreSQL as its
database backend. Adding MongoDB to the stack solely for FiftyOne contradicts the
decision to consolidate all SQL-based storage in PostgreSQL (ADR-0001).

## Decision

Implement active tagging directly against the PostgreSQL `frames` table. No FiftyOne
dependency in v1.

**`pipeline/active_learning.py`** — runs after each mission's inference pass:

1. Compute per-frame uncertainty score:
   - `0.6 × DINOv3 embedding distance from nearest cluster centroid` (k-means, k=20,
     updated incrementally after each mission)
   - `0.4 × (1 − Florence-2 caption confidence)`
2. Diversity filter: if DINOv3 cosine similarity > 0.97 to an already-tagged frame,
   skip (no new information)
3. Write to PostgreSQL `frames.al_tag`:
   - Top-K frames by score → `needs_annotation` (K configurable via `AL_TAG_K` env var,
     default 50)
   - Frames with embedding distance > 0.5 from any cluster centroid → `novel`
   - Remainder → `none`
4. Mirror `active_learning_score` to Qdrant payload for spatial active learning queries
   ("show uncertain frames near this location")

**v1 scope:** tagging only. Model retraining is v2.

**v2:** CVAT annotation service uses PostgreSQL natively — it will share the same
PostgreSQL instance and write `al_tag=annotated` back to the `frames` table when a
frame is labeled. No schema change required.

## Consequences

**Good:**
- No MongoDB dependency
- Active tagging is simple SQL + numpy — easy to test, easy to reason about
- CVAT in v2 connects to the same PostgreSQL instance with no extra integration work
- `active_learning_score` in Qdrant enables spatial queries over uncertain frames

**Bad / Tradeoffs:**
- Lose FiftyOne's built-in dataset visualization UI (embedding projections, similarity
  search UI, brain methods). The Streamlit mission timeline view partially fills this gap.
- k-means cluster centroids must be managed in memory or persisted — initial
  implementation uses in-memory centroids re-computed per mission; v2 can persist to
  PostgreSQL if incremental update is needed.
- No AWML (Tier IV active learning framework) in v1 — it was released May 2025 with thin
  documentation. Revisit for v2 alongside the training pipeline.
