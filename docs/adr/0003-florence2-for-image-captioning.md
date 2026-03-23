# ADR-0003: Florence-2 for Image-to-Text Captioning

Date: 2026-03-23
Status: Accepted
Deciders: @vola

---

## Context

The system requires English text descriptions of video frames to enable natural language
search and to provide human-readable context in the dataset. Candidates evaluated:

| Model | Strengths | Weaknesses |
|---|---|---|
| Florence-2 (Microsoft) | Fast, open-source, strong on diverse scenes, `transformers` native | Less detail on complex outdoor scenes vs. larger models |
| LLaVA-1.6 | Strong detail, community support | Heavier, slower inference |
| InternVL2 | State-of-art on dense captioning, strong outdoor scene understanding | Larger model, less mature Python packaging |
| BLIP-2 | Lightweight, well-tested | Weaker captions vs. 2025 alternatives |

## Decision

Use **Florence-2** (`microsoft/Florence-2-large`, `transformers >= 4.41`) for v1.

Store per-frame: caption text + caption confidence score (from model output logits) in
both the Qdrant payload and the PostgreSQL `frames` table.

An open question is noted in the design doc: if Florence-2 captions prove insufficient
for detailed outdoor scene description during field testing, InternVL2 is the upgrade
path. The module interface (`models/florence_model.py`) is kept thin so the model can
be swapped without pipeline changes.

## Consequences

**Good:**
- Fast inference (~200ms/frame on GPU) — fits within the 10-minute end-to-end target
- `transformers`-native: consistent with existing model loading patterns
- Caption confidence score is directly usable as an active learning signal

**Bad / Tradeoffs:**
- May produce generic captions on visually complex or unusual outdoor terrain
- Caption quality on robot camera views (fisheye, low-light, motion blur) is untested —
  field validation required before committing to this model
- Caption confidence from logits is a proxy for uncertainty, not a calibrated score
