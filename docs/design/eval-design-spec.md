# Eval Design Spec — Florence-2 Caption Quality
## Phase 1 Pass/Fail Gate for Scene Captioning

**Status:** ACTIVE
**Gate:** Phase 1 ships when Precision@5 ≥ 0.8 on general queries.
**Phase 2 trigger:** Precision@5 < 0.6 on vehicle-count queries (independent gate).
**Depends on:** Florence-2 running on at least one real mission (`pipeline/florence_model.py` deployed).

---

## 1. Why This Document Exists

"50-100 real keyframes + 20 queries" with no stratification produces noisy anecdotes —
not a decisive pass/fail gate. A poorly designed eval can (a) incorrectly pass a model
that fails on real queries, and (b) incorrectly fail a model that works well.

This spec makes the gate concrete, reproducible, and resistant to both failure modes.

---

## 2. Eval Set Construction

### Frame selection

| Requirement | Value |
|-------------|-------|
| Minimum frames | **100** |
| Sampling method | **Stratified by mission type** — not random |
| Mission type strata | Road (≥40 frames), off-road/trail (≥30 frames), aerial/drone (≥30 frames) |
| Skew to avoid | Over-representing a single mission or lighting condition |

**How to sample:**
1. Run `scripts/backfill_captions.py` on at least 2 missions from each stratum.
2. For each stratum, take frames with `caption IS NOT NULL AND caption_skip_reason IS NULL`.
3. Sort by `t_sec` within each mission, then pick every Nth frame to spread coverage.
4. Export frame_ids + frame_paths + captions to `eval/eval_frames.jsonl` (one frame per line).

### Ground-truth annotation

| Requirement | Value |
|-------------|-------|
| Annotators | 2 independent annotators (or 1 annotator with ≥48h forced review gap) |
| Label | Relevant (1) / Not relevant (0) per (query, frame) pair |
| Disagreement resolution | **Majority vote** (2 annotators → deterministic; 1 annotator → self-review) |
| Format | `eval/ground_truth.jsonl` — one query per line (see Section 3) |

**Annotation protocol:**
1. Annotator A reviews all (query, frame) pairs independently and records relevance.
2. Annotator B (or same annotator after 48h gap) does the same, blind to A's labels.
3. For each pair: if both agree → use that label. If they disagree → third review decides.
4. Treat a frame as relevant only if the query is clearly answerable from the image alone
   (no context from adjacent frames, no GPS metadata).

---

## 3. Query Taxonomy

Minimum **5 queries per category**, **25 queries total** across 5 categories.

### Category 1: Vehicle Count
"How many X are visible?"

Examples:
- "five trucks in a row"
- "three vehicles on the road"
- "convoy of more than two vehicles"
- "single car on an empty road"
- "two motorcycles side by side"
- "no vehicles visible" ← also a negative control

### Category 2: Spatial Arrangement
"Where are the vehicles relative to each other or the road?"

Examples:
- "vehicles in a convoy formation"
- "truck overtaking a car"
- "vehicle parked on the side of the road"
- "vehicles at an intersection"
- "car turning left"
- "vehicles on opposite sides of the road"

### Category 3: Road / Scene Condition
"What is the road surface or environmental state?"

Examples:
- "wet road after rain"
- "mountain road with tight curves"
- "unpaved dirt track"
- "road with snow or ice"
- "construction zone with barriers"
- "clear dry highway"

### Category 4: Negative Controls
Queries that should match **zero frames** in the eval set.
Precision@5 must be 0.0 for these (any retrieved result is a false positive).

Examples (choose queries guaranteed absent from your footage):
- "underwater scene" (if all footage is terrestrial)
- "indoor parking garage" (if all footage is outdoor)
- "airport runway with aircraft"
- "crowd of more than 50 people"
- "ski slope with skiers"

### Category 5: General Scene Recall
Broad queries testing overall caption quality, not vehicle/road specifics.

Examples:
- "sunny day with clear visibility"
- "night or low-light scene"
- "dense urban environment"
- "rural or farmland scene"
- "dust or smoke reducing visibility"

---

## 4. Ground Truth File Format

`eval/ground_truth.jsonl` — one JSON object per line:

```json
{"query": "five trucks in a row", "category": "vehicle_count", "relevant_frame_ids": ["mission_1:42:12500", "mission_1:43:14200"]}
{"query": "wet road after rain", "category": "road_condition", "relevant_frame_ids": ["mission_2:17:8100"]}
{"query": "underwater scene", "category": "negative_control", "relevant_frame_ids": []}
```

`eval/eval_frames.jsonl` — one JSON object per line:

```json
{"frame_id": "mission_1:42:12500", "frame_path": "/data/frames/vid1/frame_042.jpg", "caption": "Five white trucks driving in convoy on a mountain road."}
{"frame_id": "mission_2:17:8100", "frame_path": "/data/frames/vid2/frame_017.jpg", "caption": "Wet asphalt road with puddles visible after rain."}
```

---

## 5. Metrics

### Primary metric: Precision@5

For each query q with ground-truth relevant set R(q):

```
P@5(q) = |{retrieved_frames[1..5]} ∩ R(q)| / 5
```

### Per-category aggregate

```
P@5(category) = mean(P@5(q) for q in category)
```

### Overall P@5

```
P@5(overall) = mean(P@5(q) for all q, excluding negative_control)
```

Negative controls are evaluated separately:
```
False Positive Rate = mean(|retrieved[1..5]| for negative_control queries) / 5
Target: 0.0 (zero false positives on negative controls)
```

### 95% Confidence Interval

Use the **Agresti-Coull interval** for each category P@5:

```
n = number of queries in category
k = number of queries where P@5(q) > 0   # "successes"
ñ = n + 4
p̃ = (k + 2) / ñ
margin = 1.96 * sqrt(p̃ * (1 - p̃) / ñ)
CI = [p̃ - margin, p̃ + margin]
```

The `scripts/eval_captions.py` script computes this automatically.

---

## 6. Baselines

Two systems evaluated side-by-side on the same query set:

| System | Description | Implementation |
|--------|-------------|----------------|
| **Semantic (Florence)** | Text → OpenCLIP embedding → Qdrant cosine search on `clip` vector | `scripts/eval_captions.py --method semantic` |
| **FTS baseline** | Keyword search on `caption` column in Postgres (`ILIKE %keyword%`) | `scripts/eval_captions.py --method fts` |

**Decision rule:**
If FTS baseline achieves P@5 ≥ 0.8 on general queries → Florence adds no measurable value
over keyword search at this eval set size. Expand eval set to 250+ frames before concluding.

---

## 7. Pass/Fail Gates

| Gate | Condition | Decision |
|------|-----------|----------|
| Phase 1 ships | Semantic P@5(overall) ≥ 0.8 | Ship Phase 1 to customer |
| Phase 2 triggered | Semantic P@5(vehicle_count) < 0.6 | Start Phase 2 (Qwen structured extraction) |
| FTS dominates | FTS P@5(overall) ≥ Semantic P@5(overall) | Investigate — Florence may not be adding value |
| Negative control failure | FPR(negative_control) > 0.0 | Investigate — captions hallucinating absent content |

Gates are **independent**: Phase 2 can be triggered even if Phase 1 ships (e.g. overall recall passes
but vehicle-count precision fails).

---

## 8. Confidence-Quality Calibration

After the eval run, compute the Pearson correlation between `caption_confidence` (stored in
the DB) and binary relevance across (query, frame) pairs:

```
r = corr(caption_confidence[frame], mean_relevance[frame])
```

where `mean_relevance[frame]` is the fraction of queries for which the frame is relevant.

**Interpretation:**
- r ≥ 0.5 → `caption_confidence` is a useful signal for active learning scoring
- r < 0.5 → consider replacing with caption length or perplexity (Design doc open question #1)

The `scripts/eval_captions.py --confidence-calibration` flag computes this.

---

## 9. Eval Runner

```bash
# Run both methods on ground truth, print comparison table + 95% CI
python scripts/eval_captions.py \
    --ground-truth eval/ground_truth.jsonl \
    --method both \
    --top-k 5 \
    --output eval/results_$(date +%Y%m%d).json

# Confidence-quality calibration
python scripts/eval_captions.py \
    --ground-truth eval/ground_truth.jsonl \
    --method semantic \
    --confidence-calibration
```

Output format: per-query P@5, per-category aggregate + 95% CI, overall, FPR on negatives,
comparison table semantic vs FTS, calibration r (if requested).

---

## 10. Checklist Before Running

- [ ] Florence deployed and at least one mission indexed (`caption IS NOT NULL` for ≥1 mission)
- [ ] `eval/eval_frames.jsonl` created with ≥100 frames stratified across ≥2 mission types
- [ ] `eval/ground_truth.jsonl` created with ≥25 queries across all 5 categories
- [ ] Ground truth annotated by 2 annotators (or 1 with 48h gap)
- [ ] Negative control queries verified absent from eval frame set
- [ ] `scripts/eval_captions.py` run with `--method both`
- [ ] 95% CI computed and recorded
- [ ] Calibration r computed for `caption_confidence`
- [ ] Results saved to `eval/results_YYYYMMDD.json`
- [ ] Pass/fail gate decision recorded in TODOS.md SV-06 section
