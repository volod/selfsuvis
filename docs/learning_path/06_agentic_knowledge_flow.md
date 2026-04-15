# Agentic Knowledge Flow

The local pipeline is not a bag of isolated model calls.
Later steps reuse earlier evidence through an accumulated context object called `VideoKnowledge`.
Understanding this object is the key to understanding why the pipeline produces different outputs depending on which steps ran and in what order.

---

## Core Idea

Earlier steps produce structured evidence.
That evidence is stored in `VideoKnowledge` via `add_*` methods.
Later steps query `context_for_frame(t_sec)` or `domain_hint()` to receive all prior knowledge relevant to a specific moment.

The main accumulator is implemented in:

- [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py) — `VideoKnowledge` class

---

## VideoKnowledge: Structure And Lifecycle

```python
class VideoKnowledge:
    video_name: str
    duration_sec: float
    frame_count: int

    # Domain-level knowledge (from Gemma, Step 3)
    scene_type: str          # dominant zero-shot category ("road", "urban", etc.)
    n_transitions: int       # how many scene changes Gemma detected
    n_clusters: int          # how many visually distinct segments
    gemma_mnn_dino: float    # mean nearest-neighbour DINOv3 distance

    # Per-frame evidence (keyed by t_sec)
    _captions:   Dict[float, str]       # Florence caption per frame (Step 4)
    _asr:        Dict[float, str]       # ASR speech text per timestamp (Step 5)
    _ocr:        Dict[float, str]       # OCR visible text per frame (Step 6)
    _depth:      Dict[float, Dict]      # depth statistics per frame (Step 7)
    _detections: Dict[float, List[str]] # detected object labels per frame (Step 8)

    # Temporal structure
    _segments: List[Dict]     # scene segments derived from caption Jaccard overlap

    # Cross-frame entity inventory (from Step 8)
    known_entities: List[str] # top-15 most frequent detected labels across video

    # Qwen rolling state (Step 24)
    _last_qwen: Dict[str, Any] # most recent Qwen JSON output
```

The object is created once per video at the start of the run and passed through every step.
Each step either deposits data into it or queries it.

---

## Deposit Methods: What Gets Stored Where

| Step | Method | Data deposited |
|------|---------|---------------|
| 3 (Gemma) | `add_gemma()` | `scene_type`, `n_transitions`, `n_clusters` |
| 4 (Florence) | `add_captions()` | `_captions`, `_segments` |
| 5 (ASR) | `add_asr()` | `_asr` |
| 6 (OCR) | `add_ocr()` | `_ocr` |
| 7 (Depth) | `add_depth()` | `_depth` |
| 8 (Detection) | `add_detections()` | `_detections`, `known_entities` |
| 24 (Qwen) | `update_qwen_state()` | `_last_qwen` |

All per-frame data is keyed by `t_sec` (timestamp in seconds from video start).
Retrieval uses nearest-timestamp lookup within a configurable window (default ±2 seconds).

---

## Query Methods: How Later Steps Read Evidence

### `domain_hint()` → str

Used by Florence (Step 4) to condition its captioning prompt.

Returns a pipe-separated summary like:
```
Dominant scene: road | Known objects: truck, car, person | Visual transitions: 3
```

If Gemma was skipped or failed, this string is empty.
Florence then operates without domain context and produces more generic captions.

### `context_for_frame(t_sec)` → str

Used by Qwen (Step 24) to build its per-frame reasoning prompt.

Assembles all evidence available near `t_sec` into a multi-line string:

```
[Prior scene description]: Two vehicles on a highway visible from above.
[Scene segment 2, 12.3s–18.7s]: Industrial site with large structures.
[Audio context]: "Sector seven, all clear"
[Visible text]: STOP 40
[Depth profile]: near_ratio=0.18  mean=0.65
[Detected objects]: truck, car, traffic-sign
[Prior frame state]: vehicles=2×truck; road=highway; condition=clear
```

Each line is sourced from a different step:
- `[Prior scene description]` → Florence caption, nearest frame within ±2 s.
- `[Scene segment ...]` → Jaccard-based segment analysis from Step 4.
- `[Audio context]` → ASR text within ±2 s window.
- `[Visible text]` → OCR text within ±1 s window.
- `[Depth profile]` → depth statistics, nearest frame within ±2 s.
- `[Detected objects]` → detection labels, nearest frame within ±2 s.
- `[Prior frame state]` → previous Qwen JSON result, if it was successfully parsed.

If any upstream step was skipped or failed, that line is simply absent.
Qwen never sees a `null` or empty placeholder — it sees no line at all.

---

## The Rolling State Pattern

The `_last_qwen` field enables temporal continuity in Qwen's reasoning.

Flow:
1. Qwen processes frame at `t=5.0 s` and returns a JSON with `vehicle_groups: [{type: truck, count: 2}]`.
2. `update_qwen_state()` stores this in `_last_qwen`.
3. Qwen processes frame at `t=7.0 s`. `context_for_frame(7.0)` includes:
   `[Prior frame state]: vehicles=2×truck; road=highway; condition=clear`
4. Qwen can now observe continuity: "the prior frame had 2 trucks; this frame has 3" is a valid observation.

What can go wrong:
- If the Qwen output at `t=5.0 s` was a JSON parse error, `update_qwen_state()` skips storing it.
  The prior state is either stale (from an earlier successful frame) or empty.
  This is the correct behavior: do not propagate a known-bad state.
- If Gemma misclassified the scene, the wrong `scene_type` is stored at initialization and propagates into every `domain_hint()` call for the entire video.
  This is the most impactful failure mode in the whole system.

---

## Why This Matters For Debugging

When the final synthesis or a Qwen output is wrong, trace backward through `VideoKnowledge`:

1. **Check the domain hint**: open `gemma_analysis.md`. Is `scene_type` correct?
   If not, every Florence caption and every Qwen system prompt was off-domain.

2. **Check the context string for the suspect frame**: reconstruct `context_for_frame(t_sec)`.
   Which lines are present? Which are absent?
   Absent lines = the upstream step failed or was skipped.

3. **Check the rolling state**: look at the Qwen output for `t_sec - Δt` (the previous frame).
   Is the prior state plausible? A wrong prior state propagates forward.

4. **Check timestamps**: if ASR or OCR lines appear at the wrong timestamps, the `t_sec` alignment is off.
   This is usually a video container timestamp issue (see Step 1).

---

## What Gets Accumulated: Full Inventory

Evidence accumulated by the end of Step 24, available in `VideoKnowledge`:

| Field | Source step | Time alignment | Max age |
|-------|------------|----------------|---------|
| `scene_type` | Gemma (Step 3) | Video-global | Entire video |
| `n_transitions` | Gemma (Step 3) | Video-global | Entire video |
| `known_entities` | Detection (Step 8) | Video-global | Entire video |
| `_captions[t]` | Florence (Step 4) | Per-frame | ±2 s lookup |
| `_segments[...]` | Florence (Step 4) | Temporal range | By segment |
| `_asr[t]` | ASR (Step 5) | Per-segment | ±2 s lookup |
| `_ocr[t]` | OCR (Step 6) | Per-frame | ±1 s lookup |
| `_depth[t]` | Depth (Step 7) | Per-frame | ±2 s lookup |
| `_detections[t]` | Detection (Step 8) | Per-frame | ±2 s lookup |
| `_last_qwen` | Qwen (Step 24) | Rolling | Previous frame only |

---

## Context Contamination: The Main Risk

Agentic context flow creates a compounding risk: a wrong observation injected early propagates into all later steps.

The most impactful contamination sources, ranked by severity:

1. **Wrong Gemma scene type** (Step 3): contaminates `domain_hint()` → every Florence caption is off-domain → every Qwen system prompt is wrong for the entire video.

2. **Wrong prior Qwen state** (Step 24): a Qwen hallucination at frame N becomes a "fact" in frame N+1's context. Errors can compound over many frames before a scene change resets the state.

3. **Wrong OCR text** (Step 6): OCR noise is injected into `[Visible text]` lines. Qwen may incorporate garbled characters as "evidence".

4. **Stale ASR** (Step 5): if the ASR window is too wide (±5 s instead of ±2 s), speech from one scene contaminates frames in an adjacent scene.

The audit step (Step 35) is designed to surface these contamination paths.
If you see a wrong synthesis output, start with the audit document.

---

## Human Reading Strategy

To understand the agentic part of the pipeline, read these in order:

1. [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py) — `VideoKnowledge` class and `context_for_frame()`
2. [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py) — how Florence, ASR, OCR, depth, and detection deposit into `VideoKnowledge`
3. [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py) — how Qwen queries `context_for_frame()` and updates rolling state
4. [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py) — the top-level orchestration and the `VideoKnowledge` lifecycle

## Questions To Ask While Reading

- What evidence is added at this step?
- What later step consumes it?
- Is the evidence time-aligned (per-frame) or video-global?
- What happens if this step is skipped — which downstream lines go silent?
- Can stale or wrong context propagate from this step onward?

## Related Docs

- [Perception core: Steps 1-8](02_perception_core_steps_01_08.md)
- [Tracking and mapping: Steps 21-27](04_tracking_mapping_steps_21_27.md)
- [Adaptation and audit: Steps 28-35](05_adaptation_eval_steps_28_35.md)
- [Pipeline architecture](../pipeline.md)
