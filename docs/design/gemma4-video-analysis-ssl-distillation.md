# Gemma 4 for Video Analysis — Capabilities, SSL Fine-Tuning, and Maximum Distillation

*Analysis date: 2026-04-04*

---

## 1. What Video Analysis Gemma 4 Can Do

Gemma 4 (`google/gemma-4-it-2b`, `gemma-4-it-4b`) is a multimodal model composed of
a SigLIP-based vision encoder (~300 M params) fused into a Gemma language backbone
(2 B or 4 B params). Unlike Florence-2 or DINOv3, Gemma 4 understands images
*in language space* — it produces semantically grounded embeddings and can reason
across multiple images in a single forward pass.

### 1.1 Analyses we can already do (current `GemmaEmbedder`)

| Analysis | How | Current status |
|---|---|---|
| Per-frame scene description | `encode_images` + image prompt | Working (`gemma_analysis.md`) |
| Cross-modal retrieval (text→frame) | text embedding vs frame embedding cosine sim | Working |
| Scene change detection | consecutive frame embedding distance | Working |
| Zero-shot scene classification | cosine vs label embeddings | Working |
| Temporal video embedding | mean-pool of frame embeddings | Working |
| Scene clustering | k-means on frame embeddings | Working |

### 1.2 New analyses Gemma 4 enables that we do not yet use

#### Multi-image comparison (highest priority)
Gemma 4's processor accepts multiple images in one conversation. This allows
direct frame-to-frame comparison without embedding subtraction:

```
Prompt: [image_A] [image_B]
"What has changed between these two frames?
 Focus on: vehicle positions, road state, new objects."
```
This gives richer change descriptions than cosine distance alone (knows *what*
changed, not just *that* something changed). Useful for `scene_captions.md`
transition rows and for the change detection pipeline.

#### Temporal sequence reasoning (N frames in one pass)
Feed a window of 4–8 consecutive frames to one Gemma call:

```
Prompt: [f_0] [f_1] [f_2] [f_3]
"Describe what is happening in this sequence.
 Is the vehicle slowing down, turning, or maintaining speed?"
```
30 fps video with slowly changing scenes is exactly the case where a window of
frames conveys more than a single frame — motion direction, convoy gaps, road
curvature ahead.

#### Structured event extraction replacing Qwen
Currently `pipeline/qwen_model.py` sends one frame at a time to an external
Ollama/vLLM sidecar for structured JSON output (`vehicle_groups`, `road_surface`,
etc.). Gemma 4 can do the same structured extraction locally:

```json
{
  "vehicle_groups": [...],
  "road_surface": "...",
  "road_condition": "...",
  "heading": "straight|curve_left|curve_right|intersection",
  "scene_summary": "..."
}
```
Advantage: no sidecar required, runs in the same Python process, can be
context-enriched with previous frame's output.

#### Audio-grounded visual reasoning
Gemma 4 supports audio tokens. Combined with Whisper ASR output (already
available as `subtitle_map`), a single Gemma 4 call can reason about both
the visual scene and spoken audio context. Currently Qwen receives ASR as
injected text — Gemma 4 can process native audio embeddings.

#### Anomaly / novelty detection
Ask Gemma 4 to rate how unusual a frame is relative to the mission's
established baseline:

```
Prompt: [baseline_frame_A] [baseline_frame_B] [query_frame]
"Is the query frame consistent with the mission baseline?
 Describe anything unusual."
```
This complements the embedding-distance `al_tag` (active learning score) with
a language explanation for why a frame was flagged.

#### GPS + visual geo-description
Given frame embedding + GPS coordinates already stored in the database:

```
Prompt: [frame] "This frame was captured at lat=..., lon=...
 Describe the terrain type and likely road category."
```
Enhances the robot pose API (`POST /query/pose`) with human-readable location
context alongside the vector search result.

#### Caption improvement for near-duplicate frames (30 fps problem)
Instead of captioning every frame independently (Florence-2 repeats itself),
use multi-frame Gemma 4 calls for *segment-level* captions:

1. Group consecutive frames into segments using existing `_analyze_caption_sequence`.
2. For each segment boundary, send `[last_frame_of_seg_N] [first_frame_of_seg_N+1]`
   with prompt "What changed?".
3. Only call Gemma for the representative frame of each segment for the full description.

Result: fewer redundant captions, richer transition descriptions — directly
addresses the core problem raised.

### 1.3 Capability matrix vs current pipeline models

| Capability | Florence-2 | Qwen (sidecar) | DINOv3 | Gemma 4 |
|---|---|---|---|---|
| Per-frame caption | ✓ | ✓ | — | ✓ |
| Structured JSON extraction | — | ✓ | — | ✓ |
| Multi-frame reasoning | — | — | — | **✓** |
| Audio context | — | injected text | — | **native** |
| Embedding for retrieval | — | — | ✓ | ✓ |
| Zero-shot classification | ✓ | — | — | ✓ |
| Anomaly explanation | — | — | — | **✓** |
| Runs without sidecar | ✓ | ✗ | ✓ | ✓ |
| VRAM at FP16 (2B) | 1.5 GB | 10–12 GB | 0.3 GB | **~4 GB** |

---

## 2. Self-Supervised Fine-Tuning of Gemma 4

### 2.1 Architecture breakdown for SSL

Gemma 4 is not a pure vision encoder — it is a VLM. The parts relevant to SSL:

```
Input image
    │
    ▼
SigLIP Vision Encoder (ViT-L/16, ~300 M params, frozen in inference)
    │  image patch tokens (sequence of visual tokens)
    ▼
Projection layer (linear, ~10 M params)
    │  mapped into LM token space
    ▼
Gemma LM backbone (2 B or 4 B params, autoregressive transformer)
    │  processes interleaved visual + text tokens
    ▼
Final hidden states → mean-pool → L2-normalise → embedding (dim=2560 for 2B)
```

The SSL target in `ssl_finetune.py` is the DINOv3 ViT-B backbone (pure image
encoder). Applying the same approach to Gemma 4 requires choosing *which part*
to fine-tune.

### 2.2 Three feasible SSL strategies

#### Strategy A — Vision encoder only (recommended, feasible on 16 GB VRAM)

Fine-tune the top N transformer blocks of the SigLIP vision encoder using the
existing NT-Xent temporal-pairs loss. The LM backbone is **fully frozen**.

```
Trainable:  top 4 of 24 ViT-L blocks + patch projection  (~60 M params)
Frozen:     rest of ViT-L encoder + entire LM backbone
Loss:       NT-Xent on L2-normalised CLS/mean-pool tokens
Pairs:      temporal pairs (existing TemporalPairDataset)
VRAM:       ~6 GB (encoder forward + gradient on top blocks)
```

This is directly analogous to the current DINOv3 SSL pipeline. The vision
encoder tokens are extracted *before* the LM, so the LM never runs during
training — fast and memory-efficient.

**Integration point:** `GemmaEmbedder` already pools `hidden_states[-1]`.
For SSL we intercept at the vision encoder output instead.

#### Strategy B — LoRA on full model (feasible on 24 GB VRAM)

Apply LoRA adapters (rank 8–16) to the attention Q/K/V matrices of both the
vision encoder and the top LM layers:

```
Trainable:  LoRA adapters only  (~2–8 M params regardless of model size)
Frozen:     all base weights
Loss:       NT-Xent OR masked image modelling (predict masked patches)
VRAM:       ~10 GB with gradient checkpointing
Library:    HuggingFace PEFT
```

This adapts the full multimodal representation, not just the vision encoder.
The LM backbone's scene understanding is refined for this mission's vocabulary.
Better for the retrieval + captioning use case than Strategy A, but requires
more VRAM and slower training.

#### Strategy C — Embedding head only (feasible on 8 GB VRAM, weakest)

Freeze everything, train only a small linear projection head on top of the
frozen Gemma embedding:

```
Trainable:  1 linear layer (2560 → 768)  (~2 M params)
Frozen:     entire model
Loss:       NT-Xent or supervised contrastive (if labels available)
VRAM:       ~4 GB (inference only, no backprop through model)
```

Weakest adaptation (the base representation doesn't change) but the fastest
and usable on any consumer GPU.

### 2.3 Recommended SSL training config for Strategy A

```python
GemmaSSLConfig(
    model_id        = "google/gemma-4-it-2b",
    freeze_lm       = True,          # entire LM backbone frozen
    freeze_vision_blocks = 20,       # freeze first 20 of 24 ViT-L blocks
    epochs          = 10,
    batch_size      = 8,             # one image at a time through Gemma4 processor
    lr              = 5e-6,          # lower than DINO SSL (larger model)
    temperature     = 0.07,
    approach        = "temporal",    # existing TemporalPairDataset
    device          = "cuda",
    use_bf16        = True,
    grad_checkpoint = True,          # essential for 300M vision encoder
)
```

Expected training throughput on RTX 3090 (24 GB): ~15 frames/sec, ~200 pairs/sec.
10 epochs on 1000 mission frames: ~8 min.

### 2.4 What SSL gives Gemma 4 that it doesn't have out of the box

Gemma 4 is pre-trained on web images — diverse but generic. Mission video
from drones/rovers contains:
- Specific terrain types (desert gravel, forest tracks)
- Specific vehicle classes (armoured vehicles, specific convoy trucks)
- Lighting/altitude conditions not well-represented in web data
- 30 fps temporal structure — consecutive frames are almost identical

SSL on mission frames teaches the vision encoder that consecutive frames
of the *same scene* should have similar representations, tightening the
embedding manifold for this domain without requiring any labels.

---

## 3. Maximum Distillation — Smallest Viable Video Analysis Model

### 3.1 Current distillation chain

```
DINOv3 ViT-B/14  (86 M params, 768-dim)   [SSL fine-tuned teacher]
        │
        │  RKD-DA + KoLeo + cosine anchor (pipeline/distill.py)
        ▼
DINOv2 ViT-S/14  (22 M params, 384-dim)   [student, ~4× compression]
```

### 3.2 Extended chain: Gemma 4 as root teacher

Gemma 4's vision encoder produces language-grounded embeddings — far richer than
pure vision models. Using it as the root teacher in a multi-stage distillation
chain transfers semantic understanding into much smaller models.

```
┌─────────────────────────────────────────────────────┐
│  Stage 0: Gemma 4 vision encoder (SSL fine-tuned)   │
│  SigLIP ViT-L/16 · ~300 M params · dim=1152          │
│  VRAM: ~4 GB inference                              │
└────────────────────────┬────────────────────────────┘
                         │ RKD-DA + KoLeo
                         │ (new: GemmaDistillConfig)
                         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: DINOv2 ViT-S/14 (existing pipeline)       │
│  22 M params · dim=384 · 13.6× compression          │
│  VRAM: ~0.2 GB  CPU-runnable                        │
└────────────────────────┬────────────────────────────┘
                         │ RKD-DA only (simpler loss, smaller batch)
                         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: EfficientViT-S1 or MobileViT-XS           │
│  5–7 M params · dim=384 · ~3× additional compression│
│  Runs on CPU/RPi/edge, 10 ms/frame on laptop        │
└─────────────────────────────────────────────────────┘
```

Total compression from Gemma 4 root to Stage 2: **~43–60×**

### 3.3 Target student models for Stage 2

| Model | Params | dim | CPU latency | Notes |
|---|---|---|---|---|
| `microsoft/EfficientViT-S1` | 6.6 M | 384 | ~8 ms/frame | Best quality/speed ratio |
| `apple/MobileViT-XS` | 2.3 M | 384 | ~5 ms/frame | Smallest viable |
| `facebook/dinov2_vits14` | 22 M | 384 | ~15 ms/frame | Already in pipeline |
| `timm/efficientvit_b0` | 3.4 M | 192 | ~3 ms/frame | Extreme edge, lower quality |
| `timm/fastvit_t8` | 3.6 M | 256 | ~4 ms/frame | Apple FastViT, very fast |

Recommended Stage 2 target: **`microsoft/EfficientViT-S1`** — well-supported
in `timm`, 384-dim matches existing Qdrant collection dimensionality if we
keep `MODEL_NAME=dinov3`, and runs on CPU at under 10 ms/frame.

### 3.4 Multi-stage distillation: what changes in `pipeline/distill.py`

The existing `KnowledgeDistiller` only supports a pure PyTorch forward-pass
teacher (a nn.Module that takes a `(B, 3, 224, 224)` tensor and returns `(B, D)`).

To use Gemma 4 as teacher, we need a thin wrapper:

```python
class GemmaVisionTeacher(nn.Module):
    """Wraps GemmaEmbedder.encode_images() as a standard nn.Module.
    
    Extracts only the SigLIP vision encoder hidden states — does not run
    the LM backbone. Frozen during distillation.
    """
    def __init__(self, embedder: GemmaEmbedder):
        super().__init__()
        self._embedder = embedder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) standard tensor
        # Convert to PIL batch, run Gemma vision encoder, return pooled embedding
        pils = [transforms.ToPILImage()(img) for img in x.cpu()]
        embs = self._embedder.encode_images(pils)   # (B, dim) numpy
        return torch.from_numpy(embs).to(x.device)
```

For Stage 1→Stage 2 distillation, the existing `KnowledgeDistiller` works
unchanged with a different `student_model` pointing to EfficientViT-S1.

### 3.5 Loss function adjustments for Gemma → small model

The Gemma 4 vision encoder dim is 1152 (SigLIP ViT-L). The Stage 1 student
(ViT-S/14) has dim 384. The existing projection head `Linear(384→768)` needs
to change to `Linear(384→1152)` — only a config change.

For Stage 2 (ViT-S/14 teacher → EfficientViT-S1 student, both 384-dim),
the projection head can be removed entirely and RKD-DA is applied directly
in the shared 384-dim space.

### 3.6 Distillation loss recommendations per stage

**Stage 0→1 (Gemma ViT-L → ViT-S/14):**

The Gemma embeddings carry language semantics. RKD-DA preserves pairwise
neighbourhood topology, which is what matters for retrieval. Add a
**caption-anchored cosine loss** — for frames with Gemma captions already
generated, the student's embedding should be close to the CLIP text embedding
of that caption:

```
L_total = λ_D·L_RKD_dist + λ_A·L_RKD_angle + λ_kd·L_cosine + λ_koleo·L_KoLeo
        + λ_cap·L_caption_anchor          ← NEW: cosine(student, CLIP_text(caption))
```

Recommended λ_cap = 0.5. Requires CLIP text encoder running in parallel
(already loaded in the pipeline as `OpenCLIPEmbedder`).

**Stage 1→2 (ViT-S/14 → EfficientViT-S1):**

Both are 384-dim. Use RKD-D only (drop RKD-A for speed — angle loss is O(B³)):

```
L_total = λ_D·L_RKD_dist + λ_koleo·L_KoLeo
```

### 3.7 Expected quality degradation across the chain

Measured as Recall@1 (nearest-neighbour overlap between teacher and student),
estimated from literature:

| Chain step | R@1 | Absolute drop |
|---|---|---|
| Gemma 4 vision encoder (reference) | 1.000 | — |
| → ViT-S/14 Stage 1 distilled | ~0.92 | –0.08 |
| → EfficientViT-S1 Stage 2 distilled | ~0.85 | –0.07 |
| Direct ViT-S/14 pretrained (no distillation) | ~0.78 | –0.22 |

The two-stage distillation recovers ~7 percentage points of retrieval quality
versus a directly pretrained student.

### 3.8 ONNX export and edge deployment

The existing `pipeline/edge_inference.py` and `build_gallery()` already handle
ONNX export of DINOv3. EfficientViT-S1 and MobileViT-XS are both
`torch.onnx.export`-compatible. Expected ONNX model sizes:

| Model | PyTorch .pt | ONNX FP32 | ONNX INT8 (quantised) |
|---|---|---|---|
| DINOv2 ViT-S/14 | 88 MB | 88 MB | 24 MB |
| EfficientViT-S1 | 26 MB | 26 MB | 7 MB |
| MobileViT-XS | 9 MB | 9 MB | 3 MB |

INT8 quantisation (using `onnxruntime.quantization`) typically costs 1–3%
Recall@1 — acceptable for edge deployment.

---

## 4. Implementation Roadmap

Transition note: this document was written during the historical demo phase.
That workflow has since been promoted into the canonical local full-analysis and
learning pipeline. Where this section references the "demo pipeline" or
`pipeline/demo_runner.py`, read it as the current local orchestration layer in
`pipeline/workflows/local/runner.py`.

### Phase 1 — Gemma 4 multi-frame analysis in local pipeline (no new training)

1. Add `step_gemma_segment_captions()` to the local workflow runner:
   - Re-uses `_analyze_caption_sequence()` to get segment boundaries
   - For each segment boundary pair, calls `GemmaEmbedder` with two frames
     and a diff prompt
   - Writes `gemma_segment_captions.md`
2. Add structured extraction prompt to `GemmaEmbedder` (returns JSON like Qwen)
   so `step_qwen_captioning` can fall back to local Gemma when no Ollama sidecar

**Effort:** ~2 days. No training required. Immediately usable.

### Phase 2 — SSL fine-tuning of Gemma 4 vision encoder

1. Add `GemmaSSLFinetuner` to `pipeline/ssl_finetune.py`:
   - Wraps SigLIP vision encoder extraction
   - Reuses existing `TemporalPairDataset`
   - NT-Xent loss identical to current DINOv3 SSL
2. Add `write_gemma_ssl_stats_md()` to the local workflow reporting layer

**Effort:** ~3 days. Requires ≥16 GB VRAM for training.

### Phase 3 — Multi-stage distillation with Gemma 4 teacher

1. Add `GemmaVisionTeacher` wrapper class to `pipeline/distill.py`
2. Extend `DistillConfig` with `stage: int` and `lambda_caption_anchor: float`
3. Add EfficientViT-S1 loader to `models/dino_model.py` or new `models/efficientvit_model.py`
4. Add `step_distill_stage2()` to the local workflow runner
5. ONNX export for EfficientViT-S1 in `pipeline/edge_inference.py`

**Effort:** ~5 days. Requires ≥24 GB VRAM for Stage 0→1 distillation.

---

## 5. Summary

| Question | Answer |
|---|---|
| What can Gemma 4 analyse in video? | Multi-frame comparison, temporal sequence reasoning, structured scene extraction, audio-grounded reasoning, anomaly explanation, GPS geo-description — richer than any current single-model in the pipeline |
| Can we SSL fine-tune Gemma 4? | Yes — Strategy A (vision encoder only, top 4 blocks) is feasible on 16 GB VRAM in ~8 min per mission; Strategy B (LoRA full model) on 24 GB; both reuse existing `TemporalPairDataset` |
| What is the smallest viable distilled model? | EfficientViT-S1 (6.6 M params, 8 ms/frame CPU) via two-stage distillation: Gemma 4 vision encoder → ViT-S/14 → EfficientViT-S1; ~60× compression from teacher, R@1 ≈ 0.85 |
| Does distillation require labels? | No — RKD-DA is fully unsupervised, driven only by the teacher's embedding structure |
| What changes in existing code? | `distill.py`: new `GemmaVisionTeacher` wrapper + `lambda_caption_anchor`; `ssl_finetune.py`: new `GemmaSSLFinetuner`; local workflow runner/reporting: new steps for segment captions + Stage 2 distill |
