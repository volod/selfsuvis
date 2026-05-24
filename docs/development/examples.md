# Examples

## End-to-end test
1. `make up`
2. Open http://localhost:8501
3. Upload a video or provide a URL
4. Run text query (e.g. "green field")
5. Run image query with a reference crop

## CLI flow
Use the direct API examples in [`docs/helpers.md`](./helpers.md) to precheck and index local paths or URLs from the command line.

## Directory precheck + enqueue
```bash
curl -s \
  -F "path=/path/to/video_dir" \
  -F "enqueue=true" \
  -F "enable_tiles=true" \
  http://localhost:8000/index/precheck_dir | python -m json.tool
```

---

## Self-supervised pretraining — Denoising Autoencoder (DAE)

The DAE pretext task trains a convolutional encoder-decoder to reconstruct clean
frames from corrupted inputs (Gaussian noise + patch masking).  After training,
reconstruction MSE serves as a per-frame anomaly score: frames outside the
training distribution cannot be reconstructed well.

### Train a DAE on mission frames

```python
from selfsuvis.pipeline.training.dae import DAEFinetuneConfig, run_dae_finetune

cfg = DAEFinetuneConfig(
    frames_dir="/data/missions/my_mission/frames",
    output_dir="/data/missions/my_mission/ssl",
    epochs=15,
    batch_size=32,
    device="cuda",        # or "cpu" for small datasets
    corruption_mode="both",   # Gaussian noise + patch masking (default)
    noise_std=0.2,
    mask_frac=0.15,
)
checkpoint_path = run_dae_finetune(cfg)
# Writes:
#   /data/missions/my_mission/ssl/dae_best.pt      -- full model (best MSE epoch)
#   /data/missions/my_mission/ssl/dae_encoder.pt   -- encoder weights only
print(f"best checkpoint: {checkpoint_path}")
```

### Corruption modes

```python
from selfsuvis.pipeline.training.dae import DenoisingAutoencoder, corrupt
import torch

model = DenoisingAutoencoder(latent_ch=256, image_size=224)
x = torch.randn(1, 3, 224, 224)

corrupted_gaussian = corrupt(x, mode="gaussian", noise_std=0.2)
corrupted_masking  = corrupt(x, mode="masking",  patch_size=16, mask_frac=0.15)
corrupted_both     = corrupt(x, mode="both")    # default

recon = model(corrupted_both)   # shape: (1, 3, 224, 224)
mse   = ((recon - x) ** 2).mean().item()
```

### Extract bottleneck features

```python
import torch
from selfsuvis.pipeline.training.dae import DenoisingAutoencoder

model = DenoisingAutoencoder(latent_ch=256)
model.load_state_dict(torch.load("dae_best.pt", map_location="cpu")["model"])
model.eval()

x = torch.randn(4, 3, 224, 224)          # batch of 4 frames
features = model.encode(x)               # shape: (4, 256, 14, 14)
pooled   = features.mean(dim=(2, 3))     # (4, 256) -- global-average-pooled
```

---

## Anomaly scoring

### Score a batch of frames

```python
from selfsuvis.pipeline.analysis.anomaly import (
    load_dae_scorer,
    score_frames_anomaly,
    tag_anomalous_frames,
)

# load_dae_scorer returns None if the checkpoint does not exist;
# all downstream functions treat None as "disabled" and return zeros/normal tags.
scorer = load_dae_scorer(
    checkpoint_path="/data/missions/my_mission/ssl/dae_best.pt",
    device="cuda",
)

frame_paths = [
    "/data/missions/my_mission/frames/f0001.jpg",
    "/data/missions/my_mission/frames/f0002.jpg",
    # ...
]
raw_scores = score_frames_anomaly(frame_paths, scorer, batch_size=32)
# raw_scores: list[float] of per-frame MSE values

norm_scores, tags = tag_anomalous_frames(raw_scores)
# norm_scores: min-max normalised to [0, 1] within the batch
# tags: list of "high_anomaly" | "anomaly" | "normal"

for path, score, tag in zip(frame_paths, norm_scores, tags):
    if tag != "normal":
        print(f"{tag}: {path}  (score={score:.3f})")
```

### Custom thresholds

```python
norm_scores, tags = tag_anomalous_frames(
    raw_scores,
    anomaly_percentile=85.0,       # default 90.0
    high_anomaly_percentile=95.0,  # default 97.0
)
```

---

## Active learning with reconstruction scores

The updated `assign_al_tags` and `compute_al_score` functions accept an optional
`reconstruction_scores` argument.  When provided, the DAE reconstruction error
is blended into the active-learning score alongside DINOv3 distance and Florence
caption confidence.

### Two-signal formula (original, no DAE checkpoint)

```python
from selfsuvis.pipeline.analysis.active_learning import assign_al_tags

al_scores, al_tags = assign_al_tags(
    dino_dists=dino_dists,               # list[float], 0-1
    caption_confidences=caption_confs,   # list[float], 0-1
    top_k=50,
)
# al_score = 0.6 * dino_dist + 0.4 * (1 - caption_confidence)
```

### Three-signal formula (with DAE reconstruction scores)

```python
al_scores, al_tags = assign_al_tags(
    dino_dists=dino_dists,
    caption_confidences=caption_confs,
    reconstruction_scores=norm_scores,   # from tag_anomalous_frames
    top_k=50,
)
# al_score = 0.45 * dino_dist + 0.30 * (1 - caption_confidence)
#           + 0.25 * reconstruction_score
```

### Four-signal formula (with RSSM temporal surprise + DAE)

```python
al_scores, al_tags = assign_al_tags(
    dino_dists=dino_dists,
    caption_confidences=caption_confs,
    rssm_surprises=rssm_surprises,       # from DreamerV3 world model
    reconstruction_scores=norm_scores,
    top_k=50,
)
# al_score = 0.25 * dino_dist + 0.20 * (1 - caption_confidence)
#           + 0.30 * rssm_surprise + 0.25 * reconstruction_score
```

### Per-frame score (single frame)

```python
from selfsuvis.pipeline.analysis.active_learning import compute_al_score

score = compute_al_score(
    dino_dist=0.72,
    caption_confidence=0.45,
    reconstruction_score=0.81,
)
print(f"AL score: {score:.4f}")
```

---

## Full DAE + anomaly + AL workflow

```python
from selfsuvis.pipeline.training.dae import DAEFinetuneConfig, run_dae_finetune
from selfsuvis.pipeline.analysis.anomaly import (
    load_dae_scorer,
    score_frames_anomaly,
    tag_anomalous_frames,
)
from selfsuvis.pipeline.analysis.active_learning import assign_al_tags

frames_dir = "/data/missions/my_mission/frames"
ssl_dir    = "/data/missions/my_mission/ssl"

# 1. Train DAE
ckpt = run_dae_finetune(
    DAEFinetuneConfig(frames_dir=frames_dir, output_dir=ssl_dir, epochs=15, device="cuda")
)

# 2. Score frames for anomalies
frame_paths = sorted(glob.glob(f"{frames_dir}/*.jpg"))
scorer      = load_dae_scorer(checkpoint_path=ckpt, device="cuda")
raw_scores  = score_frames_anomaly(frame_paths, scorer)
norm_scores, anomaly_tags = tag_anomalous_frames(raw_scores)

# 3. Active-learning tagging (assumes dino_dists and caption_confs are available)
al_scores, al_tags = assign_al_tags(
    dino_dists=dino_dists,
    caption_confidences=caption_confs,
    reconstruction_scores=norm_scores,
)

# 4. Print frames queued for annotation
for path, al_tag, anom_tag in zip(frame_paths, al_tags, anomaly_tags):
    if al_tag == "needs_annotation" or anom_tag != "normal":
        print(f"[{al_tag:18s}] [{anom_tag:12s}] {path}")
```

---
[← Architecture](../reference/architecture.md) | [Data layout →](../reference/data_layout.md)
