# Data Layout

By default the repository writes runtime data to `./data` and model caches to `./cache`.

```text
data/
  videos/         stored video inputs
  frames/         extracted keyframes and dense SfM frames
  tiles/          tile crops used for retrieval
  audio/          temporary audio extracted for ASR
  reports/        mission summary HTML output
  maps/           pycolmap and splat outputs
  checkpoints/    SSL and supervised DINO checkpoints
  models/         exported edge models
  gallery/        edge gallery NPZ files
  qdrant/         Qdrant volume data
  postgres/       PostgreSQL volume data

cache/
  torch/
  open_clip/
  huggingface/
```

Common map layouts:

```text
data/maps/{mission_id}/splat.ply
data/maps/{mission_id}/scene-0/splat.ply
data/maps/{mission_id}/colmap/
```

Integration tests use `data/` and `cache_test/` directories.

Legacy SQLite files are no longer part of the runtime path. Schema state lives in PostgreSQL and is created by `src/selfsuvis/scripts/migrate_postgres.py`.

---

## Sensor sidecar files

Each sidecar sits beside the video with the same basename. `scripts/setup_local_full.sh` generates synthetic test sidecars automatically when you run it against a test video.

```
data/videos/mission.mp4
data/videos/mission.iq              # Step  9 — RF/SDR IQ (float32 interleaved)
data/videos/mission.thermal.mp4     # Step 10 — FLIR LWIR radiometric video
data/videos/mission.multispectral/  # Step 11 — per-band GeoTIFF directory
data/videos/mission.events.raw      # Step 12 — Prophesee event stream
data/videos/mission.lidar.pcd       # Step 13 — LiDAR point cloud (PCD/MCAP)
data/videos/mission.radar.bin       # Step 14 — radar ADC IQ (TI DCA1000)
data/videos/mission.adsb.jsonl      # Step 15 — ADS-B aircraft log (dump1090)
data/videos/mission.gnssr.bin       # Step 15 — GNSS-R IQ capture
data/videos/mission.imu.jsonl       # Step 16 — IMU samples (200 Hz)
data/videos/mission.baro.jsonl      # Step 16 — barometer (5 Hz)
data/videos/mission.wind.jsonl      # Step 16 — anemometer (1 Hz)
data/videos/mission.env.jsonl       # Step 17 — atmospheric (temp/humidity/wind)
data/videos/mission.gas.jsonl       # Step 18 — gas/radiation (CO2, VOC, dose rate)
data/videos/mission.audio.wav       # Step 19 — acoustic (48 kHz WAV)
```

To regenerate sidecars for a different video:

```bash
cp /path/to/my_mission.mp4 data/videos/
bash scripts/setup_local_full.sh --sensor-data-only
```

---

## Output artifacts

For each video `<name>.mp4` the pipeline writes to `data/maps/<mission_id>/` and the directories below.

| File / Dir | Pipeline step | Contents |
|---|---|---|
| `gemma_analysis.md` | 3 | Gemma scene change detection, clustering, CLIP+DINOv3 comparison |
| `gemma_captions.md` | 3 | Per-frame natural-language descriptions |
| `scene_captions.md` | 4 | Florence-2 captions per keyframe |
| `asr_subtitles.md` | 5 | Whisper ASR segments + per-frame subtitle coverage |
| `multimodal_features.md` | 6–8 | OCR text, depth percentiles, detections, world model |
| `detailed_captions.md` | 24 | Qwen VLM structured per-frame analysis |
| `unidrive_analysis.md` | 25 | UniDriveVLA understanding / perception / planning / MoE |
| `multi_model_comparison.md` | 32 | Gemma vs Qwen vs UniDriveVLA expert-agreement summary |
| `finetune_stats.md` | 28 | SSL fine-tuning loss curve + config |
| `finetuned_search.md` | 31 | Queries re-run with fine-tuned model |
| `comparison.md` | 32 | Side-by-side model comparison + video-to-text description |
| `edge_models/` | 30 | ONNX model + frame gallery for edge deployment |
| `checkpoints/` | 28 | Fine-tuned `.pt` checkpoints |
| `3d_map/sparse_map.ply` | 27 | Sparse SfM or PCA point cloud |
| `3d_map/gaussian_splat.ply` | 27 | 3D Gaussian Splat (see [gaussian_splat.md](gaussian_splat.md)) |
| `3d_map/semantic_environment_graph.json` | 27 | YOLO SSG scene graph |
| `gemma_tracking_results.json` | 22 | Gemma-directed tracking per frame |
| `gemma_tracking/frame_*_tracked.jpg` | 22 | Annotated frames with RF-DETR track boxes |
| `final_stats.md` | 35 | Per-video and aggregate statistics |

---
[← Examples](examples.md) | [Performance →](performance.md)
