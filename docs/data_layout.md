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

Integration tests use `data_test/` and `cache_test/` instead of the main runtime directories.

Legacy SQLite files are no longer part of the runtime path. Schema state lives in PostgreSQL and is created by `scripts/migrate_postgres.py`.

---
[← Examples](examples.md) | [Performance →](performance.md)
