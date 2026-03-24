# Data Layout

Default data root is `./data`. Model cache is `./cache`.

```
./data/
  frames/                   extracted frames by video_id
    {video_id}/             sparse search keyframes (Pass B)
    {video_id}_sfm/         dense SfM frames at SFM_FPS (Pass A)
  tiles/                    extracted tiles by video_id/segment_id
  videos/                   stored mp4 copies by video_id
  maps/                     3DGS output
    {mission_id}/splat.ply            single-scene splatfacto output
    {mission_id}/scene-{N}/splat.ply  multi-scene chunked output
    {mission_id}/colmap/              pycolmap database + reconstruction
  reports/                  auto-generated HTML mission summaries
    {mission_id}/summary.html
  qdrant/                   Qdrant storage volume
  postgres/                 PostgreSQL 16 data volume (jobs, missions, frames, ...)

./cache/        model downloads (torch, open_clip)
  torch/
  open_clip/
```

`make up` creates `data/` and `cache/` with correct ownership. Integration tests use `./data_test` and `./cache_test` instead.

**Note:** `jobs.db` and `processed.db` (legacy SQLite) are no longer used in v1. All job and frame state is stored in PostgreSQL. Run `python scripts/migrate_postgres.py` after first `make up` to create the schema.

---
[← Examples](examples.md) | [Performance →](performance.md)
