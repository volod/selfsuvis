# Data Layout

Default data root is `./data`. Model cache is `./cache`.

```
./data/
  frames/        extracted frames by video_id
  tiles/         extracted tiles by video_id/segment_id
  videos/        stored mp4 copies by video_id
  qdrant/        Qdrant storage volume
  jobs.db        job queue state
  processed.db   dedup registry (hash-based)

./cache/        model downloads (torch, open_clip)
  torch/
  open_clip/
```

`make up` creates `data/` and `cache/` with correct ownership. Integration tests use `./data_test` and `./cache_test` instead.

---
[← Examples](examples.md) | [Performance →](performance.md)
