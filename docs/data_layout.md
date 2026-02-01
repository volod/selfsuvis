# Data Layout

Default data root is `./data`.

```
./data/
  frames/        extracted frames by video_id
  tiles/         extracted tiles by video_id/segment_id
  videos/        stored mp4 copies by video_id
  qdrant/        Qdrant storage volume
  jobs.db        job queue state
  processed.db   dedup registry (hash-based)
```
