# Helpers

## Pre-download weights for offline use
```bash
python scripts/prepare_models.py
DOWNLOAD_DINO=true DINO_MODEL=dinov2_vitb14 python scripts/prepare_models.py
```

## Sample API flow (index + query)
```bash
./scripts/sample_requests.sh /path/to/video.mp4 /path/to/image.jpg
```

## Batch index a directory
```bash
./scripts/index_dir.sh /path/to/video_dir true
```

## Index a URL
```bash
./scripts/index_url.sh https://example.com/video.mp4 true
```

## Watch a job
```bash
./scripts/job_watch.sh <job_id>
```

## Precheck (avoid double load)
```bash
./scripts/precheck.sh file /path/to/video.mp4
./scripts/precheck.sh path /path/to/video.mp4
./scripts/precheck.sh url https://example.com/video.mp4
```

## Precheck directory (optionally enqueue new)
```bash
./scripts/precheck_dir.sh /path/to/video_dir
./scripts/precheck_dir.sh /path/to/video_dir true true
```

## Clean frames/tiles cache
```bash
./scripts/clean_data.sh ./data
```

## Reset Qdrant collection
```bash
./scripts/reset_qdrant.sh
```

## List processed registry
```bash
python scripts/list_processed.py
```

## Hash a video
```bash
python scripts/hash_video.py /path/to/video.mp4
```
