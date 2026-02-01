# Examples

## End-to-end test
1. `docker compose up --build`
2. Upload a sample flight video in the UI
3. Run text query: `green field`
4. Run image query with a reference crop

## CLI flow (index + query)
```bash
./scripts/sample_requests.sh /path/to/video.mp4 /path/to/image.jpg
```

## Directory precheck + enqueue
```bash
./scripts/precheck_dir.sh /path/to/video_dir true true
```
