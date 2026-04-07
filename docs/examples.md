# Examples

## End-to-end test
1. `make up`
2. Open http://localhost:8501
3. Upload a video or provide a URL
4. Run text query (e.g. "green field")
5. Run image query with a reference crop

## CLI flow
Use the retained helpers from [`docs/helpers.md`](./helpers.md) to precheck and index local paths or URLs from the command line.

## Directory precheck + enqueue
```bash
./scripts/precheck_dir.sh /path/to/video_dir true true
```

---
[← Architecture](architecture.md) | [Data layout →](data_layout.md)
