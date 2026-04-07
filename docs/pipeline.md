# Pipeline

This page describes the current indexing and demo flows, not historical design stages.

## Production indexing flow

1. A client calls one of the indexing endpoints.
2. The API validates input and writes a job to PostgreSQL.
3. The worker claims the job and runs `pipeline.workflows.indexer.VideoIndexer`.
4. The pipeline:
   - extracts frames
   - performs adaptive keep/skip decisions
   - embeds kept frames
   - optionally extracts and indexes tiles
   - captions frames with Florence and optional sidecar-backed enrichments
   - runs YOLO/SAM when enabled and writes a mission-scoped semantic environment graph
   - writes metadata to PostgreSQL and vectors to Qdrant
5. Optional spatial and reporting stages run after indexing:
   - pycolmap pose estimation
   - nerfstudio/mapper outputs
   - change detection
   - mission reports
   - active-learning tagging

## Useful command-line helpers

```bash
./scripts/precheck.sh path /path/to/video.mp4
./scripts/precheck_dir.sh /path/to/video_dir true true
./scripts/index_url.sh https://example.com/video.mp4 true
./scripts/index_dir.sh /path/to/video_dir true
./scripts/job_watch.sh <job_id>
```

## Typical API flow

Start the stack and initialize PostgreSQL:

```bash
make up
python scripts/migrate_postgres.py
```

Index a file:

```bash
curl -s -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/video.mp4" \
  -F "enable_tiles=true" \
  http://localhost:8000/index/video | python -m json.tool
```

Search by text:

```bash
curl -s -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"green field"}' \
  "http://localhost:8000/query/text?top_k=5&search_type=both" | python -m json.tool
```

Search by image:

```bash
curl -s -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/query.jpg" \
  -F "top_k=5" \
  -F "search_type=both" \
  -F "vector_space=clip" \
  http://localhost:8000/query/image | python -m json.tool
```

## Running the indexer directly in Python

```python
from pipeline.workflows.indexer import VideoIndexer

indexer = VideoIndexer(enable_tiles=True)
result = indexer.index_video("/path/to/video.mp4", "dev_test")
print(result)
```

This path is useful for local debugging when PostgreSQL/Qdrant are already reachable. The returned dict now includes a `semantic_graph` summary when YOLO SSG is enabled.

## Demo mode

`main.py --mode demo` runs the standalone demo pipeline defined by the current CLI in [`pipeline/workflows/cli_parser.py`](/home/vola/src/selfsuvis/pipeline/workflows/cli_parser.py) and the demo runner modules under [`pipeline/workflows/demo`](/home/vola/src/selfsuvis/pipeline/workflows/demo).

Common options:

```bash
python main.py --mode demo
python main.py --mode demo --no-qdrant --no-sfm --no-gsplat
python main.py --mode demo --asr --ocr --depth --detection
python main.py --mode demo --qwen --qwen-api-url http://localhost:8010/v1
python main.py --mode demo --gemma-api-url http://localhost:11434/v1
python main.py --mode demo --no-yolo --no-sam
```

The demo can combine local models and sidecar-backed models for Gemma, Qwen, Florence, and final reasoning.

## Pipeline outputs

Expect artifacts under `data/` such as:

- `data/videos/`
- `data/frames/`
- `data/tiles/`
- `data/reports/`
- `data/maps/`
- `data/checkpoints/`
- `data/models/`
- `data/gallery/`

Relevant semantic-graph artifacts:

- Production: `data/maps/<mission_id>/semantic_environment_graph.json`
- Demo: `<output_dir>/<video>/3d_map/semantic_environment_graph.json`
- Demo summary: `<output_dir>/<video>/3d_map/semantic_environment_graph.md`

For exact directories and defaults, see [`configuration.md`](./configuration.md).

---
[← Developer Guide](develop.md) | [Configuration →](configuration.md)
