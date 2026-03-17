# Tests

## Unit tests (no Docker)
```bash
make test-unit
# or
pytest tests/unit/ -v
```

Uses `.venv` if present. Skips cv2-dependent tests if numpy/opencv mismatch:
```bash
make test-unit-no-cv2
```

Unit tests cover: utils, utils_path, dedup, heuristics, downloader, config, job_db, ffmpeg, net_utils, frame_extractor, processed_db, etc.

## Integration tests (Docker)
```bash
make test          # with GPU
make test-no-gpu   # without GPU (NVIDIA Container Toolkit not required)
```

Runs `test-dirs` first to create `data_test` and `cache_test` with correct ownership. Then starts api, worker, qdrant, and a tests container. Uses `docker/docker-compose.test.yml` with `ALLOWED_INDEX_PATHS=/app/tests/assets` and `MAX_UPLOAD_BYTES=150000`.

Directory-indexing tests:
```bash
make test-dir
```
Same as `make test`; `INDEX_DIR_PATH` is set for dir tests.

## Assets
```
./tests/assets/   # small test videos and reference image
```

## Integration test coverage
- Health, index (video/url/dir), precheck, precheck_dir, jobs, query (image/text)
- Validation, errors (400, 403, 404, 413), job_id validation

## Lint
```bash
make lint
```
Runs `ruff check` and `ruff format --check`. Install ruff first (e.g. `pip install ruff`).

## CLI test (end-to-end)
```bash
./scripts/test_cli.sh
```
Requires `make up` running; indexes test assets and runs queries.

---
[← Licensing](licensing.md) | [README ↑](../README.md)
