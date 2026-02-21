# Tests

## Unit tests (no services required)

```bash
make test-unit
# or
pytest tests/unit/ -v
```

Unit tests cover:
- **utils**: clamp, stable_point_id, file_sha256, file_sha256_bytes, resolve_allowed_path, RateTimer
- **utils_path**: path validation with ALLOWED_INDEX_PATHS (inside/outside base, must_be_file/dir, traversal)
- **dedup**: PhashLRU (eviction, near-duplicate), dhash (when cv2 available)
- **heuristics**: downsample_gray, mean_intensity, histogram_diff, mean_abs_diff, edge_density, tile_std, tile_entropy (when cv2/skimage available)
- **downloader**: max_bytes limit (Content-Length and stream)
- **config**: validate_settings, _parse_allowed_paths
- **job_db**: update_job whitelist
- **ffmpeg**: extract_frames timeout

## Assets
Small test videos and a reference image live in:
```
./tests/assets/
```

## API tests (pytest)
These require a running API + worker + Qdrant.
```bash
pytest -q
```

Optional env vars:
- `API_URL` (default `http://localhost:8000`)
- `ASSETS_DIR` (default `./tests/assets`)
- `INDEX_DIR_PATH` (path to a directory for precheck_dir test)

## CLI test (end-to-end)
```bash
./scripts/test_cli.sh
```

## Docker Compose test
```bash
make test
```

Note: the compose test stack uses `./data_test` as its data volume to avoid polluting `./data`.
Model weights are cached under `./cache_test` and reused across test runs.
Containers run as the current host user via `UID`/`GID` to avoid root-owned files in `data_test` and `cache_test`.
`HOME` is set to `/app/cache` so libraries that default to `~/.cache` still write to a writable location.

## Docker Compose directory test
```bash
make test-dir
```
Runs the same stack as `make test`. Set `INDEX_DIR_PATH` to a path visible to the API container for directory-indexing tests.

## Integration tests (refactored functionality)

When API is running, these tests verify:
- **Health**: GET /health returns 200 with qdrant connected
- **Validation**: query_text (missing body, empty text, invalid search_type, invalid top_k), query_image (invalid search_type, invalid vector_space)
- **Errors**: index_video no file/path (400), job not found (404)
- **Security**: upload size limit (413), path outside ALLOWED_INDEX_PATHS (403)

## Lint
```bash
make lint
```
Runs `ruff check` and `ruff format --check`. Install ruff first (e.g. `pip install ruff` or add to dev requirements).

## Notes
- Tests are integration-style and will skip if `API_URL` is unreachable.
- For directory tests, set `INDEX_DIR_PATH` to a path visible to the API container.
- docker-compose.test.yml sets ALLOWED_INDEX_PATHS=/app/tests/assets and MAX_UPLOAD_BYTES=150000 for integration tests.
