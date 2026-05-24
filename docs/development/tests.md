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

### Unit test layout

`tests/unit/` mirrors `src/selfsuvis/` where practical:

```text
tests/unit/
  app/
  models/
  pipeline/
    analysis/
    core/
    mapping/
    media/
    realtime/
    storage/
    training/
    vision/
    workflows/
  scripts/
  worker/
```

Reusable test helpers live in `tests/support/`. Keep `conftest.py` focused on pytest
fixture wiring; move reusable fake DB pools, mock rows, factories, and helper classes
into `tests/support/` when they are not fixture-specific.

Most unit tests should sit under the package area they cover. One exception remains on
purpose:

```text
tests/unit/test_multisite_enu.py
```

That file is intentionally kept flat because it exercises storage, worker, and app
behavior together. Cross-cutting tests can stay at `tests/unit/` root when forcing them
under one subsystem would make ownership less clear.

## Integration tests (Docker)
```bash
make test          # with GPU
make test-no-gpu   # without GPU (NVIDIA Container Toolkit not required)
```

Runs `test-dirs` first to create `data` and `cache_test` with correct ownership. Then starts api, worker, qdrant, and a tests container. Uses `docker/test/docker-compose.test.yml` with `ALLOWED_INDEX_PATHS=/app/tests/assets` and `MAX_UPLOAD_BYTES=150000`.

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

---
[← Licensing](../reference/licensing.md) | [README ↑](../../README.md)
