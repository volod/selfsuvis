# Tests

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

## Docker Compose directory test
```bash
make test-dir
```

## Notes
- Tests are integration-style and will skip if `API_URL` is unreachable.
- For directory tests, set `INDEX_DIR_PATH` to a path visible to the API container.
