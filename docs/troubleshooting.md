# Troubleshooting

## Docker permission denied
- Add your user to the docker group: `sudo usermod -aG docker $USER`
- Log out and back in, or run `newgrp docker`

## Qdrant not reachable
- Ensure `docker compose -f docker/docker-compose.yml ps` shows `qdrant` up
- Check `QDRANT_HOST` and `QDRANT_PORT` (default `qdrant:6333`)

## No GPU / "could not select device driver with capabilities: [[gpu]]"
- Install NVIDIA Container Toolkit: `sudo ./scripts/install_nvidia_docker.sh`
- Or run without GPU: `make test-no-gpu` for tests; for `make up`, use a compose override that removes GPU from api/worker

## Root-owned data or cache / Qdrant "Permission denied" on Snapshots
- Ensure `make up` ran (it runs `data-dirs` first)
- Fix existing data: `make fix-data` or `sudo chown -R $(id -u):$(id -g) data cache`
- Then run `make up` again

## Unable to open database file (tests)
- `make test` runs `test-dirs` to create `data_test` and `cache_test` with correct ownership
- If it still fails: `sudo chown -R $(id -u):$(id -g) data_test cache_test`

## Low recall / too many duplicates
- Increase `EMBED_DRIFT_THRESH` and `HIST_THRESH`
- Reduce `MAX_GAP_SEC`
- Tighten `DEDUP_COS_SIM_THRESH`

## Indexing too slow
- Increase `SAMPLE_FPS_MIN` and reduce `SAMPLE_FPS_MAX`
- Reduce `MAX_TILES_PER_SEGMENT`
- Disable DINO (`MODEL_NAME=openclip`)

## Empty UI thumbnails
- Ensure `./data` volume is mounted into UI container (default in `make up`)
- Validate file paths are accessible from UI

---
[← Performance](performance.md) | [Licensing →](licensing.md)
