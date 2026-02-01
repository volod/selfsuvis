# Troubleshooting

## Qdrant not reachable
- Ensure `docker compose ps` shows `qdrant` up
- Check `QDRANT_HOST` and `QDRANT_PORT`

## No GPU detected
- Ensure NVIDIA Container Toolkit is installed
- Verify `nvidia-smi` on host
- Ensure Docker has GPU access

## Low recall / too many duplicates
- Increase `EMBED_DRIFT_THRESH` and `HIST_THRESH`
- Reduce `MAX_GAP_SEC`
- Tighten `DEDUP_COS_SIM_THRESH`

## Indexing too slow
- Increase `SAMPLE_FPS_MIN` and reduce `SAMPLE_FPS_MAX`
- Reduce `MAX_TILES_PER_SEGMENT`
- Disable DINO (`MODEL_NAME=openclip`)

## Empty UI thumbnails
- Ensure `./data` volume is mounted into UI container
- Validate file paths are accessible from UI
