# ADR-0006: MediaMTX as the Streaming Edge for Live Video Ingest

Date: 2026-03-23  
Status: Accepted

## Context

The product handles both file-based indexing and live stream ingestion. Live RTSP/RTMP /
WebRTC handling should be delegated to a dedicated streaming service rather than embedded
directly into the FastAPI app.

## Decision

Use MediaMTX as the streaming edge and control-plane companion for live video ingest.

Current integration:
- `docker/mediamtx.yml`
- `docker/docker-compose.yml`
- realtime / ingest paths in `src/selfsuvis/app/routers/realtime.py`
- stream recording / validation helpers in `src/selfsuvis/pipeline/media/rtsp_ingest.py`

MediaMTX is the protocol-facing service; the main app and worker consume the resulting
streams or recordings through the normal pipeline.

## Consequences

Positive:
- Streaming concerns stay outside the API process
- Supports the protocol mix the project needs without custom Python transport code
- Same service works for local test restreaming and live deployments

Trade-offs:
- Adds another operational service to the compose stack
- Live ingest still depends on network and camera-side configuration quality
