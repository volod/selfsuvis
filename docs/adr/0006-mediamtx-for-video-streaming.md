# ADR-0006: MediaMTX for Video Streaming Ingestion

Date: 2026-03-23
Status: Accepted
Deciders: @vola

---

## Context

The system needs to ingest video from multiple sources. File upload and HTTPS link
ingestion are handled by the existing FastAPI pipeline. Live video streaming (RTSP/RTMP
from cameras, drones, or other sources) requires a dedicated streaming server.

## Decision

Use **MediaMTX** (`bluenviron/mediamtx` Docker image) as the streaming server.

MediaMTX is a production-grade, zero-dependency streaming server (Go binary) supporting
RTSP, RTMP, WebRTC, HLS, and SRT. It is the reference implementation cited by the
upstream specification.

Docker Compose service added: `mediamtx` (new container). Integrates with the existing
FastAPI ingest pipeline via frame capture → HTTP POST to the job queue API.

**v1 scope — file re-streaming only:**
- MediaMTX is configured to re-stream existing local video files via RTSP
- This validates the streaming code path without requiring a live camera
- Use case: `ffmpeg -re -i mission.mp4 -f rtsp rtsp://localhost:8554/test`

**v1.5 — live camera ingestion:**
- RTSP/RTMP from real cameras (drone FPV, IP cameras, rovers)
- Same pipeline; MediaMTX handles the protocol translation

## Consequences

**Good:**
- Zero Python dependency — Go binary in Docker, no pip packages
- Supports every relevant streaming protocol in one service
- File re-streaming enables full end-to-end streaming test without hardware
- Architecture is the same for v1 (file) and v1.5 (live) — no rework needed

**Bad / Tradeoffs:**
- Adds a container to Docker Compose even in v1 where it only re-streams files —
  acceptable given the zero-overhead nature of the Go binary
- Live camera integration (v1.5) requires network configuration and camera authentication
  that is out of scope for v1
