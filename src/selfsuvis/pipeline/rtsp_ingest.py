"""Backward-compatibility shim. Use pipeline.media.rtsp_ingest directly.

`socket` and `subprocess` are imported here so that existing
``patch("pipeline.rtsp_ingest.socket.getaddrinfo")`` test targets keep working.
Both shim and real module import the same stdlib objects, so patching through
either path reaches the same module-level reference.
"""
import socket  # noqa: F401
import subprocess  # noqa: F401

from selfsuvis.pipeline.media.rtsp_ingest import validate_rtsp_url, record_rtsp  # noqa: F401
