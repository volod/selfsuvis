"""Backward-compatibility shim. Use pipeline.media.ffmpeg directly.

``os``, ``subprocess``, and ``settings`` are imported here so that existing
``patch("pipeline.ffmpeg_utils.*")`` test targets keep working.
"""
import os  # noqa: F401
import subprocess  # noqa: F401

from pipeline.core import settings  # noqa: F401
from pipeline.media.ffmpeg import extract_frames  # noqa: F401
