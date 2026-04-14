"""Backward-compatibility shim. Use pipeline.storage.recent_index directly.

``time`` is imported here so that existing
``patch("pipeline.recent_index.time")`` test targets keep working.
"""
import time  # noqa: F401

from pipeline.storage.recent_index import RecentEmbeddingIndex  # noqa: F401
