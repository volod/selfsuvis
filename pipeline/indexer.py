"""Backward-compatibility shim. Use pipeline.workflows.indexer directly.

``settings`` is imported here so that existing
``patch.object(indexer_mod, "settings", ...)`` test targets keep working.
"""
from pipeline.core import settings  # noqa: F401
from pipeline.workflows.indexer import VideoIndexer  # noqa: F401
