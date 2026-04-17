"""Backward-compatibility shim. Use pipeline.mapping.mapper directly.

``requests`` is imported here so that existing
``patch("pipeline.mapper.requests.post")`` test targets keep working.
"""
import requests  # noqa: F401

from selfsuvis.pipeline.mapping.mapper import (  # noqa: F401
    _call_icp_fuse,
    _train_scene,
    run_mapper,
)
