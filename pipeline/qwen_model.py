"""Backward-compatibility shim. Use pipeline.vision.qwen directly.

``_health_check_ollama`` and ``_health_check_vllm`` are imported here so that
existing ``patch("pipeline.qwen_model._health_check_*")`` test targets keep working.
"""
from pipeline.vision.qwen import (  # noqa: F401
    QwenModel,
    _health_check_ollama,
    _health_check_vllm,
)
