"""selfsuvis demo pipeline runner — thin compatibility shim.

All logic lives in :mod:`pipeline.demo`.  Import ``run_demo`` from there directly,
or use this module for backward-compatible imports.

    from pipeline.demo_runner import run_demo   # legacy
    from pipeline.demo.runner import run_demo   # preferred
"""

from pipeline.demo.runner import run_demo  # noqa: F401  re-exported for callers

__all__ = ["run_demo"]
