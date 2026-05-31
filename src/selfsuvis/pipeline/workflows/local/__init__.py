"""Compatibility shim — local pipeline moved to the ssv_vdp package.

Import from ssv_vdp directly:
    from ssv_vdp import run_local
"""

from ssv_vdp import run_local  # noqa: F401

__all__ = ["run_local"]
