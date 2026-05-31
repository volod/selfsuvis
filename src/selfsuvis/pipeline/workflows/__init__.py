"""High-level pipeline workflows and orchestration helpers.

Production exports (VideoIndexer, reporting) live here.
Local-pipeline exports (run_local, build_parser, etc.) are shims to ssv_vdp.
"""

from importlib import import_module

_EXPORTS = {
    # Production — stay in selfsuvis
    "VideoIndexer": (".indexer", "VideoIndexer"),
    "generate_summary_html": (".reporting", "generate_summary_html"),
    "write_mission_report": (".reporting", "write_mission_report"),
    "latlon_bbox": ("selfsuvis.pipeline.analysis.change_detection", "latlon_bbox"),
    # Local pipeline — shims to ssv_vdp
    "apply_local_env": ("ssv_vdp.local_env", "apply_local_env"),
    "build_parser": ("ssv_vdp.commands.parser", "build_parser"),
    "run_local": ("ssv_vdp", "run_local"),
    "run_file_mode": ("ssv_vdp.commands.runner", "run_file_mode"),
    "run_stream_mode": ("ssv_vdp.commands.runner", "run_stream_mode"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    package = __name__ if module_name.startswith(".") else None
    return getattr(import_module(module_name, package), attr_name)
