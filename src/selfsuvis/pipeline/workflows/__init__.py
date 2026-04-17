"""High-level pipeline workflows and orchestration helpers."""

from importlib import import_module

_EXPORTS = {
    "apply_local_env": (".local_env", "apply_local_env"),
    "VideoIndexer": (".indexer", "VideoIndexer"),
    "build_parser": (".cli_parser", "build_parser"),
    "generate_summary_html": (".reporting", "generate_summary_html"),
    "latlon_bbox": ("selfsuvis.pipeline.analysis.change_detection", "latlon_bbox"),
    "run_local": (".local", "run_local"),
    "run_file_mode": (".cli_runner", "run_file_mode"),
    "run_stream_mode": (".cli_runner", "run_stream_mode"),
    "write_mission_report": (".reporting", "write_mission_report"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    package = __name__ if module_name.startswith(".") else None
    return getattr(import_module(module_name, package), attr_name)
