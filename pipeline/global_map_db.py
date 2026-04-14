"""Backward-compatibility shim. Use pipeline.storage.global_maps directly."""
from pipeline.storage.global_maps import (  # noqa: F401
    get_global_map_by_id,
    get_global_map_origin,
    get_global_map_splats,
    get_or_create_global_map,
    list_global_maps,
    list_mission_registrations,
    register_mission,
    update_global_map_splat,
    update_mission_splat_path,
)
