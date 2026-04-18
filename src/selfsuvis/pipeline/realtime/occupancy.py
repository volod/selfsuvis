"""Realtime occupancy/tile helpers."""


from typing import Any, Dict, Optional

def normalize_map_tile(tile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tile_key": str(tile["tile_key"]),
        "map_type": str(tile.get("map_type", "occupancy")).strip().lower(),
        "storage_path": str(tile["storage_path"]),
        "resolution_m": float(tile.get("resolution_m", 0.2)),
        "bounds": dict(tile.get("bounds") or {}),
        "stats": dict(tile.get("stats") or {}),
        "global_map_id": int(tile["global_map_id"]) if tile.get("global_map_id") is not None else None,
    }
