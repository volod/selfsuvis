"""Backward-compatibility shim. Use pipeline.storage.processed directly."""
from pipeline.storage.processed import (  # noqa: F401
    aget_by_hash,
    aget_by_size,
    aget_by_url,
    aupsert,
    get_by_hash,
    get_by_size,
    get_by_url,
    init_db,
    upsert,
)
