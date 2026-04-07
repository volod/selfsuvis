"""Storage, indexing, and persistence helpers."""

from importlib import import_module

_EXPORTS = {
    "InMemoryStore": (".vector_store", "InMemoryStore"),
    "QdrantStore": (".qdrant", "QdrantStore"),
    "RecentEmbeddingIndex": (".recent_index", "RecentEmbeddingIndex"),
    "build_map_cache": (".map_cache", "build_map_cache"),
    "create_job": (".jobs", "create_job"),
    "fetch_and_claim_next_pending": (".jobs", "fetch_and_claim_next_pending"),
    "fetch_job": (".jobs", "fetch_job"),
    "list_global_maps": (".global_maps", "list_global_maps"),
    "update_job": (".jobs", "update_job"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    return getattr(import_module(module_name, __name__), attr_name)
