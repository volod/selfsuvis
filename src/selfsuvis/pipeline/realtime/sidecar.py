"""Shared HTTP helpers for realtime sidecar clients."""


from typing import Any

import httpx


class RealtimeSidecarClient:
    """Small shared base for HTTP-backed realtime sidecars."""

    def __init__(
        self,
        *,
        backend_name: str,
        base_url: str | None,
        timeout_sec: float,
    ) -> None:
        self._backend_name = str(backend_name or "").strip().lower()
        self._base_url = str(base_url or "").rstrip("/")
        self._timeout_sec = float(timeout_sec)

    @property
    def is_configured(self) -> bool:
        return bool(self._base_url)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Any | None:
        if not self.is_configured:
            return None
        async with httpx.AsyncClient(timeout=self._timeout_sec) as client:
            resp = await client.request(method, f"{self._base_url}{path}", json=payload)
            if allow_404 and resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def unwrap_dict_payload(data: Any, *, field: str | None = None) -> dict[str, Any] | None:
        if not isinstance(data, dict):
            return None
        if field and isinstance(data.get(field), dict):
            return data[field]
        return data

    async def stats(self) -> dict[str, Any]:
        if not self.is_configured:
            return {"configured": False, "backend": self._backend_name}
        data = await self._request_json("GET", "/stats")
        if isinstance(data, dict):
            return {"configured": True, "backend": self._backend_name, **data}
        return {"configured": True, "backend": self._backend_name}
