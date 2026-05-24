from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ChatRequest:
    model: str
    prompt: str
    system: str = "You are a careful reasoning assistant. Give concise final answers."
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 512


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.timeout = timeout

    def health(self) -> bool:
        try:
            response = httpx.get(
                self.base_url.removesuffix("/v1") + "/health",
                timeout=5.0,
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def wait_until_ready(self, timeout_s: float = 900.0, interval_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.health():
                return
            time.sleep(interval_s)
        raise TimeoutError(f"Endpoint did not become healthy within {timeout_s:.0f}s: {self.base_url}")

    def chat(self, request: ChatRequest) -> dict[str, Any]:
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_models(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=10.0)
        response.raise_for_status()
        return response.json()

