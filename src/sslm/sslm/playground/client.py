from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sslm.playground.constants import (
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TOP_P,
    DEFAULT_CLIENT_TIMEOUT_S,
    DEFAULT_OPENAI_API_KEY,
    ENDPOINT_READY_TIMEOUT_S,
    HEALTH_REQUEST_TIMEOUT_S,
    HTTP_OK,
    MODEL_LIST_TIMEOUT_S,
    READY_LOG_INTERVAL_S,
    READY_POLL_INTERVAL_S,
)


@dataclass(frozen=True)
class ChatRequest:
    model: str
    prompt: str
    system: str = DEFAULT_CHAT_SYSTEM_PROMPT
    temperature: float = DEFAULT_CHAT_TEMPERATURE
    top_p: float = DEFAULT_CHAT_TOP_P
    max_tokens: int = DEFAULT_CHAT_MAX_TOKENS


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = DEFAULT_OPENAI_API_KEY,
        timeout: float = DEFAULT_CLIENT_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.timeout = timeout

    def health(self) -> bool:
        # Poll /v1/models and require a non-empty model list.
        # /health returns 200 as soon as the HTTP server starts (before model
        # loads). /v1/models with an empty data[] also returns 200 in some forks
        # before the model is registered. Requiring at least one entry is the
        # safest proxy for "model is loaded and ready to serve requests."
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                timeout=HEALTH_REQUEST_TIMEOUT_S,
            )
            if response.status_code != HTTP_OK:
                return False
            data = response.json()
            return bool(data.get("data"))
        except (httpx.HTTPError, Exception):
            return False

    def wait_until_ready(
        self,
        timeout_s: float = ENDPOINT_READY_TIMEOUT_S,
        interval_s: float = READY_POLL_INTERVAL_S,
        is_alive: Callable[[], bool] | None = None,
    ) -> None:
        deadline = time.time() + timeout_s
        start = time.time()
        last_log = start - READY_LOG_INTERVAL_S  # log immediately on first check
        print(f"[sslm] Waiting for {self.base_url} to become ready (timeout {timeout_s:.0f}s) ...", flush=True)
        while time.time() < deadline:
            if is_alive is not None and not is_alive():
                elapsed = time.time() - start
                raise RuntimeError(
                    f"Container exited after {elapsed:.0f}s while waiting for {self.base_url} -- "
                    "check container logs for startup errors"
                )
            if self.health():
                elapsed = time.time() - start
                print(f"[sslm] Ready after {elapsed:.0f}s", flush=True)
                return
            now = time.time()
            if now - last_log >= READY_LOG_INTERVAL_S:
                elapsed = now - start
                remaining = deadline - now
                print(f"[sslm] Still waiting ... {elapsed:.0f}s elapsed, {remaining:.0f}s remaining", flush=True)
                last_log = now
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
        response = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=MODEL_LIST_TIMEOUT_S)
        response.raise_for_status()
        return response.json()
