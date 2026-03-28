"""Thin HTTP client for Qwen2.5-VL-7B structured scene extraction via vLLM/ollama sidecar.

Phase 2 of the selfsuvis scene captioning system. This module provides
`QwenModel`, which calls an OpenAI-compatible vision endpoint and returns
structured `frame_facts_json` dicts for vehicle/road scene understanding.

The contract: `extract_frame_facts` never returns None.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from pipeline.config import settings
from pipeline.logging_utils import get_logger

logger = get_logger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────

_VEHICLE_LABELS: frozenset = frozenset(
    {
        "vehicle",
        "truck",
        "car",
        "bus",
        "convoy",
        "emergency vehicle",
        "armoured vehicle",
        "tank",
        "motorcycle",
        "van",
    }
)

_QWEN_SYSTEM_PROMPT = (
    "You are a precise outdoor-scene analyst specialised in military and "
    "logistics convoy imagery. Extract structured facts from the image. "
    "Respond ONLY with valid JSON — no markdown, no extra text."
)

_QWEN_USER_PROMPT = (
    "Analyse the image and return ONLY a JSON object with these keys:\n"
    '{\n'
    '  "vehicle_groups": [\n'
    '    {"type": "truck|car|bus|motorcycle|emergency|military|van|other",\n'
    '     "count": <integer>,\n'
    '     "color": "<dominant color or unknown>",\n'
    '     "position": "<front|centre|rear|left|right|scattered>"}\n'
    '  ],\n'
    '  "road_surface": "asphalt|concrete|gravel|dirt|unknown",\n'
    '  "road_condition": "clear|wet|snow|ice|debris|unknown",\n'
    '  "scene_summary": "<one sentence describing the scene>"\n'
    '}\n'
    "If no vehicles are visible, return an empty vehicle_groups list."
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_user_content(
    image: Image.Image,
    subtitle_text: Optional[str] = None,
    ocr_text: Optional[str] = None,
) -> list:
    """Build the user-message content list, optionally enriched with audio/OCR context.

    Injects subtitle transcription and OCR text as additional context blocks
    so Qwen can use spoken/written words to disambiguate scene content.
    """
    b64 = _encode_image_base64(image)
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
        {"type": "text", "text": _QWEN_USER_PROMPT},
    ]
    if subtitle_text and subtitle_text.strip():
        content.append({
            "type": "text",
            "text": f"\n[Audio context at this moment]: {subtitle_text.strip()}",
        })
    if ocr_text and ocr_text.strip():
        content.append({
            "type": "text",
            "text": f"\n[Text visible in frame]: {ocr_text.strip()}",
        })
    return content


def _encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG string (data URI body only)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _health_check_vllm(base_url: str, timeout: int) -> bool:
    """Return True if the vLLM health endpoint responds with HTTP 200."""
    try:
        import httpx

        # Strip trailing /v1 if present to get the server root.
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        resp = httpx.get(f"{root}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _health_check_ollama(base_url: str, timeout: int) -> bool:
    """Return True if the ollama /api/tags endpoint responds with HTTP 200."""
    try:
        import httpx

        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        resp = httpx.get(f"{root}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _parse_qwen_response(raw_text: str) -> Dict[str, Any]:
    """Parse raw Qwen response text into a structured dict.

    Strips markdown code fences, parses JSON, validates top-level structure,
    and returns a normalised dict. On any error returns a parse_error dict.
    """
    text = raw_text.strip()

    # Strip markdown code fences: ```json...``` or ```...```
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json or ```) and last line (```)
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": raw_text[:500]}

    if not isinstance(data, dict):
        return {"parse_error": True, "raw": raw_text[:500]}

    # Normalise: ensure expected keys are present with safe defaults
    normalised: Dict[str, Any] = {
        "vehicle_groups": data.get("vehicle_groups", []),
        "road_surface": data.get("road_surface", "unknown"),
        "road_condition": data.get("road_condition", "unknown"),
        "scene_summary": data.get("scene_summary", ""),
    }

    # Coerce vehicle_groups to a list
    if not isinstance(normalised["vehicle_groups"], list):
        normalised["vehicle_groups"] = []

    return normalised


# ── Main class ────────────────────────────────────────────────────────────────


class QwenModel:
    """HTTP client for Qwen2.5-VL structured scene extraction.

    Parameters
    ----------
    clip_prescreen_fn:
        Optional callable that accepts a PIL.Image and returns True if the
        image likely contains a vehicle (above threshold). When provided,
        this avoids loading a second CLIP model inside QwenModel — the
        caller (VideoIndexer) passes a closure that reuses the already-loaded
        OpenCLIPEmbedder.
    """

    def __init__(self, clip_prescreen_fn: Optional[Callable[[Image.Image], bool]] = None):
        self._clip_prescreen_fn = clip_prescreen_fn
        self._tagger = None  # lazily initialised OpenCLIPTagger
        self._healthy: Optional[bool] = None  # cached health state

    # ── Public interface ──────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        """Return True when QWEN_API_URL is configured (non-empty)."""
        return bool(settings.QWEN_API_URL)

    def is_healthy(self) -> bool:
        """Return True when the configured sidecar is reachable.

        Result is cached after the first call to avoid repeated network round
        trips during batch processing.
        """
        if not self.is_enabled():
            return False
        if self._healthy is None:
            self._check_health()
        return bool(self._healthy)

    def extract_frame_facts(
        self,
        image: Image.Image,
        subtitle_text: Optional[str] = None,
        ocr_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structured scene facts from a single frame image.

        Parameters
        ----------
        image:
            The frame to analyse.
        subtitle_text:
            Optional ASR transcription near this frame's timestamp (from the
            audio track).  When provided, injected into the Qwen prompt as
            ``[Audio context at this moment]`` to help disambiguate scene content.
        ocr_text:
            Optional OCR-extracted text visible in the frame.  When provided,
            injected as ``[Text visible in frame]`` in the Qwen prompt.

        Returns a dict — never None — with one of the following shapes:

        - Disabled:             ``{"disabled": True}``
        - Service unavailable:  ``{"service_unavailable": True}``
        - CLIP filtered:        ``{"clip_filtered": True, "reason": "below_vehicle_threshold"}``
        - Success:              ``{"vehicle_groups": [...], "road_surface": ..., ...}``
        - Timeout:              ``{"timeout": True, "timeout_sec": N}``
        - Parse error:          ``{"parse_error": True, "raw": "..."}``
        """
        if not self.is_enabled():
            return {"disabled": True}

        if not self.is_healthy():
            return {"service_unavailable": True}

        # CLIP pre-screen: skip frames that are unlikely to contain vehicles.
        if self._clip_prescreen_fn is not None:
            try:
                if not self._clip_prescreen_fn(image):
                    return {"clip_filtered": True, "reason": "below_vehicle_threshold"}
            except Exception:
                logger.debug("CLIP prescreen raised an exception; proceeding without filter", exc_info=True)
        elif settings.QWEN_CLIP_THRESHOLD > 0:
            tagger = self._lazy_tagger()
            if tagger is not None:
                try:
                    result = tagger.describe_image(image, top_k=1)
                    if result:
                        top_label, top_score = result[0]
                        if top_label.lower() not in _VEHICLE_LABELS or top_score < settings.QWEN_CLIP_THRESHOLD:
                            return {"clip_filtered": True, "reason": "below_vehicle_threshold"}
                    else:
                        return {"clip_filtered": True, "reason": "below_vehicle_threshold"}
                except Exception:
                    logger.debug("Tagger prescreen failed; proceeding without filter", exc_info=True)

        # Build the user message content, including optional audio/OCR context.
        user_content = _build_user_content(image, subtitle_text, ocr_text)

        # Call the Qwen sidecar via OpenAI-compatible API.
        try:
            from openai import OpenAI, APITimeoutError

            client = OpenAI(
                api_key="EMPTY",  # vLLM/ollama do not require a real key
                base_url=settings.QWEN_API_URL,
                timeout=settings.QWEN_TIMEOUT_SEC,
            )

            response = client.chat.completions.create(
                model=settings.QWEN_MODEL,
                messages=[
                    {"role": "system", "content": _QWEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=512,
                temperature=0.0,
            )

            raw_text = response.choices[0].message.content or ""
            return _parse_qwen_response(raw_text)

        except Exception as exc:
            # Check for timeout specifically (openai >= 1.0 raises APITimeoutError)
            exc_type = type(exc).__name__
            if exc_type == "APITimeoutError" or "timeout" in exc_type.lower():
                logger.warning("Qwen timeout after %ds", settings.QWEN_TIMEOUT_SEC)
                return {"timeout": True, "timeout_sec": settings.QWEN_TIMEOUT_SEC}

            # Try to import and check the proper class if available
            try:
                from openai import APITimeoutError as _APITimeoutError
                if isinstance(exc, _APITimeoutError):
                    logger.warning("Qwen timeout after %ds", settings.QWEN_TIMEOUT_SEC)
                    return {"timeout": True, "timeout_sec": settings.QWEN_TIMEOUT_SEC}
            except ImportError:
                pass

            logger.warning("Qwen extraction failed: %s", exc, exc_info=True)
            return {"service_unavailable": True}

    def extract_batch(
        self,
        images: List[Image.Image],
        subtitle_texts: Optional[List[Optional[str]]] = None,
        ocr_texts: Optional[List[Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract frame facts for a list of images with optional per-image context.

        ``subtitle_texts`` and ``ocr_texts`` must be the same length as
        ``images`` when provided.  Calls ``extract_frame_facts`` sequentially
        (the sidecar is typically single-GPU and does not benefit from
        concurrent requests).
        """
        n = len(images)
        sub = subtitle_texts if subtitle_texts and len(subtitle_texts) == n else [None] * n
        ocr = ocr_texts if ocr_texts and len(ocr_texts) == n else [None] * n
        return [self.extract_frame_facts(img, s, o) for img, s, o in zip(images, sub, ocr)]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_health(self) -> None:
        """Check sidecar health and cache the result in self._healthy."""
        backend = settings.QWEN_BACKEND.lower()
        timeout = min(settings.QWEN_TIMEOUT_SEC, 10)  # quick health check
        # Auto-detect ollama from the default port (11434) when backend is not
        # explicitly set to "ollama" — the vllm health endpoint (/health) returns
        # 404 on ollama servers, which only expose /api/tags.
        if backend != "ollama" and ":11434" in settings.QWEN_API_URL:
            backend = "ollama"
        if backend == "ollama":
            self._healthy = _health_check_ollama(settings.QWEN_API_URL, timeout)
        else:
            # default: vllm
            self._healthy = _health_check_vllm(settings.QWEN_API_URL, timeout)
            # Fallback: if vllm check fails, try ollama (covers non-standard ports)
            if not self._healthy:
                self._healthy = _health_check_ollama(settings.QWEN_API_URL, timeout)
                if self._healthy:
                    backend = "ollama"

        if self._healthy:
            logger.info("Qwen sidecar healthy (backend=%s url=%s)", backend, settings.QWEN_API_URL)
        else:
            logger.warning(
                "Qwen sidecar unreachable (backend=%s url=%s); Phase 2 will be skipped",
                backend,
                settings.QWEN_API_URL,
            )

    def _lazy_tagger(self):
        """Lazily create and cache an OpenCLIPTagger for vehicle pre-screening."""
        if self._tagger is not None:
            return self._tagger
        try:
            from pipeline.vision_models import OpenCLIPTagger

            self._tagger = OpenCLIPTagger()
            return self._tagger
        except Exception:
            logger.debug("Could not load OpenCLIPTagger for Qwen pre-screen", exc_info=True)
            return None
