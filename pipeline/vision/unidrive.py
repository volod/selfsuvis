"""Thin HTTP client for UniDriveVLA-style multimodal driving analysis.

The upstream UniDriveVLA project is a full autonomous-driving stack with
understanding, perception, and planning experts. This adapter keeps the
integration lightweight for selfsuvis by treating UniDrive as an external
OpenAI-compatible vision endpoint and normalising its response into a stable
JSON shape that local and production workflows can consume.
"""
from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Optional

from PIL import Image

from pipeline.core import get_logger, settings

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are UniDriveVLA, a driving-oriented multimodal analyst with three "
    "specialised experts: understanding, perception, and planning. "
    "Respond only with valid JSON."
)

_USER_PROMPT = (
    "Analyse the frame and return ONLY a JSON object with this schema:\n"
    "{\n"
    '  "understanding": {\n'
    '    "scene_summary": "<short summary>",\n'
    '    "traffic_context": "<what is happening>",\n'
    '    "risk_level": "low|medium|high|unknown",\n'
    '    "key_agents": ["<agent>", "..."]\n'
    "  },\n"
    '  "perception": {\n'
    '    "objects": [\n'
    '      {"label": "<object>", "count": <integer>, "salience": "low|medium|high"}\n'
    "    ],\n"
    '    "drivable_area": "clear|partial|blocked|unknown",\n'
    '    "lane_structure": "<brief lane / road geometry summary>"\n'
    "  },\n"
    '  "planning": {\n'
    '    "recommended_action": "<short action>",\n'
    '    "trajectory_hint": "<path suggestion>",\n'
    '    "hazards": ["<hazard>", "..."]\n'
    "  },\n"
    '  "mixture_of_experts": {\n'
    '    "consensus_summary": "<combined expert answer>",\n'
    '    "expert_agreement": "high|medium|low|unknown",\n'
    '    "disagreement_points": ["<difference>", "..."]\n'
    "  }\n"
    "}\n"
    "Keep all fields concise. If a field is unknown, use a safe default."
)


def _encode_image_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    inner = lines[1:]
    if inner and inner[-1].strip() == "```":
        inner = inner[:-1]
    return "\n".join(inner).strip()


def _normalise_object_list(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            count = int(item.get("count", 1) or 1)
        except Exception:
            count = 1
        out.append({
            "label": str(item.get("label", "unknown") or "unknown"),
            "count": max(0, count),
            "salience": str(item.get("salience", "unknown") or "unknown"),
        })
    return out


def _normalise_string_list(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    return [str(v).strip() for v in raw if str(v).strip()]


def _parse_unidrive_response(raw_text: str) -> Dict[str, Any]:
    text = _strip_code_fences(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": raw_text[:500]}
    if not isinstance(data, dict):
        return {"parse_error": True, "raw": raw_text[:500]}

    understanding = data.get("understanding")
    if not isinstance(understanding, dict):
        understanding = {}
    perception = data.get("perception")
    if not isinstance(perception, dict):
        perception = {}
    planning = data.get("planning")
    if not isinstance(planning, dict):
        planning = {}
    moe = data.get("mixture_of_experts")
    if not isinstance(moe, dict):
        moe = {}

    return {
        "understanding": {
            "scene_summary": str(understanding.get("scene_summary", "") or ""),
            "traffic_context": str(understanding.get("traffic_context", "") or ""),
            "risk_level": str(understanding.get("risk_level", "unknown") or "unknown"),
            "key_agents": _normalise_string_list(understanding.get("key_agents")),
        },
        "perception": {
            "objects": _normalise_object_list(perception.get("objects")),
            "drivable_area": str(perception.get("drivable_area", "unknown") or "unknown"),
            "lane_structure": str(perception.get("lane_structure", "") or ""),
        },
        "planning": {
            "recommended_action": str(planning.get("recommended_action", "") or ""),
            "trajectory_hint": str(planning.get("trajectory_hint", "") or ""),
            "hazards": _normalise_string_list(planning.get("hazards")),
        },
        "mixture_of_experts": {
            "consensus_summary": str(moe.get("consensus_summary", "") or ""),
            "expert_agreement": str(moe.get("expert_agreement", "unknown") or "unknown"),
            "disagreement_points": _normalise_string_list(moe.get("disagreement_points")),
        },
    }


def _build_user_content(
    image: Image.Image,
    *,
    subtitle_text: Optional[str] = None,
    ocr_text: Optional[str] = None,
    extra_context: Optional[str] = None,
    domain_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    b64 = _encode_image_base64(image)
    content: List[Dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
        {"type": "text", "text": _USER_PROMPT},
    ]
    if domain_hint and domain_hint.strip():
        content.append({"type": "text", "text": f"\n[Domain hint]\n{domain_hint.strip()}"})
    if extra_context and extra_context.strip():
        content.append({"type": "text", "text": f"\n[Prior context]\n{extra_context.strip()}"})
    if subtitle_text and subtitle_text.strip():
        content.append({"type": "text", "text": f"\n[Audio context]\n{subtitle_text.strip()}"})
    if ocr_text and ocr_text.strip():
        content.append({"type": "text", "text": f"\n[Visible text]\n{ocr_text.strip()}"})
    return content


class UniDriveVLAModel:
    """OpenAI-compatible client for external UniDriveVLA inference."""

    def is_enabled(self) -> bool:
        return bool(settings.UNIDRIVE_ENABLED and settings.UNIDRIVE_API_URL)

    def analyze_frame(
        self,
        image: Image.Image,
        *,
        subtitle_text: Optional[str] = None,
        ocr_text: Optional[str] = None,
        extra_context: Optional[str] = None,
        domain_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_enabled():
            return {"service_unavailable": True, "reason": "UniDrive disabled"}

        try:
            import httpx
        except Exception as exc:
            return {"service_unavailable": True, "reason": f"httpx unavailable: {exc}"}

        endpoint = f"{settings.UNIDRIVE_API_URL.rstrip('/')}/chat/completions"
        payload = {
            "model": settings.UNIDRIVE_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_content(
                        image,
                        subtitle_text=subtitle_text,
                        ocr_text=ocr_text,
                        extra_context=extra_context,
                        domain_hint=domain_hint,
                    ),
                },
            ],
            "max_tokens": 400,
            "temperature": 0.1,
        }
        try:
            resp = httpx.post(endpoint, json=payload, timeout=float(settings.UNIDRIVE_TIMEOUT_SEC))
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.debug("UniDrive request failed: %s", exc)
            return {"service_unavailable": True, "reason": str(exc)}

        parsed = _parse_unidrive_response(str(raw))
        if parsed.get("parse_error"):
            logger.debug("UniDrive JSON parse failed: %s", parsed.get("raw", ""))
        return parsed

    def extract_batch(
        self,
        images: List[Image.Image],
        *,
        subtitle_texts: Optional[List[Optional[str]]] = None,
        ocr_texts: Optional[List[Optional[str]]] = None,
        extra_contexts: Optional[List[Optional[str]]] = None,
        domain_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        subtitle_texts = subtitle_texts or [None] * len(images)
        ocr_texts = ocr_texts or [None] * len(images)
        extra_contexts = extra_contexts or [None] * len(images)
        results: List[Dict[str, Any]] = []
        for image, subtitle, ocr, extra in zip(images, subtitle_texts, ocr_texts, extra_contexts):
            results.append(
                self.analyze_frame(
                    image,
                    subtitle_text=subtitle,
                    ocr_text=ocr,
                    extra_context=extra,
                    domain_hint=domain_hint,
                )
            )
        return results


__all__ = ["UniDriveVLAModel", "_parse_unidrive_response"]
