"""Main orchestration for the demo pipeline.

Contains: model/store init, per-video orchestrator, and the top-level
``run_demo`` entry point.  Step helpers are imported from sibling modules.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pipeline.core import settings
from pipeline.media import extract_frames
from pipeline.storage import InMemoryStore
from pipeline.mapping.viewer import view_npz, _HAS_MPL
from models.openclip_model import OpenCLIPEmbedder

try:
    from models.dino_model import DINOEmbedder
    _HAS_DINO = True
except Exception:
    _HAS_DINO = False

try:
    from models.gemma_model import GemmaEmbedder
    _HAS_GEMMA = True
except Exception:
    _HAS_GEMMA = False

from ._common import (
    _log,
    _banner,
    _step,
    _Timer,
    _configure_logging,
    _configure_warnings,
    _TEXT_PROMPTS,
    VideoKnowledge,
)

# ── Constants ──────────────────────────────────────────────────────────────────

_TOTAL_STEPS = 20
_VIDEO_EXTS  = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Phase 3 SSL gate: skip distillation / ONNX / search comparison when the
# SSL fine-tune best loss is ≥ this threshold (indicates a failed run).
_SSL_GATE_MAX_LOSS = 10.0


# ── Device resolution ──────────────────────────────────────────────────────────

def _resolve_device(device_cfg: str) -> str:
    import torch
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


# ── Model & store initialisation ───────────────────────────────────────────────

def init_models(device: str) -> Dict[str, Any]:
    from .steps_caption import _unload_known_sidecars, _log_vram_snapshot
    _banner("Initialising models")
    models: Dict[str, Any] = {"device": device, "uses_api_embedder": False}

    # The pre-flight check above may have left Ollama sidecars resident in VRAM.
    # Evict them now so local model loads (GemmaEmbedder / OpenCLIP / DINO) have
    # enough headroom.  We'll re-load the sidecar models on-demand in each step.
    if device == "cuda":
        import gc as _gc
        import torch as _torch_init
        _unload_known_sidecars([
            (settings.GEMMA_API_URL, settings.GEMMA_API_MODEL),
            (getattr(settings, "QWEN_API_URL", ""), getattr(settings, "QWEN_MODEL", "")),
            (getattr(settings, "REASONING_API_URL", ""), getattr(settings, "REASONING_MODEL", "")),
        ])
        _gc.collect()
        _torch_init.cuda.empty_cache()

    if settings.MODEL_NAME == "gemma":
        if not _HAS_GEMMA:
            raise ImportError(
                "models.gemma_model is unavailable — install transformers and accelerate."
            )
        hf_token = settings.HF_TOKEN
        if not hf_token:
            _log.warning(
                "HF_TOKEN is not set. Gemma is a gated model — set HF_TOKEN=hf_... in .env "
                "or run: huggingface-cli login"
            )
        else:
            from pipeline.core.config import mask_secret as _mask  # noqa: PLC0415
            _log.info("  HF_TOKEN: %s", _mask(hf_token))
        _log.info("Loading GemmaEmbedder (%s) …", settings.GEMMA_MODEL_ID)
        t0 = time.time()
        try:
            models["clip"] = GemmaEmbedder(
                model_id=settings.GEMMA_MODEL_ID,
                device=device,
                use_bf16=settings.GEMMA_USE_BF16,
                hf_token=hf_token,
            )
        except Exception as exc:
            if settings.GEMMA_API_URL:
                _log.warning(
                    "  GemmaEmbedder load failed (%s) — falling back to OpenCLIP for embeddings. "
                    "Sidecar (%s) will still handle generative analysis.",
                    exc, settings.GEMMA_API_URL,
                )
            else:
                raise RuntimeError(
                    f"GemmaEmbedder failed to load: {exc}\n"
                    "Fix: set HF_TOKEN=hf_... in .env (accept license at "
                    "huggingface.co/google/gemma-4-it-2b) or run: huggingface-cli login"
                ) from exc
        else:
            models["dino"] = None
            models["uses_api_embedder"] = False
            _log.info(
                "  ✓ GemmaEmbedder ready in %.1fs  (dim=%d)",
                time.time() - t0,
                models["clip"].image_dim(),
            )
            _log.info(
                "  ℹ  SSL fine-tuning and distillation steps are skipped for Gemma embedder."
            )
            return models
        # Fall through to load OpenCLIP when local Gemma failed but sidecar is set

    _log.info("Loading OpenCLIP ViT-B-16 …")
    t0 = time.time()
    _log_vram_snapshot("before Gemma analysis step")
    models["clip"] = OpenCLIPEmbedder()
    _log.info("  ✓ CLIP ready in %.1fs  (dim=%d)", time.time() - t0, models["clip"].image_dim())

    if _HAS_DINO:
        _log.info("Loading DINOv3 ViT-B/14 …  (first run downloads ~330 MB)")
        t0 = time.time()
        try:
            models["dino"] = DINOEmbedder("dinov3_vitb14")
            _log.info("  ✓ DINO ready in %.1fs  (dim=%d)",
                      time.time() - t0, models["dino"].image_dim())
        except Exception as exc:
            _log.warning("  ✗ DINOv3 load failed (%s) — using CLIP only", exc)
            models["dino"] = None
    else:
        _log.warning("  ✗ models.dino_model unavailable — using CLIP only")
        models["dino"] = None

    return models


def init_store(models: Dict[str, Any], use_qdrant: bool) -> Tuple[Any, bool]:
    if not use_qdrant:
        _log.info("Qdrant disabled (--no-qdrant) — using in-memory cosine store")
        return InMemoryStore(), False
    try:
        from pipeline.storage.qdrant import QdrantStore
        clip_dim = models["clip"].image_dim()
        dino_dim = models["dino"].image_dim() if models.get("dino") else None
        store    = QdrantStore(clip_dim=clip_dim, dino_dim=dino_dim)
        store.client.get_collections()
        _log.info("✓ Qdrant connected at %s:%s  collection=%s",
                  settings.QDRANT_HOST, settings.QDRANT_PORT, settings.QDRANT_COLLECTION)
        return store, True
    except Exception as exc:
        _log.warning("Qdrant unavailable (%s) — falling back to in-memory store", exc)
        _log.warning("  To enable: docker run -p 6333:6333 qdrant/qdrant")
        return InMemoryStore(), False


# ── Video discovery ────────────────────────────────────────────────────────────

def find_videos(videos_dir: Path) -> List[Path]:
    return sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)


# ── Step H: compare + describe ────────────────────────────────────────────────

def step_compare_and_describe(
    frame_list: List[Tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    base_results: List[Dict],
    ft_results: List[Dict],
    models: Dict[str, Any],
    video_id: str,
    video_name: str,
    video_dir: Path,
    ckpt_mb: float,
    onnx_mb: float,
) -> Dict[str, Any]:
    """Step H: compare results, caption video, write comparison.md."""
    from .steps_report import write_comparison_md, write_description_md
    out_md       = video_dir / "comparison.md"
    sample_paths = [fp for fp, _ in frame_list[:10]]
    clip_model: OpenCLIPEmbedder = models["clip"]
    dino_model = models.get("dino")
    t0 = time.time()
    clip_model.encode_images([Image.open(p).convert("RGB") for p in sample_paths])
    base_infer_ms = (time.time() - t0) * 1000 / len(sample_paths)
    ft_infer_ms   = base_infer_ms
    if dino_model:
        t0 = time.time()
        dino_model.encode_images([Image.open(p).convert("RGB") for p in sample_paths])
        ft_infer_ms = (time.time() - t0) * 1000 / len(sample_paths)
    _log.info("Computing video-to-text description …")
    try:
        step = max(1, len(frame_list) // 32)
        sampled_imgs  = [Image.open(fp).convert("RGB") for fp, _ in frame_list[::step]]
        frame_embeds  = clip_model.encode_images(sampled_imgs)
        avg_embed     = frame_embeds.mean(axis=0)
        text_embeds   = clip_model.encode_texts(_TEXT_PROMPTS)
        scores        = text_embeds @ avg_embed
        ranked        = sorted(zip(_TEXT_PROMPTS, scores.tolist()), key=lambda x: x[1], reverse=True)
        text_descriptions = ranked[:3]; all_scored = ranked
        for desc, score in text_descriptions:
            _log.info("  Video description: \"%s\" (sim=%.3f)", desc, score)
    except Exception as exc:
        _log.warning("  Video-to-text failed (%s)", exc)
        text_descriptions = [("description unavailable", 0.0)]; all_scored = text_descriptions
    write_comparison_md(out_md, video_name, base_results, ft_results,
                        base_infer_ms, ft_infer_ms, ckpt_mb, onnx_mb, text_descriptions)
    desc_md = video_dir / "description.md"
    write_description_md(desc_md, video_name, frame_list, text_descriptions, all_scored)
    return {"text_descriptions": text_descriptions, "base_infer_ms": base_infer_ms,
            "ft_infer_ms": ft_infer_ms,
            "top_description": text_descriptions[0][0] if text_descriptions else ""}


# ── Agentic video synthesis helpers ───────────────────────────────────────────

def _build_context_prompt(video_name: str, video_context: Dict[str, Any]) -> str:
    """Build a text prompt summarising accumulated observations for the LLM."""
    parts = [f"Video: {video_name}"]

    meta = video_context.get("meta", {})
    if meta:
        parts.append(
            f"Duration: {meta.get('duration_sec', 0):.1f}s | Frames: {meta.get('frame_count', 0)}"
        )

    gem_ctx = video_context.get("gemma_analysis", {})
    if gem_ctx:
        parts.append(
            f"\nGemma analysis ({gem_ctx.get('n_frames', 0)} frames, "
            f"{gem_ctx.get('n_tasks', 0)} analyses):"
        )
        task_res = gem_ctx.get("task_results", {})
        clf = task_res.get("scene_classification", {})
        if clf.get("category_distribution"):
            top_cat = next(iter(clf["category_distribution"]))
            parts.append(f"  - dominant scene type: {top_cat}")
        fv = task_res.get("fact_verification", {})
        if fv.get("claims"):
            top_claim = max(fv["claims"], key=lambda r: r.get("mean_score", 0.0))
            parts.append(
                f"  - strongest visual claim: {top_claim['claim']} "
                f"(score={top_claim['mean_score']:.3f})"
            )
        sc = task_res.get("scene_change_detection", {})
        if sc.get("n_changes") is not None:
            parts.append(f"  - scene transitions detected: {sc['n_changes']}")
        cl = task_res.get("scene_clustering", {})
        if cl.get("n_clusters"):
            parts.append(f"  - semantic clusters: {cl['n_clusters']}")
        mnn = gem_ctx.get("mnn_rate")
        if mnn is not None:
            parts.append(f"  - Gemma/DINOv3 MNN agreement: {mnn:.1%}")

    top_descs = video_context.get("top_descriptions", [])
    if top_descs:
        parts.append("\nTop scene descriptions (CLIP similarity):")
        for desc, score in top_descs[:5]:
            parts.append(f"  - {desc} (score={score:.3f})")

    captions = video_context.get("captions", [])
    if captions:
        step = max(1, len(captions) // 20)
        sampled = captions[::step][:20]
        parts.append(
            f"\nPer-frame captions ({len(sampled)} sampled from {len(captions)}):"
        )
        for r in sampled:
            cap = r.get("caption", "")
            if cap:
                parts.append(f"  [{r.get('t_sec', 0.0):.1f}s] {cap}")

    asr_segs = video_context.get("asr_segments", [])
    if asr_segs:
        parts.append(f"\nAudio transcript ({len(asr_segs)} segments):")
        for seg in asr_segs[:10]:
            ts = seg.get("timestamp") or (0.0, 0.0)
            text = seg.get("text", "").strip()
            if text:
                parts.append(f"  [{ts[0]:.1f}s–{ts[1]:.1f}s] {text}")

    ocr_list = video_context.get("ocr", [])
    if ocr_list:
        ocr_with_text = [r for r in ocr_list if r.get("ocr_text")][:10]
        if ocr_with_text:
            parts.append(
                f"\nVisible text (OCR, {len(ocr_with_text)} frames with text):"
            )
            for r in ocr_with_text[:5]:
                parts.append(f"  [{r['t_sec']:.1f}s] {r['ocr_text'][:100]}")

    obj_counts = video_context.get("detections", {})
    if obj_counts:
        parts.append("\nDetected objects (label: count):")
        for label, count in sorted(obj_counts.items(), key=lambda x: -x[1])[:10]:
            parts.append(f"  - {label}: {count}")

    qwen_caps = video_context.get("qwen_captions", [])
    if qwen_caps:
        step = max(1, len(qwen_caps) // 10)
        sampled = qwen_caps[::step][:10]
        parts.append(
            f"\nDetailed scene analysis ({len(sampled)} sampled from {len(qwen_caps)}):"
        )
        for r in sampled:
            cap = r.get("caption") or r.get("scene_description") or ""
            if cap:
                parts.append(f"  [{r.get('t_sec', 0.0):.1f}s] {str(cap)[:200]}")

    return "\n".join(parts)


def _append_agentic_step(
    trace: List[Dict[str, Any]],
    *,
    step_id: str,
    title: str,
    description: str,
    status: str,
    context_inputs: Optional[List[str]] = None,
    context_outputs: Optional[List[str]] = None,
    risks: Optional[List[str]] = None,
    artifacts: Optional[List[str]] = None,
) -> None:
    trace.append(
        {
            "step_id": step_id,
            "title": title,
            "description": description,
            "status": status,
            "context_inputs": context_inputs or [],
            "context_outputs": context_outputs or [],
            "risks": risks or [],
            "artifacts": artifacts or [],
        }
    )


def _build_agentic_flow_prompt(video_name: str, video_context: Dict[str, Any]) -> str:
    trace = video_context.get("agentic_trace", [])
    lines = [
        f"Video: {video_name}",
        "You are auditing an agentic video-demo pipeline.",
        "Analyze how context is accumulated step by step, how later steps depend on earlier outputs, and where wrong context can propagate.",
        "",
        "Per-step trace:",
    ]
    for item in trace:
        lines.extend(
            [
                f"- Step {item.get('step_id')} {item.get('title')}",
                f"  Description: {item.get('description', '')}",
                f"  Status: {item.get('status', 'unknown')}",
                f"  Context received: {', '.join(item.get('context_inputs', [])) or 'none'}",
                f"  Context produced: {', '.join(item.get('context_outputs', [])) or 'none'}",
                f"  Risks: {', '.join(item.get('risks', [])) or 'none'}",
                f"  Artifacts: {', '.join(item.get('artifacts', [])) or 'none'}",
            ]
        )

    lines.extend(
        [
            "",
            "Write markdown with these sections exactly:",
            "## Flow Summary",
            "Short explanation of how context evolves through the pipeline.",
            "## Step-by-Step Agentic Context",
            "Use one bullet per step. For each step explain what context entered, what new context was created, and what later steps rely on it.",
            "## Risk Register",
            "Use one bullet per step. Explicitly call out misidentification risk, wrong-context risk, and propagation risk.",
            "## Highest-Risk Context Failures",
            "List the most important compounded failure modes across the pipeline.",
            "## Mitigations",
            "Recommend concrete checks or gates to reduce context corruption.",
            "",
            "Be specific. Focus on agentic context flow, not generic ML commentary.",
        ]
    )
    return "\n".join(lines)


def _build_agentic_flow_prompt_compact(video_name: str, video_context: Dict[str, Any]) -> str:
    """Compact audit prompt tuned for slow reasoning models on Ollama."""
    trace = video_context.get("agentic_trace", [])
    lines = [
        f"Video: {video_name}",
        "Audit the agentic pipeline context flow.",
        "Return markdown with these exact sections:",
        "## Flow Summary",
        "## Step-by-Step Agentic Context",
        "## Risk Register",
        "## Highest-Risk Context Failures",
        "## Mitigations",
        "",
        "Per-step trace:",
    ]
    for item in trace:
        lines.append(
            f"- {item.get('step_id')} {item.get('title')} | "
            f"status={item.get('status', 'unknown')} | "
            f"in={'; '.join(item.get('context_inputs', [])[:3]) or 'none'} | "
            f"out={'; '.join(item.get('context_outputs', [])[:3]) or 'none'} | "
            f"risks={'; '.join(item.get('risks', [])[:3]) or 'none'}"
        )
    lines += [
        "",
        "Be concise and specific.",
        "Keep the whole answer under 900 words.",
        "Focus on context propagation, stale context, misidentification, and mitigation.",
    ]
    return "\n".join(lines)


def _reasoning_timeout_for_model(model: str) -> float:
    base = float(getattr(settings, "REASONING_TIMEOUT_SEC", 240))
    m = (model or "").lower()
    if any(tag in m for tag in ("32b", "30b", "27b", "26b")):
        return max(base, 600.0)
    if any(tag in m for tag in ("14b", "12b")):
        return max(base, 360.0)
    return base


def _fallback_agentic_flow_analysis(video_context: Dict[str, Any]) -> str:
    trace = video_context.get("agentic_trace", [])
    lines = [
        "## Flow Summary",
        "The demo pipeline accumulates context progressively: frame sampling establishes the timeline, multimodal steps add semantic and geometric evidence, and later reasoning steps consume that evidence to produce higher-level conclusions. The main agentic risk is not a single wrong model output but error carry-over from early observations into later narrative and structured reasoning.",
        "",
        "## Step-by-Step Agentic Context",
    ]
    for item in trace:
        received = ", ".join(item.get("context_inputs", [])) or "no prior context"
        produced = ", ".join(item.get("context_outputs", [])) or "no durable context"
        lines.append(
            f"- **{item.get('step_id')} {item.get('title')}** receives {received}; "
            f"produces {produced}; downstream consumers inherit both its evidence and its errors."
        )
    lines += ["", "## Risk Register"]
    for item in trace:
        risk_text = "; ".join(item.get("risks", [])) or "low direct risk"
        lines.append(f"- **{item.get('step_id')} {item.get('title')}**: {risk_text}.")
    lines += [
        "",
        "## Highest-Risk Context Failures",
        "- Qwen detailed captioning is the most exposed step because it consumes accumulated Florence, ASR, OCR, depth, detection, and prior-Qwen state. One bad upstream cue can shift the frame narrative.",
        "- Final synthesis can convert uncertain intermediate evidence into confident-looking ontology or narrative text if confidence and disagreement are not surfaced explicitly.",
        "- Distillation and fine-tuning can preserve or amplify weak teacher assumptions if retrieval gains are accepted without semantic validation.",
        "",
        "## Mitigations",
        "- Gate downstream prompts with confidence and disagreement summaries rather than only positive evidence.",
        "- Keep per-step provenance in artifacts so a reviewer can trace each claim to its source step.",
        "- Add contradiction checks between captions, OCR, ASR, detections, and final narratives before exporting final conclusions.",
    ]
    return "\n".join(lines)


# ── Step AA: agentic flow artifact ───────────────────────────────────────────

def step_agentic_flow_artifact(
    video_name: str,
    video_dir: Path,
    video_context: Dict[str, Any],
    api_url: str,
    model: str,
) -> Dict[str, Any]:
    """Final step: generate an artifact tracing agentic context and risks."""
    from .steps_report import write_agentic_flow_md
    from .steps_caption import _log_vram_snapshot
    result: Dict[str, Any] = {"skipped": True, "llm_used": False, "model": model or "deterministic"}
    output_path = video_dir / "agentic_flow.md"
    llm_analysis = ""
    t0 = time.time()
    _log_vram_snapshot("before reasoning sidecar use")

    if api_url:
        try:
            import httpx

            endpoint = f"{api_url.rstrip('/')}/chat/completions"
            timeout_sec = _reasoning_timeout_for_model(model)
            attempts = [
                {
                    "prompt": _build_agentic_flow_prompt_compact(video_name, video_context),
                    "max_tokens": 1100,
                },
                {
                    "prompt": _build_agentic_flow_prompt(video_name, video_context),
                    "max_tokens": 1500,
                },
            ]
            last_exc: Optional[Exception] = None
            for idx, attempt in enumerate(attempts, 1):
                try:
                    _log.info(
                        "  Agentic flow reasoning attempt %d/%d (model=%s timeout=%.0fs max_tokens=%d)",
                        idx, len(attempts), model, timeout_sec, attempt["max_tokens"],
                    )
                    resp = httpx.post(
                        endpoint,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": attempt["prompt"]}],
                            "max_tokens": attempt["max_tokens"],
                            "temperature": 0.2,
                        },
                        timeout=timeout_sec,
                    )
                    resp.raise_for_status()
                    llm_analysis = resp.json()["choices"][0]["message"]["content"].strip()
                    if llm_analysis:
                        result["llm_used"] = True
                        _log.info("  ✓ Agentic flow analysis generated with %s", model)
                        break
                except Exception as exc:
                    last_exc = exc
                    _log.warning("  Agentic flow reasoning attempt %d failed (%s)", idx, exc)
            if not llm_analysis and last_exc is not None:
                raise last_exc
        except Exception as exc:
            _log.warning("  Agentic flow reasoning failed (%s) — using deterministic fallback", exc)

    if not llm_analysis:
        llm_analysis = _fallback_agentic_flow_analysis(video_context)
        result["model"] = "deterministic-fallback"

    elapsed = time.time() - t0
    write_agentic_flow_md(
        output_path,
        video_name,
        video_context.get("agentic_trace", []),
        elapsed,
        result["model"],
        llm_analysis,
    )
    _log_vram_snapshot("after reasoning sidecar use")
    result.update({"skipped": False, "elapsed_sec": elapsed, "output_path": str(output_path)})
    return result


# ── Step Z: video synthesis ───────────────────────────────────────────────────

def step_video_synthesis(
    video_name: str,
    video_dir: Path,
    video_context: Dict[str, Any],
    api_url: str,
    model: str,
) -> Dict[str, Any]:
    """Step Z: synthesise video ontology + narrative via Ollama/vLLM API.

    Uses all accumulated context from steps A–H as input.  No local model is
    loaded — this is a pure API call, so CLIP+DINO can remain offloaded.
    Writes ``video_synthesis.md`` and ``video_ontology.json``.
    """
    from .steps_report import write_video_synthesis_md
    from .steps_caption import _log_vram_snapshot
    result: Dict[str, Any] = {"skipped": True, "ontology": {}, "narrative": ""}
    if not api_url:
        _log.info("  Synthesis skipped (no QWEN_API_URL / --qwen-api-url set)")
        return result

    try:
        import httpx
    except ImportError:
        _log.warning("  httpx unavailable — skipping video synthesis")
        return result

    context_str = _build_context_prompt(video_name, video_context)
    endpoint    = f"{api_url.rstrip('/')}/chat/completions"
    t0          = time.time()
    _log_vram_snapshot("before synthesis sidecar use")
    ontology: Dict[str, Any] = {}
    narrative = ""

    # 1. Request structured ontology JSON
    ontology_prompt = (
        f"{context_str}\n\n"
        "Based on all the above observations, produce a structured video ontology "
        "as valid JSON with these fields:\n"
        '{\n'
        '  "domain": "string (e.g. outdoor_surveillance, urban_traffic, aerial_reconnaissance)",\n'
        '  "environment": "string (terrain/setting description)",\n'
        '  "primary_activities": ["list of main activities observed"],\n'
        '  "key_objects": ["list of key objects/entities"],\n'
        '  "temporal_structure": "string (how scene evolves over time)",\n'
        '  "scene_complexity": "low|medium|high",\n'
        '  "confidence": 0.0\n'
        '}\n\n'
        "Output only the JSON object, no other text."
    )
    try:
        resp = httpx.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": ontology_prompt}],
                "max_tokens": 512,
                "temperature": 0.1,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        ontology = json.loads(raw.strip())
        _log.info("  ✓ Video ontology generated  (domain=%s)", ontology.get("domain", "?"))
    except Exception as exc:
        _log.warning("  Ontology generation failed (%s)", exc)

    # 2. Request fine-grained narrative
    narrative_prompt = (
        f"{context_str}\n\n"
        "Write a fine-grained narrative description of this video in markdown. Cover:\n"
        "1. **Opening scene** — what is visible in the first frames\n"
        "2. **Main activity** — primary events, motion, and content\n"
        "3. **Environmental context** — terrain, lighting, setting details\n"
        "4. **Notable details** — specific objects, text, audio cues if any\n"
        "5. **Temporal evolution** — how the scene changes over time\n"
        "6. **Summary** — one-sentence overall description\n\n"
        "Be specific and grounded in the observations above. Use technical language "
        "appropriate for outdoor robotics and surveillance contexts."
    )
    try:
        resp = httpx.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": narrative_prompt}],
                "max_tokens": 1024,
                "temperature": 0.3,
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        narrative = resp.json()["choices"][0]["message"]["content"].strip()
        _log.info("  ✓ Video narrative generated (%d chars)", len(narrative))
    except Exception as exc:
        _log.warning("  Narrative generation failed (%s)", exc)

    elapsed = time.time() - t0
    _log.info("  ✓ Video synthesis complete in %.1fs", elapsed)

    write_video_synthesis_md(
        video_dir / "video_synthesis.md",
        video_name, ontology, narrative, elapsed, model,
    )
    if ontology:
        (video_dir / "video_ontology.json").write_text(
            json.dumps(ontology, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _log.info("  ✓ Ontology saved → video_ontology.json")

    result.update({"skipped": False, "ontology": ontology,
                   "narrative": narrative, "elapsed_sec": elapsed})
    _log_vram_snapshot("after synthesis sidecar use")
    return result


# ── Per-video orchestrator ────────────────────────────────────────────────────

def run_video_pipeline(
    args: Any,
    video_path: Path,
    output_dir: Path,
    models: Dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
    _out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all pipeline steps for a single video. Returns per-video stats dict.

    *_out* is an optional external dict that is used as the stats container.
    When provided, callers can inspect it for partial results if an exception
    escapes — the timings and frame counts recorded up to the failure point
    are preserved.
    """
    from .steps_embed import (
        step_extract_frames,
        step_index_to_store,
        step_base_model_search_test,
        step_finetuned_model_search_test,
    )
    from .steps_caption import (
        step_gemma_analysis,
        step_scene_captioning,
        step_asr_transcription,
        step_ocr_extraction,
        step_depth_estimation,
        step_object_detection,
        step_world_model_pass,
        step_qwen_captioning,
        _offload_models_to_cpu,
        _restore_models_to_gpu,
        _models_on_device,
        _prep_vram_for_step,
        _unload_ollama_model,
        _unload_known_sidecars,
        _log_vram_snapshot,
    )
    from .steps_ssl import step_ssl_finetune
    from .steps_distill import step_distill, step_export_model
    from .steps_map import step_create_3d_map
    from .steps_semantic_graph import step_build_semantic_environment_graph
    from .steps_yolo_sam import step_yolo_sam_detection
    from .steps_report import write_multimodal_md, write_final_stats_md, print_run_stats

    import concurrent.futures as _cf

    video_name = video_path.stem
    video_id   = video_name.replace(" ", "_").lower()
    video_dir  = output_dir / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    _banner(f"Processing video: {video_path.name}")
    _log.info("Output directory: %s", video_dir)

    # Use the shared container when provided so partial state is visible outside.
    if _out is None:
        _out = {}
    _out.update({"name": video_name, "video_path": str(video_path), "timings": {}})
    stats: Dict[str, Any] = _out
    T = stats["timings"]

    # Accumulated context passed through the pipeline; enriches synthesis at step Z.
    video_context: Dict[str, Any] = {"video_name": video_name}
    agentic_trace: List[Dict[str, Any]] = []
    video_context["agentic_trace"] = agentic_trace

    # Tracks whether CLIP+DINO backbones are on GPU (relevant only when device=="cuda").
    clip_dino_on_gpu = (device == "cuda" and _models_on_device(models, "cuda"))

    # ── Phase 1: Foundational ingestion (no gate) ─────────────────────────────
    _banner("Phase 1 — Foundational ingestion")

    # A: Extract frames
    _step(1, _TOTAL_STEPS, "Frame extraction")
    with _Timer(T, "A_extract"):
        a = step_extract_frames(video_path, video_id, video_dir, fps=args.fps)
    frame_list: List[Tuple[str, float]] = a["frame_list"]
    stats["frames"]       = a["meta"]["frame_count"]
    stats["duration_sec"] = a["meta"]["duration_sec"]
    video_context["meta"] = {
        "frame_count": stats["frames"],
        "duration_sec": stats["duration_sec"],
    }
    _append_agentic_step(
        agentic_trace,
        step_id="A",
        title="Frame extraction",
        description="Decode the source video into a timestamped frame sequence that every later step reuses.",
        status="ok" if frame_list else "empty",
        context_inputs=["raw video bytes"],
        context_outputs=[
            f"{len(frame_list)} timestamped frames",
            f"duration {stats['duration_sec']:.1f}s",
            "frame timeline for all downstream alignment",
        ],
        risks=[
            "sampling can miss short-lived objects or events",
            "timestamp drift can misalign later ASR/OCR/detection context",
            "wrong extraction rate biases all downstream context",
        ],
        artifacts=["frames_metadata.json"],
    )
    if not frame_list:
        _log.error("No frames extracted — skipping video %s", video_path.name)
        return stats

    # Agentic knowledge accumulator — enriches downstream steps as each completes.
    knowledge = VideoKnowledge(
        video_name=video_name,
        duration_sec=stats["duration_sec"],
        frame_count=stats["frames"],
    )

    # B: Index — needs CLIP+DINO on GPU
    if device == "cuda" and not clip_dino_on_gpu:
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)
    _step(2, _TOTAL_STEPS, "Vector store indexing")
    with _Timer(T, "B_index"):
        b = step_index_to_store(video_path, video_id, store, is_qdrant, models, frame_list)
    if device == "cuda":
        clip_dino_on_gpu = _models_on_device(models, device)
    stats["index_sec"] = b["elapsed_sec"]
    _append_agentic_step(
        agentic_trace,
        step_id="B",
        title="Vector store indexing",
        description="Embed frames for retrieval and establish the baseline semantic memory used by search steps.",
        status="ok",
        context_inputs=["timestamped frames", "base CLIP/DINO embeddings"],
        context_outputs=[
            "retrieval index populated",
            f"index latency {b['elapsed_sec']:.1f}s",
            "baseline visual neighborhoods",
        ],
        risks=[
            "embedding collisions can mix semantically different frames",
            "duplicate-heavy footage can distort nearest-neighbor context",
            "wrong baseline neighborhoods affect later search comparisons",
        ],
        artifacts=[],
    )

    # ── Phase 2: Multimodal analysis (no gate, parallel where feasible) ───────
    # The 3D-map step (I) is CPU-bound (pycolmap SfM) and is submitted to a
    # background thread after step L offloads CLIP+DINO to CPU.  All GPU steps
    # remain serialised on the main thread.  The background result is collected
    # at step 12, before CLIP+DINO are restored for the base search (step 11).
    _banner("Phase 2 — Multimodal analysis (parallel)")
    _map_executor = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sfm-bg")
    _map_future: Optional[_cf.Future] = None

    # J: Gemma open-weight multimodal analysis
    _step(3, _TOTAL_STEPS, "Gemma multimodal analysis → gemma_analysis.md")
    with _Timer(T, "J_gemma"):
        j = step_gemma_analysis(
            video_path, video_id, video_name, video_dir, frame_list, models,
            gemma_api_url=getattr(args, "gemma_api_url", ""),
            gemma_api_model=getattr(args, "gemma_api_model", ""),
        )
    if not j.get("skipped"):
        video_context["gemma_analysis"] = {
            "n_frames":        j.get("n_frames", 0),
            "n_tasks":         len(j.get("task_results", {})),
            "mnn_rate_dino":   j.get("dino_comparison", {}).get("mnn_rate"),
            "mnn_rate_clip":   j.get("clip_comparison", {}).get("mnn_rate"),
        }
        knowledge.add_gemma(
            j.get("task_results", {}),
            mnn_dino=j.get("dino_comparison", {}).get("mnn_rate") or 0.0,
        )
    _append_agentic_step(
        agentic_trace,
        step_id="J",
        title="Gemma multimodal analysis",
        description="Run coarse video-level reasoning to infer dominant scene type, transitions, clusters, and teacher-signal compatibility.",
        status="skipped" if j.get("skipped") else "ok",
        context_inputs=["sampled video frames", "existing embeddings"],
        context_outputs=[
            f"scene type {knowledge.scene_type or 'unknown'}",
            f"{knowledge.n_transitions} transitions",
            f"{knowledge.n_clusters} semantic clusters",
            "domain hint for captioning and later reasoning",
        ] if not j.get("skipped") else ["no persistent Gemma context"],
        risks=[
            "scene classification can over-generalize from sparse samples",
            "wrong domain hint can bias Florence and Qwen toward the wrong narrative",
            "teacher-similarity judgments can be mistaken for semantic truth",
        ],
        artifacts=["gemma_analysis.md"] if not j.get("skipped") else [],
    )
    # Unload Gemma from Ollama immediately after analysis — frees ~12+ GiB for Florence.
    _gemma_api_url_j = settings.GEMMA_API_URL or getattr(args, "gemma_api_url", "")
    _gemma_api_model_j = settings.GEMMA_API_MODEL or getattr(args, "gemma_api_model", "")
    if _gemma_api_url_j and _gemma_api_model_j and device == "cuda":
        _unload_ollama_model(_gemma_api_url_j, _gemma_api_model_j)

    # L: Scene captioning — offloads CLIP+DINO internally, does NOT restore them
    caption_results: List[Dict[str, Any]] = []
    if not args.no_caption:
        _step(4, _TOTAL_STEPS, "Florence-2 scene captioning → scene_captions.md")
        with _Timer(T, "L_caption"):
            l_cap = step_scene_captioning(
                frame_list, video_name, video_dir, device,
                models=models,
                qwen_api_url=getattr(args, "qwen_api_url", ""),
                qwen_model=getattr(args, "qwen_model", "") or settings.QWEN_MODEL,
                florence_api_url=getattr(args, "florence_api_url", ""),
                florence_model=getattr(args, "florence_model", ""),
                domain_hint=knowledge.domain_hint(),
            )
        caption_results = l_cap.get("captions", [])
        knowledge.add_captions(caption_results)
        if device == "cuda":
            clip_dino_on_gpu = False  # Florence offloaded them; we keep them off for M–Q
    else:
        T["L_caption"] = 0.0
        _step(4, _TOTAL_STEPS, "Scene captioning (skipped — --no-caption)")
    video_context["captions"] = caption_results
    _append_agentic_step(
        agentic_trace,
        step_id="L",
        title="Scene captioning",
        description="Generate per-frame scene captions and coarse temporal segments to seed later context-aware reasoning.",
        status="skipped" if args.no_caption else "ok",
        context_inputs=[
            "timestamped frames",
            knowledge.domain_hint() or "no domain hint",
        ],
        context_outputs=[
            f"{len(caption_results)} scene captions",
            f"{len(getattr(knowledge, '_segments', []))} caption segments",
            "frame-level prior scene descriptions",
        ] if caption_results else ["no caption context"],
        risks=[
            "caption hallucinations can create false scene priors",
            "repeated captions may hide real transitions",
            "wrong segment boundaries can contaminate later frame context",
        ],
        artifacts=["scene_captions.md"] if caption_results else [],
    )

    # Submit 3D-map (step I) to the background executor now that CLIP+DINO have
    # been offloaded to CPU by Florence.  The SfM reconstruction (pycolmap) is
    # purely CPU-bound and will overlap with steps M–R on the main thread.
    # The result is collected at step 12, before CLIP+DINO are restored to GPU.
    _log.info("  ▷ Submitting 3D-map step I to background thread (SfM+Splat) …")
    _map_future = _map_executor.submit(
        step_create_3d_map,
        video_path, video_id, video_dir, frame_list, models,
        run_sfm_flag=not args.no_sfm,
        run_gsplat_flag=not getattr(args, "no_gsplat", False),
        device=device,
    )

    # M: ASR — no CLIP/DINO needed; Whisper manages its own VRAM
    asr_result: Dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}
    if args.asr:
        _step(5, _TOTAL_STEPS, "ASR transcription → asr_subtitles.md")
        with _Timer(T, "M_asr"):
            asr_result = step_asr_transcription(video_path, frame_list, video_name, video_dir)
    else:
        T["M_asr"] = 0.0
    video_context["asr_segments"] = asr_result.get("segments", [])
    knowledge.add_asr(asr_result.get("subtitle_map", {}))
    _append_agentic_step(
        agentic_trace,
        step_id="M",
        title="ASR transcription",
        description="Transcribe audio and align subtitles to frames so later reasoning can use speech context.",
        status="skipped" if asr_result.get("skipped") else "ok",
        context_inputs=["video audio stream", "frame timestamps"],
        context_outputs=[
            f"{len(asr_result.get('segments', []))} ASR segments",
            f"{asr_result.get('covered_frames', 0)} subtitle-covered frames",
            "audio context aligned to timestamps",
        ] if not asr_result.get("skipped") else ["no audio context"],
        risks=[
            "transcription errors can inject false entities or actions",
            "language mismatch can produce wrong context with high confidence",
            "subtitle-frame misalignment can contaminate visual reasoning",
        ],
        artifacts=["asr_subtitles.md"] if not asr_result.get("skipped") else [],
    )

    # N: OCR
    ocr_result: Dict[str, Any] = {"skipped": True, "ocr_results": []}
    if args.ocr:
        _step(6, _TOTAL_STEPS, "OCR text extraction")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "N_ocr"):
            ocr_result = step_ocr_extraction(
                frame_list,
                video_name,
                video_dir,
                caption_results=caption_results,
            )
    else:
        T["N_ocr"] = 0.0
    video_context["ocr"] = ocr_result.get("ocr_results", [])
    knowledge.add_ocr(ocr_result.get("ocr_results", []))
    _append_agentic_step(
        agentic_trace,
        step_id="N",
        title="OCR extraction",
        description="Extract visible text from frames to enrich object and scene interpretation.",
        status="skipped" if ocr_result.get("skipped") else "ok",
        context_inputs=["frames", "caption-confidence prescreen when available"],
        context_outputs=[
            f"{ocr_result.get('non_empty', 0)} frames with OCR text",
            "visible-text evidence for Qwen and final synthesis",
        ] if not ocr_result.get("skipped") else ["no OCR context"],
        risks=[
            "small or low-contrast text can be missed",
            "false OCR tokens can create wrong named-entity context",
            "prescreen skips may discard frames with useful text",
        ],
        artifacts=[],
    )

    # O: Depth
    depth_result: Dict[str, Any] = {"skipped": True, "depth_results": []}
    if args.depth:
        _step(7, _TOTAL_STEPS, "Depth estimation")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "O_depth"):
            depth_result = step_depth_estimation(frame_list, video_name, video_dir)
        knowledge.add_depth(depth_result.get("depth_results", []))
    else:
        T["O_depth"] = 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="O",
        title="Depth estimation",
        description="Estimate relative scene geometry for near/far reasoning and scene-structure cues.",
        status="skipped" if depth_result.get("skipped") else "ok",
        context_inputs=["frames"],
        context_outputs=[
            f"{depth_result.get('ok_count', 0)} depth-estimated frames",
            "relative geometry cues for later prompts",
        ] if not depth_result.get("skipped") else ["no depth context"],
        risks=[
            "monocular depth can confuse scale and elevation",
            "depth failure in low-texture scenes can misstate geometry",
            "wrong depth priors can bias later scene explanations",
        ],
        artifacts=[],
    )

    # P: Detection — accumulate per-label object counts into context
    det_result: Dict[str, Any] = {"skipped": True, "detection_results": []}
    if args.detection:
        _step(8, _TOTAL_STEPS, "Object detection")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "P_detection"):
            det_result = step_object_detection(frame_list, video_name, video_dir)
        knowledge.add_detections(det_result.get("detection_results", []))
    else:
        T["P_detection"] = 0.0
    if not det_result.get("skipped"):
        obj_counts: Dict[str, int] = {}
        for _r in det_result.get("detection_results", []):
            for _d in _r.get("detections", []):
                lbl = _d.get("label", "unknown")
                obj_counts[lbl] = obj_counts.get(lbl, 0) + 1
        video_context["detections"] = obj_counts
    _append_agentic_step(
        agentic_trace,
        step_id="P",
        title="Object detection",
        description="Detect frame-level entities so later reasoning can reference concrete objects instead of only global scene text.",
        status="skipped" if det_result.get("skipped") else "ok",
        context_inputs=["frames"],
        context_outputs=[
            f"{det_result.get('total_objects', 0)} detected objects",
            f"top entities: {', '.join(knowledge.known_entities[:5]) or 'none'}",
        ] if not det_result.get("skipped") else ["no detection context"],
        risks=[
            "class confusion can misidentify critical objects",
            "open-vocabulary labels can drift semantically across frames",
            "false positives can become persistent agentic context",
        ],
        artifacts=[],
    )

    # P2: YOLO11 + SAM2/3 detection and segmentation
    yolo_sam_result: Dict[str, Any] = {"skipped": True, "detection_results": []}
    if not getattr(args, "no_yolo", False):
        _step(9, _TOTAL_STEPS, "YOLO11 + SAM2/3 detection → yolo_sam/ + detection_comparison.md")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "P2_yolo_sam"):
            yolo_sam_result = step_yolo_sam_detection(
                frame_list, video_name, video_dir, device,
                det_result=det_result,
            )
        if not yolo_sam_result.get("skipped"):
            knowledge.add_detections(yolo_sam_result.get("detection_results", []))
    else:
        T["P2_yolo_sam"] = 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="P2",
        title="YOLO11 + SAM2/3 detection and segmentation",
        description=(
            "Run YOLO11 for fast instance detection with priority-ordered output "
            "(human > vehicle > artificial > other), optionally refined with SAM2/3 "
            "segmentation masks. Produces annotated frames and a comparison artifact "
            "against the HF detector (step P)."
        ),
        status="skipped" if yolo_sam_result.get("skipped") else "ok",
        context_inputs=["frames", "HF detection results from step P"],
        context_outputs=[
            f"{yolo_sam_result.get('total_objects', 0)} YOLO detections",
            f"human={yolo_sam_result.get('human_count', 0)} vehicle={yolo_sam_result.get('vehicle_count', 0)} artificial={yolo_sam_result.get('artificial_count', 0)}",
            "annotated frames + JSON results + comparison.md",
        ] if not yolo_sam_result.get("skipped") else ["no YOLO context"],
        risks=[
            "YOLO class confusion can misidentify humans as objects (safety-critical)",
            "priority ordering treats all persons equally regardless of role",
            "SAM masks may bleed across object boundaries in cluttered frames",
            "comparison vs HF detector may hide YOLO-specific failure modes",
        ],
        artifacts=[
            "yolo_sam_results.json",
            "yolo_sam/frame_*_annotated.jpg",
            "detection_comparison.md",
        ] if not yolo_sam_result.get("skipped") else [],
    )

    # Q: World model
    world_result: Dict[str, Any] = {"skipped": True, "world_results": []}
    if args.world_model:
        _step(10, _TOTAL_STEPS, "World model video embeddings")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "Q_world"):
            world_result = step_world_model_pass(frame_list, video_name, video_dir)
    else:
        T["Q_world"] = 0.0
    if not world_result.get("skipped"):
        video_context["world_model_clips"] = world_result.get("ok_count", 0)
    _append_agentic_step(
        agentic_trace,
        step_id="Q",
        title="World model pass",
        description="Compress clips into temporal embeddings to capture motion-level context not visible in single frames.",
        status="skipped" if world_result.get("skipped") else "ok",
        context_inputs=["ordered frame clips"],
        context_outputs=[
            f"{world_result.get('ok_count', 0)} temporal clip embeddings",
            "coarse motion-context signal",
        ] if not world_result.get("skipped") else ["no temporal clip context"],
        risks=[
            "clip pooling can smooth away rare but important events",
            "temporal embeddings are hard to interpret and easy to overtrust",
            "wrong clip-level context can bias synthesis without clear provenance",
        ],
        artifacts=[],
    )

    # R: Qwen — uses ASR + OCR context from previous steps (already agentic)
    qwen_result: Dict[str, Any] = {"skipped": True, "results": []}
    if args.qwen:
        _step(11, _TOTAL_STEPS, "Qwen VLM detailed captioning → detailed_captions.md")
        with _Timer(T, "R_qwen"):
            qwen_result = step_qwen_captioning(
                frame_list, video_name, video_dir,
                subtitle_map=asr_result.get("subtitle_map", {}),
                ocr_results=ocr_result.get("ocr_results", []),
                # Pass a passthrough so QwenModel never creates a second CLIP
                # embedder (OpenCLIPTagger) that competes for VRAM.  In demo
                # mode we want full coverage; prescreening is not needed.
                clip_prescreen_fn=lambda _img: True,
                knowledge=knowledge,
            )
    else:
        T["R_qwen"] = 0.0
    if not qwen_result.get("skipped"):
        video_context["qwen_captions"] = qwen_result.get("results", [])
    _append_agentic_step(
        agentic_trace,
        step_id="R",
        title="Qwen detailed captioning",
        description="Fuse visual frames with accumulated Florence, ASR, OCR, depth, detections, and prior-Qwen state for structured per-frame reasoning.",
        status="skipped" if qwen_result.get("skipped") else "ok",
        context_inputs=[
            "frame image",
            "Florence scene priors",
            "ASR-aligned subtitle context",
            "OCR/depth/detection cues",
            "previous Qwen structured state",
        ],
        context_outputs=[
            f"{qwen_result.get('ok_count', 0)} detailed captions",
            "structured scene facts for downstream synthesis",
            "updated prior-state chain across frames",
        ] if not qwen_result.get("skipped") else ["no detailed reasoning context"],
        risks=[
            "upstream misidentification compounds inside one prompt",
            "previous-frame state can anchor the model to stale or wrong context",
            "rich prompt context can make uncertain claims look internally consistent",
        ],
        artifacts=["detailed_captions.md"] if not qwen_result.get("skipped") else [],
    )

    if any([args.asr, args.ocr, args.depth, args.detection, args.world_model, args.qwen]):
        _mm_md = video_dir / "multimodal_features.md"
        write_multimodal_md(_mm_md, video_name, asr_result, ocr_result,
                            depth_result, det_result, world_result, qwen_result)

    # C: Base model search — restore CLIP+DINO to GPU before joining 3D-map thread
    # (I must be joined first so the background thread no longer accesses models).
    # Evict Ollama (reloaded during step R) before restoring CLIP+DINO.
    if device == "cuda" and not clip_dino_on_gpu:
        if getattr(args, "qwen", False):
            _qwen_url   = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
            _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
            if _qwen_url and _qwen_model:
                _unload_ollama_model(_qwen_url, _qwen_model)
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)
    _step(12, _TOTAL_STEPS, "Base model transformation test → base_search.md")
    with _Timer(T, "C_base_search"):
        c = step_base_model_search_test(frame_list, store, is_qdrant, models,
                                        video_id, video_name, video_dir, top_k=args.top_k)
    base_results = c["results"]; query_frame = c["query_frame"]; query_t_sec = c["query_t_sec"]
    stats["base_top_score"] = base_results[0]["score"] if base_results else 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="C",
        title="Base search test",
        description="Measure retrieval behavior of the base model as the control reference for adaptation steps.",
        status="ok",
        context_inputs=["retrieval index", "query frame"],
        context_outputs=[
            f"top-{len(base_results)} baseline matches",
            f"query at {query_t_sec:.1f}s",
        ],
        risks=[
            "search quality may favor visual similarity over semantic identity",
            "one query frame can underrepresent broader retrieval behavior",
            "baseline errors can distort later before/after comparisons",
        ],
        artifacts=["base_search.md"],
    )

    # I: 3D map + Gaussian Splat — collect background-thread result (Phase 2 close)
    _step(13, _TOTAL_STEPS, "3D map + Gaussian Splat → 3d_map/ (joining background thread)")
    with _Timer(T, "I_3dmap"):
        if _map_future is not None:
            try:
                h = _map_future.result(timeout=600)  # up to 10 min for SfM
            except Exception as _map_exc:
                _log.warning("  3D-map background thread raised: %s", _map_exc, exc_info=True)
                h = {
                    "sfm_poses": 0, "method": "failed",
                    "points": None, "gsplat_method": "failed", "splat_ply": None,
                    "viewer_html": "",
                }
            finally:
                _map_executor.shutdown(wait=False)
        else:
            _map_executor.shutdown(wait=False)
            h = {
                "sfm_poses": 0, "method": "skipped",
                "points": None, "gsplat_method": "skipped", "splat_ply": None,
                "viewer_html": "",
            }
    stats["sfm_poses"]     = h["sfm_poses"]
    stats["map_method"]    = h["method"]
    stats["map_points"]    = int(h["points"].shape[0]) if h.get("points") is not None else 0
    stats["gsplat_method"] = h.get("gsplat_method", "skipped")
    stats["splat_ply"]     = h.get("splat_ply")
    semantic_graph_result: Dict[str, Any] = {"skipped": True}
    if not getattr(args, "no_yolo", False) and settings.YOLO_SSG_ENABLED:
        semantic_graph_result = step_build_semantic_environment_graph(
            video_id=video_id,
            video_name=video_name,
            video_dir=video_dir,
            yolo_sam_result=yolo_sam_result,
            map_result=h,
        )
    stats["semantic_graph_nodes"] = (
        semantic_graph_result.get("graph", {}).get("summary", {}).get("node_count", 0)
        if not semantic_graph_result.get("skipped")
        else 0
    )
    stats["semantic_graph_edges"] = (
        semantic_graph_result.get("graph", {}).get("summary", {}).get("edge_count", 0)
        if not semantic_graph_result.get("skipped")
        else 0
    )
    if h.get("splat_ply"):
        _log.info("  ✓ Gaussian Splat → %s", h["splat_ply"])
        _log.info("  ✓ Interactive viewer → %s", h.get("viewer_html", ""))
    video_context["map"] = {
        "method":        h["method"],
        "points":        stats["map_points"],
        "sfm_poses":     h["sfm_poses"],
        "gsplat_method": stats["gsplat_method"],
        "splat_ply":     stats["splat_ply"],
        "semantic_graph": semantic_graph_result.get("graph", {}).get("summary", {}),
    }
    _append_agentic_step(
        agentic_trace,
        step_id="I",
        title="3D map creation",
        description="Recover scene geometry and export sparse-map or splat artifacts for spatial interpretation (ran concurrently with steps M–R).",
        status="ok" if h["method"] not in ("failed", "skipped") else h["method"],
        context_inputs=["video frames", "camera-motion consistency"],
        context_outputs=[
            f"{stats['map_points']} map points",
            f"{stats['sfm_poses']} SfM poses",
            f"map method {stats['map_method']}",
            f"{stats['semantic_graph_nodes']} semantic nodes",
        ],
        risks=[
            "geometry failure can create confident but wrong spatial context",
            "SfM fallback outputs may look valid while lacking metric truth",
            "map artifacts can be overinterpreted as semantic evidence",
        ],
        artifacts=[
            "3d_map/sparse_map.npz",
            "3d_map/map_stats.json",
            "3d_map/semantic_environment_graph.json",
            "3d_map/semantic_environment_graph.md",
        ] if not semantic_graph_result.get("skipped") else ["3d_map/sparse_map.npz", "3d_map/map_stats.json"],
    )

    # ── Phase 3: SSL-gated adaptation (SSL gate required) ─────────────────────
    # Step D (SSL fine-tuning) always runs to evaluate the gate.
    # Steps E, F, G, H only proceed when D produces a valid checkpoint with
    # best_loss < _SSL_GATE_MAX_LOSS.  Z and AA always run as finalization.
    _banner("Phase 3 — SSL-gated adaptation")

    # D: SSL fine-tuning — DINOFineTuner loads its own separate DINO; offload ours first.
    # Skipped when using an API-based embedder — no local backbone to fine-tune.
    checkpoint_path = ""
    if models.get("uses_api_embedder"):
        T["D_finetune"] = 0.0
        T["E_distill"] = 0.0
        _step(14, _TOTAL_STEPS, "SSL DINOv3 fine-tuning (skipped — API embedder)")
        _step(15, _TOTAL_STEPS, "Knowledge distillation (skipped — API embedder)")
        student_backbone = None; student_dim = 768
    else:
        if device == "cuda" and clip_dino_on_gpu:
            _offload_models_to_cpu(models)
            clip_dino_on_gpu = False
        _step(14, _TOTAL_STEPS, "SSL DINOv3 fine-tuning → finetune_stats.md")
        with _Timer(T, "D_finetune"):
            d = step_ssl_finetune(video_id, video_name, video_dir, frame_list, device,
                                  epochs=args.epochs, batch_size=args.batch_size)
        stats["best_loss"] = d["best_loss"]; stats["ckpt_mb"] = d["ckpt_mb"]
        checkpoint_path    = d["checkpoint"]
        _append_agentic_step(
            agentic_trace,
            step_id="D",
            title="SSL fine-tuning",
            description="Adapt the local DINO backbone to mission-specific footage so retrieval neighborhoods reflect this video domain more closely.",
            status="ok",
            context_inputs=["frame sequence", "base DINO initialization"],
            context_outputs=[
                f"best loss {d['best_loss']:.4f}",
                "mission-adapted backbone checkpoint",
            ],
            risks=[
                "small-video adaptation can overfit to accidental patterns",
                "temporal positives can encode wrong sameness assumptions",
                "adapted features can improve scores while harming semantics",
            ],
            artifacts=["finetune_stats.md", "checkpoints/dino_ssl_best.pt"],
        )

        # ── SSL gate: only proceed to E/F/G/H if D produced a usable checkpoint ──
        import os as _os
        _ssl_best_loss = stats.get("best_loss", float("inf"))
        ssl_gate_passed = (
            bool(checkpoint_path)
            and _os.path.exists(checkpoint_path)
            and _ssl_best_loss < _SSL_GATE_MAX_LOSS
        )
        if ssl_gate_passed:
            _log.info(
                "  ✓ SSL gate passed (best_loss=%.4f < %.1f) — proceeding to distillation, "
                "ONNX export, and search comparison",
                _ssl_best_loss, _SSL_GATE_MAX_LOSS,
            )
        else:
            _log.warning(
                "  ✗ SSL gate did not pass (checkpoint=%r, best_loss=%.4f, threshold=%.1f) — "
                "skipping steps E/F/G/H (distillation, ONNX export, search comparison)",
                checkpoint_path, _ssl_best_loss, _SSL_GATE_MAX_LOSS,
            )

        # E: Distillation — maximum-hydration chain (Gemma teacher + caption anchor when available)
        student_backbone = None; student_dim = 768
        if ssl_gate_passed and not args.no_distill:
            # Build caption anchor embeddings from Florence captions via CLIP text encoder
            _cap_anchor_embs: Optional[np.ndarray] = None
            _scene_captions = caption_results  # set by step_scene_captioning (step L)
            if _scene_captions and models.get("clip"):
                try:
                    _cap_texts = [r.get("caption") or "" for r in _scene_captions]
                    _cap_texts = [t for t in _cap_texts if t.strip()]
                    if _cap_texts:
                        _clip_model = models["clip"]
                        # Only OpenCLIPEmbedder has encode_texts suitable for anchoring
                        if hasattr(_clip_model, "encode_texts") and not isinstance(_clip_model, GemmaEmbedder if _HAS_GEMMA else type(None)):
                            _cap_anchor_embs = _clip_model.encode_texts(_cap_texts)
                            _log.info(
                                "  Distillation caption anchors: %d CLIP text embeddings from Florence captions",
                                len(_cap_anchor_embs),
                            )
                except Exception as _exc:
                    _log.debug("  Caption anchor prep failed (%s) — distilling without anchor", _exc)

            # Use Gemma embedder as teacher when loaded and MODEL_NAME=gemma
            _gemma_teacher = None
            if _HAS_GEMMA and isinstance(models.get("clip"), GemmaEmbedder):
                _gemma_teacher = models["clip"]
                _log.info("  Using GemmaVisionTeacher for distillation (max hydration)")

            _step(15, _TOTAL_STEPS, "Knowledge distillation (max hydration) → ViT-S/14 student")
            with _Timer(T, "E_distill"):
                e_distill = step_distill(
                    checkpoint_path, frame_list, video_name, video_dir, device,
                    distill_epochs=args.distill_epochs, batch_size=args.batch_size,
                    caption_embeddings=_cap_anchor_embs,
                    gemma_embedder=_gemma_teacher,
                )
            if not e_distill["skipped"]:
                student_backbone         = e_distill["student_backbone"]
                student_dim              = e_distill["student_dim"]
                stats["distill_loss"]    = e_distill["best_loss"]
                stats["student_ckpt_mb"] = e_distill["ckpt_mb"]
                stats["student_dim"]     = student_dim
                stats["teacher_dim"]     = e_distill["teacher_dim"]
            _append_agentic_step(
                agentic_trace,
                step_id="E",
                title="Knowledge distillation",
                description="Compress teacher geometry and optional language anchors into a smaller student suitable for deployment.",
                status="skipped" if e_distill.get("skipped") else "ok",
                context_inputs=[
                    "fine-tuned teacher checkpoint",
                    "optional Gemma teacher alignment",
                    "optional Florence caption anchors",
                ],
                context_outputs=[
                    f"student dim {student_dim}",
                    f"best distill loss {e_distill.get('best_loss', float('nan')):.4f}",
                    "student deployment checkpoint",
                ] if not e_distill.get("skipped") else ["no distilled student"],
                risks=[
                    "teacher mistakes transfer into the student representation",
                    "caption anchors can inject wrong semantics into retrieval space",
                    "compression can erase rare but important distinctions",
                ],
                artifacts=["distill_stats.md", "checkpoints/student_best.pt"] if not e_distill.get("skipped") else [],
            )
        else:
            T["E_distill"] = 0.0
            _gate_reason = "SSL gate did not pass" if not ssl_gate_passed else "--no-distill"
            _step(15, _TOTAL_STEPS, f"Knowledge distillation (skipped — {_gate_reason})")
            _append_agentic_step(
                agentic_trace,
                step_id="E",
                title="Knowledge distillation",
                description="Compress teacher knowledge into a smaller deployable student.",
                status="skipped",
                context_inputs=["fine-tuned teacher checkpoint"],
                context_outputs=["no distilled student"],
                risks=[
                    "without this step, deployment relies on the larger teacher or ONNX export only",
                    "no compression audit is produced",
                ],
                artifacts=[],
            )
    if models.get("uses_api_embedder"):
        ssl_gate_passed = False
        student_backbone = None; student_dim = 768
        _append_agentic_step(
            agentic_trace,
            step_id="D",
            title="SSL fine-tuning",
            description="Adapt the local DINO backbone to mission-specific footage.",
            status="skipped",
            context_inputs=["API embedder mode"],
            context_outputs=["no local fine-tuning checkpoint"],
            risks=[
                "no task-specific adaptation is learned in API-embedder mode",
            ],
            artifacts=[],
        )
        _append_agentic_step(
            agentic_trace,
            step_id="E",
            title="Knowledge distillation",
            description="Compress teacher knowledge into a smaller deployable student.",
            status="skipped",
            context_inputs=["API embedder mode"],
            context_outputs=["no distillation artifacts"],
            risks=[
                "no student compression path is available in API-embedder mode",
            ],
            artifacts=[],
        )

    # F: ONNX export + gallery — restore CLIP+DINO (export uses models["dino"])
    # Skipped when SSL gate did not pass (no valid checkpoint to package).
    if ssl_gate_passed:
        if device == "cuda" and not clip_dino_on_gpu:
            _restore_models_to_gpu(models, device)
            clip_dino_on_gpu = _models_on_device(models, device)
        _step(16, _TOTAL_STEPS, "ONNX export + gallery build → edge_models/")
        with _Timer(T, "F_export"):
            e = step_export_model(checkpoint_path, frame_list, video_dir, device, models,
                                  no_onnx=args.no_onnx,
                                  student_backbone=student_backbone, student_dim=student_dim)
        stats["onnx_mb"] = e.get("onnx_mb", 0.0); stats["onnx_exported"] = e.get("exported", False)
        _append_agentic_step(
            agentic_trace,
            step_id="F",
            title="ONNX export",
            description="Package the best available backbone and gallery into deployment artifacts.",
            status="ok",
            context_inputs=["teacher or student backbone", "retrieval gallery frames"],
            context_outputs=[
                f"onnx exported={e.get('exported', False)}",
                "gallery.npz for edge classification",
            ],
            risks=[
                "export mismatches can change runtime behavior versus training",
                "gallery coverage can be too narrow for field use",
                "deployment artifacts can hide upstream semantic errors behind good latency",
            ],
            artifacts=["edge_models/dino_demo.onnx", "edge_models/gallery.npz"],
        )
    else:
        T["F_export"] = 0.0
        stats.setdefault("onnx_mb", 0.0); stats.setdefault("onnx_exported", False)
        _step(16, _TOTAL_STEPS, "ONNX export (skipped — SSL gate did not pass)")
        _append_agentic_step(
            agentic_trace,
            step_id="F",
            title="ONNX export",
            description="Package the best available backbone and gallery into deployment artifacts.",
            status="skipped",
            context_inputs=["SSL gate did not pass"],
            context_outputs=["no deployment artifacts"],
            risks=["no edge deployment artifacts produced"],
            artifacts=[],
        )

    # G: Fine-tuned search — only if SSL gate passed (needs fine-tuned or distilled backbone)
    ft_results: List[Dict] = []
    if ssl_gate_passed:
        _step(17, _TOTAL_STEPS, "Fine-tuned model transformation test → finetuned_search.md")
        with _Timer(T, "G_ft_search"):
            f = step_finetuned_model_search_test(frame_list, store, is_qdrant, models,
                                                 query_frame, query_t_sec, video_id, video_name,
                                                 video_dir, top_k=args.top_k)
        ft_results = f["results"]; stats["ft_top_score"] = ft_results[0]["score"] if ft_results else 0.0
        _append_agentic_step(
            agentic_trace,
            step_id="G",
            title="Fine-tuned search test",
            description="Re-run retrieval after adaptation to quantify search-space changes.",
            status="ok",
            context_inputs=["fine-tuned or distilled backbone", "same query frame as baseline"],
            context_outputs=[
                f"top-{len(ft_results)} adapted matches",
                f"top score {stats['ft_top_score']:.4f}",
            ],
            risks=[
                "score improvements can hide semantic regressions",
                "query reuse can overstate adaptation gains",
                "retrieval differences may reflect memorization rather than better context",
            ],
            artifacts=["finetuned_search.md"],
        )
    else:
        T["G_ft_search"] = 0.0
        stats.setdefault("ft_top_score", 0.0)
        _step(17, _TOTAL_STEPS, "Fine-tuned search (skipped — SSL gate did not pass)")
        _append_agentic_step(
            agentic_trace,
            step_id="G",
            title="Fine-tuned search test",
            description="Re-run retrieval after adaptation to quantify search-space changes.",
            status="skipped",
            context_inputs=["SSL gate did not pass"],
            context_outputs=["no fine-tuned retrieval results"],
            risks=["no before/after retrieval comparison available"],
            artifacts=[],
        )

    # H: Comparison + description — only runs when ssl_gate_passed (needs ft_results)
    if ssl_gate_passed:
        _step(18, _TOTAL_STEPS, "Model comparison + video description → comparison.md, description.md")
        with _Timer(T, "H_compare"):
            g = step_compare_and_describe(frame_list, store, is_qdrant, base_results, ft_results,
                                          models, video_id, video_name, video_dir,
                                          stats.get("ckpt_mb", 0.0), stats.get("onnx_mb", 0.0))
        if g:
            stats["base_infer_ms"]   = g.get("base_infer_ms", 0.0)
            stats["ft_infer_ms"]     = g.get("ft_infer_ms", 0.0)
            stats["top_description"] = g.get("top_description", "")
            video_context["top_descriptions"] = g.get("text_descriptions", [])
        _append_agentic_step(
            agentic_trace,
            step_id="H",
            title="Comparison and description",
            description="Summarize retrieval changes and derive a CLIP-based coarse natural-language description of the video.",
            status="ok",
            context_inputs=["baseline and adapted retrieval outputs", "sampled frame embeddings"],
            context_outputs=[
                f"top description: {stats.get('top_description', 'unknown')}",
                "comparison summary across model variants",
            ],
            risks=[
                "top text prompt may sound plausible but be too coarse or wrong",
                "comparison metrics can privilege ranking stability over semantics",
                "narrative labels can bias the final synthesis context",
            ],
            artifacts=["comparison.md", "description.md"],
        )
    else:
        T["H_compare"] = 0.0
        _step(18, _TOTAL_STEPS, "Model comparison (skipped — SSL gate did not pass)")
        _append_agentic_step(
            agentic_trace,
            step_id="H",
            title="Comparison and description",
            description="Summarize retrieval changes and derive a CLIP-based coarse natural-language description of the video.",
            status="skipped",
            context_inputs=["SSL gate did not pass"],
            context_outputs=["no comparison or description artifacts"],
            risks=["no adaptation quality signal produced"],
            artifacts=[],
        )

    # ── Finalization (always runs regardless of SSL gate) ──────────────────────
    # Z: Video synthesis — offload CLIP+DINO; Ollama API call only (no local model)
    if device == "cuda" and clip_dino_on_gpu:
        _offload_models_to_cpu(models)
        clip_dino_on_gpu = False  # noqa: F841
    _step(19, _TOTAL_STEPS, "Video synthesis (ontology + narrative) → video_synthesis.md")
    _qwen_url   = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
    _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
    with _Timer(T, "Z_synthesis"):
        step_video_synthesis(
            video_name, video_dir, video_context,
            api_url=_qwen_url, model=_qwen_model,
        )
    _append_agentic_step(
        agentic_trace,
        step_id="Z",
        title="Video synthesis",
        description="Use accumulated multimodal context to generate a structured ontology and narrative summary of the whole video.",
        status="ok" if _qwen_url else "skipped",
        context_inputs=[
            "Gemma summary",
            "captions, ASR, OCR, detections, Qwen frame reasoning",
            "retrieval description and map summary",
        ],
        context_outputs=[
            "video ontology",
            "global narrative summary",
        ] if _qwen_url else ["no synthesis output"],
        risks=[
            "final narrative can collapse uncertain evidence into a single confident story",
            "contradictions across modalities may be hidden in the synthesized summary",
            "wrong high-level framing can mask the original source of context errors",
        ],
        artifacts=["video_synthesis.md", "video_ontology.json"] if _qwen_url else [],
    )

    # AA: Agentic flow artifact — prefer Gemma reasoning, fall back to Qwen, then deterministic text.
    _step(20, _TOTAL_STEPS, "Agentic flow audit → agentic_flow.md")
    _agentic_url = (
        getattr(args, "reasoning_api_url", "")
        or getattr(settings, "REASONING_API_URL", "")
        or getattr(args, "gemma_api_url", "")
        or settings.GEMMA_API_URL
        or _qwen_url
    )
    _agentic_model = (
        getattr(args, "reasoning_model", "")
        or getattr(settings, "REASONING_MODEL", "")
        or getattr(args, "gemma_api_model", "")
        or settings.GEMMA_API_MODEL
        or _qwen_model
    )
    _append_agentic_step(
        agentic_trace,
        step_id="AA",
        title="Agentic flow audit",
        description="Audit the full context chain, explain step-to-step reasoning state, and register per-step risks of misidentification and wrong context.",
        status="ok",
        context_inputs=["complete pipeline trace", "all accumulated artifacts and summaries"],
        context_outputs=["agentic_flow.md audit report"],
        risks=[
            "reasoning model can restate upstream errors coherently",
            "audit quality depends on provenance captured from earlier steps",
            "fallback deterministic summary is less nuanced than the LLM audit",
        ],
        artifacts=["agentic_flow.md"],
    )
    with _Timer(T, "AA_agentic"):
        step_agentic_flow_artifact(
            video_name,
            video_dir,
            video_context,
            api_url=_agentic_url,
            model=_agentic_model,
        )
    if device == "cuda":
        _unload_known_sidecars(
            [
                (_agentic_url, _agentic_model),
                (_qwen_url, _qwen_model),
                (getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL,
                 getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL),
            ]
        )

    stats["pipeline_sec"] = sum(T.values())

    _banner(f"✓ Video complete: {video_path.name}")
    _log.info("  Output dir: %s", video_dir)
    return stats


def _run_video_pipeline_safe(
    args: Any,
    video_path: "Path",
    output_dir: "Path",
    models: Dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
) -> Dict[str, Any]:
    """Wrapper around :func:`run_video_pipeline` that always returns a stats dict.

    On exception, returns the partial stats dict with timings recorded up to
    the failure point so step times and frame counts are not lost.
    """
    _out: Dict[str, Any] = {}
    try:
        run_video_pipeline(args, video_path, output_dir, models, store, is_qdrant, device, _out=_out)
    except Exception as exc:
        _log.error("Pipeline failed for %s: %s", video_path.name, exc, exc_info=True)
        _out.setdefault("name", video_path.stem)
        _out["error"] = str(exc)
        _out.setdefault("timings", {})
        _out.setdefault("frames", 0)
        _out.setdefault("duration_sec", 0.0)
        timings = _out.get("timings", {})
        _out.setdefault("pipeline_sec", sum(timings.values()))
    return _out


# ── Main entry point ──────────────────────────────────────────────────────────

def run_demo(args: Any) -> None:
    """Run the end-to-end demo pipeline.

    Called by ``main.py --mode demo``.
    Env vars must be set by the caller (via :func:`apply_demo_env`) **before**
    this module is imported.
    """
    from .steps_caption import (
        _unload_known_sidecars,
        _resolve_ollama_gemma_model,
        _resolve_ollama_reasoning_model,
        _recommend_gemma_sidecar_models,
        _list_ollama_models,
        _log_vram_snapshot,
    )
    from .steps_report import write_final_stats_md, print_run_stats

    _configure_logging()
    _configure_warnings()

    output_dir = Path(args.output_dir).resolve()

    # --view-npz shortcut: just visualise existing NPZ files
    if getattr(args, "view_npz", None) is not None:
        if not _HAS_MPL:
            _log.error("matplotlib is required for the 3D viewer.  Install: pip install matplotlib")
            sys.exit(1)
        view_npz(args.view_npz if args.view_npz is not None else "", output_dir)
        return

    t_start = time.time()
    _banner("selfsuvis — End-to-End Demo Pipeline")
    _log.info("Videos directory : %s", args.videos_dir)
    _log.info("Output directory : %s", output_dir)
    _log.info("Device           : %s", args.device)
    _log.info("Epochs           : %d", args.epochs)
    _log.info("Qdrant           : %s", "disabled" if args.no_qdrant else "auto-detect")
    _log.info("SfM              : %s", "disabled" if args.no_sfm else "auto-detect (pycolmap)")
    multimodal_active = [args.asr, args.ocr, args.depth, args.detection, args.world_model, args.qwen]
    if any(multimodal_active):
        _log.info("Multimodal steps : %s",
                  " ".join(s for s, e in [("ASR", args.asr), ("OCR", args.ocr),
                                           ("Depth", args.depth), ("Detection", args.detection),
                                           ("WorldModel", args.world_model),
                                           ("Qwen", args.qwen)] if e))

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        _log.error("Videos directory does not exist: %s", videos_dir)
        _log.error("Create it with:  mkdir -p %s", videos_dir)
        sys.exit(1)

    videos = find_videos(videos_dir)
    if not videos:
        _log.error("No video files found in %s", videos_dir)
        _log.error("Supported formats: %s", " ".join(sorted(_VIDEO_EXTS)))
        sys.exit(1)

    _log.info("Found %d video(s): %s", len(videos), [v.name for v in videos])

    device   = _resolve_device(args.device)
    _log.info("Using device: %s", device)

    from pipeline.vision.registry import detect_resources  # noqa: PLC0415

    if device == "cuda":
        _unload_known_sidecars(
            [
                (getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL,
                 getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL),
                (getattr(args, "qwen_api_url", "") or getattr(settings, "QWEN_API_URL", ""),
                 getattr(args, "qwen_model", "") or getattr(settings, "QWEN_MODEL", "")),
                (getattr(args, "reasoning_api_url", "") or getattr(settings, "REASONING_API_URL", ""),
                 getattr(args, "reasoning_model", "") or getattr(settings, "REASONING_MODEL", "")),
            ]
        )
    resources = detect_resources()
    _log.info(
        "Detected resources: VRAM total %.1f GiB | VRAM free %.1f GiB | RAM %.1f GiB",
        resources.get("vram_gb", 0.0),
        resources.get("free_vram_gb", 0.0),
        resources.get("ram_gb", 0.0),
    )
    if device == "cuda" and resources.get("vram_gb", 0.0) <= 0.0:
        _log.warning(
            "CUDA was requested but VRAM auto-detection returned 0.0 GiB. "
            "If the NVIDIA driver is temporarily inaccessible, set GPU_TOTAL_GB_HINT "
            "and optionally GPU_FREE_GB_HINT to preserve correct model planning."
        )

    explicit_gemma_model = getattr(args, "gemma_api_model", "") or os.getenv("GEMMA_API_MODEL", "")
    explicit_reasoning_model = getattr(args, "reasoning_model", "") or os.getenv("REASONING_MODEL", "")
    auto_analysis_model, auto_reasoning_model = _recommend_gemma_sidecar_models(resources)
    if not explicit_gemma_model:
        os.environ["GEMMA_API_MODEL"] = auto_analysis_model
        settings.GEMMA_API_MODEL = auto_analysis_model  # type: ignore[misc]
    if not explicit_reasoning_model:
        os.environ["REASONING_MODEL"] = auto_reasoning_model
        settings.REASONING_MODEL = auto_reasoning_model  # type: ignore[misc]
    if not os.getenv("REASONING_API_URL") and not getattr(args, "reasoning_api_url", ""):
        fallback_reasoning_url = (
            getattr(args, "gemma_api_url", "")
            or settings.GEMMA_API_URL
            or getattr(args, "qwen_api_url", "")
            or settings.QWEN_API_URL
        )
        if fallback_reasoning_url:
            os.environ["REASONING_API_URL"] = fallback_reasoning_url
            settings.REASONING_API_URL = fallback_reasoning_url  # type: ignore[misc]

    _log.info(
        "Demo LLM plan: analysis model=%s | reasoning model=%s",
        settings.GEMMA_API_MODEL or auto_analysis_model,
        settings.REASONING_MODEL or auto_reasoning_model,
    )

    # Pre-flight: if a Gemma API URL is configured, verify it responds before
    # loading any models.  Fail loudly rather than silently skipping later.
    _gemma_url = settings.GEMMA_API_URL or getattr(args, "gemma_api_url", "")
    if _gemma_url:
        _gemma_model_cfg = getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL
        # Auto-resolve: swap for a model that's actually available in Ollama
        _gemma_model = _resolve_ollama_gemma_model(_gemma_url, _gemma_model_cfg)
        if _gemma_model != _gemma_model_cfg:
            # Persist resolution so all downstream steps see the correct model
            os.environ["GEMMA_API_MODEL"] = _gemma_model
            settings.GEMMA_API_MODEL = _gemma_model  # type: ignore[misc]
        _log.info("Gemma API pre-flight check (url=%s  model=%s) …", _gemma_url, _gemma_model)
        try:
            import httpx as _httpx
            _r = _httpx.post(
                f"{_gemma_url.rstrip('/')}/chat/completions",
                json={"model": _gemma_model, "messages": [{"role": "user", "content": "ping"}],
                      "max_tokens": 1},
                timeout=20.0,
            )
            if _r.status_code == 404:
                _log.error(
                    "Gemma model '%s' not found in Ollama (HTTP 404). "
                    "Pull it with: ollama pull %s\n"
                    "Available models: %s",
                    _gemma_model, _gemma_model,
                    _list_ollama_models(_gemma_url),
                )
                sys.exit(1)
            if _r.status_code >= 500:
                _log.error(
                    "Gemma API pre-flight failed (HTTP %d). "
                    "Ensure Ollama is running: ollama pull %s",
                    _r.status_code, _gemma_model,
                )
                sys.exit(1)
            _log.info("  ✓ Gemma API reachable (HTTP %d  model=%s)", _r.status_code, _gemma_model)
        except Exception as _exc:
            _log.error(
                "Gemma API pre-flight error: %s. "
                "Check that Ollama is running at %s",
                _exc, _gemma_url,
            )
            sys.exit(1)

    _reasoning_url = (
        getattr(args, "reasoning_api_url", "")
        or getattr(settings, "REASONING_API_URL", "")
    )
    if _reasoning_url:
        _reasoning_model_cfg = getattr(args, "reasoning_model", "") or getattr(settings, "REASONING_MODEL", "")
        if (
            (getattr(args, "reasoning_backend", "") or getattr(settings, "REASONING_BACKEND", "")).lower() == "ollama"
            or ":11434" in _reasoning_url
        ):
            _reasoning_model = _resolve_ollama_reasoning_model(_reasoning_url, _reasoning_model_cfg)
        else:
            _reasoning_model = _reasoning_model_cfg
        if _reasoning_model != _reasoning_model_cfg:
            os.environ["REASONING_MODEL"] = _reasoning_model
            settings.REASONING_MODEL = _reasoning_model  # type: ignore[misc]
        _log.info(
            "Reasoning API pre-flight check (url=%s  model=%s) …",
            _reasoning_url, _reasoning_model,
        )
        try:
            import httpx as _httpx
            _r = _httpx.post(
                f"{_reasoning_url.rstrip('/')}/chat/completions",
                json={"model": _reasoning_model, "messages": [{"role": "user", "content": "ping"}],
                      "max_tokens": 1},
                timeout=20.0,
            )
            if _r.status_code == 404:
                _log.error(
                    "Reasoning model '%s' not found at %s (HTTP 404). "
                    "Pull or serve it before running the demo.",
                    _reasoning_model, _reasoning_url,
                )
                sys.exit(1)
            if _r.status_code >= 500:
                _log.error(
                    "Reasoning API pre-flight failed (HTTP %d) for model '%s'.",
                    _r.status_code, _reasoning_model,
                )
                sys.exit(1)
            _log.info("  ✓ Reasoning API reachable (HTTP %d  model=%s)", _r.status_code, _reasoning_model)
        except Exception as _exc:
            _log.error(
                "Reasoning API pre-flight error: %s. Check endpoint %s",
                _exc, _reasoning_url,
            )
            sys.exit(1)

    t_init   = time.time()
    models   = init_models(device)
    store, is_qdrant = init_store(models, use_qdrant=not args.no_qdrant)
    init_elapsed = time.time() - t_init

    per_video_stats: List[Dict[str, Any]] = []
    try:
        for i, video_path in enumerate(videos, 1):
            _banner(f"Video {i}/{len(videos)}: {video_path.name}")
            try:
                vstats = _run_video_pipeline_safe(args, video_path, output_dir,
                                                  models, store, is_qdrant, device)
            except KeyboardInterrupt:
                raise
            per_video_stats.append(vstats)

    except KeyboardInterrupt:
        _log.warning("")
        _log.warning("Interrupted by user (Ctrl-C) — shutting down gracefully …")
        _log.warning("  %d/%d video(s) completed.", len(per_video_stats), len(videos))
        if per_video_stats:
            total_elapsed = time.time() - t_start
            stats_path    = output_dir / "final_stats.md"
            write_final_stats_md(stats_path, per_video_stats, total_elapsed)
            print_run_stats(per_video_stats, total_elapsed, init_elapsed, device)
            _log.warning("  Partial results written to: %s", stats_path)
        _log.warning("  Re-run to process remaining videos.")
        sys.exit(130)

    if not args.no_view:
        view_npz("", output_dir)

    total_elapsed = time.time() - t_start
    stats_path    = output_dir / "final_stats.md"
    write_final_stats_md(stats_path, per_video_stats, total_elapsed)
    print_run_stats(per_video_stats, total_elapsed, init_elapsed, device)

    _log.info("  Final statistics: %s", stats_path)
    _log.info("")
    _log.info("  Next steps:")
    _log.info("    • Edge inference:  EdgeClassifier('edge_models/dino_demo.onnx', 'edge_models/gallery.npz')")
    _log.info("    • Full stack:      make up")
    _log.info("    • Fine-tune rerun: DINO_CHECKPOINT=<path> python main.py --mode demo")
    _log.info("")
