"""Captioning steps: Gemma, Florence, Qwen, ASR, OCR, depth, detection, world model."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from pipeline.core import settings
from ._common import (
    _log,
    _open_frame_image,
    _open_frame_batch,
    _run_batched_frame_inference,
    _GEMMA_ANALYSIS_SAMPLE_N,
    _SCENE_CHANGE_THRESH,
    _GEMMA_TEXT_PROBES,
    VideoKnowledge,
)

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


# ── VRAM snapshot helper ──────────────────────────────────────────────────────

def _log_vram_snapshot(label: str) -> None:
    """Best-effort VRAM snapshot for both local process and sidecar-heavy runs."""
    try:
        from pipeline.vision.registry import detect_resources  # noqa: PLC0415

        resources = detect_resources()
        total = resources.get("vram_gb", 0.0)
        free = resources.get("free_vram_gb", 0.0)
        used = max(0.0, total - free) if total > 0 else 0.0
        _log.info(
            "  [VRAM] %s | total=%.1f GiB free=%.1f GiB used~=%.1f GiB ram=%.1f GiB",
            label,
            total,
            free,
            used,
            resources.get("ram_gb", 0.0),
        )
        return
    except Exception as exc:
        _log.debug("  [VRAM] %s | resource snapshot failed: %s", label, exc)


# ── Memory helpers for GPU-constrained machines ───────────────────────────────

def _offload_models_to_cpu(models: Dict[str, Any]) -> None:
    """Move CLIP and DINO backbones to CPU and flush the CUDA allocator cache.

    Called before loading a large model (Florence-2, ASR) when VRAM is tight.
    The embedders keep their ``self.device`` attribute unchanged so they work
    correctly once the backbone is moved back by :func:`_restore_models_to_gpu`.
    """
    _log_vram_snapshot("before offload CLIP+DINO to CPU")
    import gc
    import torch as _torch
    for key in ("clip", "dino"):
        m = models.get(key)
        if m is None:
            continue
        backbone = getattr(m, "model", None)
        if backbone is not None:
            try:
                backbone.cpu()
            except Exception:
                pass
    try:
        from models.dino_model import _set_dino_xformers_enabled
        _set_dino_xformers_enabled(False)
    except Exception:
        pass
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    free_mb = _torch.cuda.mem_get_info(0)[0] / 1024 ** 2 if _torch.cuda.is_available() else 0
    _log.info("  CLIP+DINO offloaded to CPU — %.0f MiB free on GPU", free_mb)
    _log_vram_snapshot("after offload CLIP+DINO to CPU")


def _prep_vram_for_step(
    models: Dict[str, Any],
    device: str,
    ollama_url: str = "",
    ollama_model: str = "",
) -> None:
    """Offload CLIP+DINOv3, evict any Ollama resident, and flush the CUDA allocator.

    Call this before loading any local inference model (OCR, depth, detection,
    world model) to maximise available VRAM and avoid OOM on 16 GiB class GPUs.
    """
    import gc
    import torch as _torch
    if device != "cuda":
        return
    _log_vram_snapshot("before prep VRAM for next step")
    _offload_models_to_cpu(models)
    _unload_known_sidecars(
        [
            (ollama_url, ollama_model),
            (settings.GEMMA_API_URL, settings.GEMMA_API_MODEL),
            (getattr(settings, "QWEN_API_URL", ""), getattr(settings, "QWEN_MODEL", "")),
            (getattr(settings, "REASONING_API_URL", ""), getattr(settings, "REASONING_MODEL", "")),
        ]
    )
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    free_mb = _torch.cuda.mem_get_info(0)[0] / 1024 ** 2 if _torch.cuda.is_available() else 0
    _log.info("  VRAM cleared for next step — %.0f MiB free", free_mb)
    _log_vram_snapshot("after prep VRAM for next step")


def _restore_models_to_gpu(models: Dict[str, Any], device: str) -> None:
    """Move CLIP and DINO backbones back to *device* after a large model releases."""
    _log_vram_snapshot(f"before restore models to {device}")
    import gc
    import os as _os
    import torch as _torch
    # Expandable segments let the allocator grow existing blocks rather than
    # searching for a new contiguous region — eliminates most fragmentation OOMs
    # when moving models on/off GPU between pipeline steps.
    _os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # Free any GPU memory held by objects that were just released before trying to
    # restore the backbones — prevents partial moves caused by transient OOM.
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    for key in ("clip", "dino"):
        m = models.get(key)
        if m is None:
            continue
        backbone = getattr(m, "model", None)
        if backbone is not None:
            try:
                backbone.to(device)
            except RuntimeError as exc:
                # OOM mid-.to() leaves the model in a mixed-device state.
                # Roll back to CPU first (releases all partially-moved params),
                # flush the allocator, then retry once — transient fragmentation
                # clears after the rollback frees the contiguous blocks it needs.
                try:
                    backbone.cpu()
                except Exception:
                    pass
                gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                try:
                    backbone.to(device)
                    _log.debug("  %s backbone moved to %s (retry succeeded)", key, device)
                except RuntimeError:
                    _log.warning(
                        "  Could not move %s backbone to %s (%s) — staying on CPU",
                        key, device, exc,
                    )
    try:
        from models.dino_model import _set_dino_xformers_enabled
        _set_dino_xformers_enabled(str(device).startswith("cuda"))
    except Exception:
        pass
    _log.info("  CLIP+DINO restored to %s", device)
    _log_vram_snapshot(f"after restore models to {device}")


def _models_on_device(models: Dict[str, Any], device: str) -> bool:
    import torch as _torch
    expected = _torch.device(device)
    for key in ("clip", "dino"):
        m = models.get(key)
        if m is None:
            continue
        backbone = getattr(m, "model", None)
        if backbone is None:
            continue
        try:
            actual = next(backbone.parameters()).device
        except StopIteration:
            continue
        if actual != expected:
            return False
    return True


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _list_ollama_models(api_url: str) -> List[str]:
    """Return model names available in the Ollama instance at *api_url*."""
    try:
        import httpx
        base = api_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        resp = httpx.get(f"{base}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


# Preferred Gemma model order: smallest usable first so we never pick a 26B/31B
# when a lighter option is available.
_GEMMA_PREFERENCE_ORDER = [
    "gemma4:e4b", "gemma4:4b", "gemma3:4b", "gemma3:1b",
    "gemma4:12b", "gemma3:12b",
    "gemma4:26b", "gemma4:31b", "gemma3:27b",
]


_REASONING_PREFERENCE_ORDER = [
    "deepseek-r1:32b", "qwen3:32b", "qwen3:30b",
    "deepseek-r1:14b", "qwen3:14b",
    "deepseek-r1:8b", "qwen3:8b",
    "gemma3:27b", "gemma3:12b", "gemma4:12b",
    "gemma3:4b", "gemma4:4b", "gemma4:e4b", "gemma3:1b",
]


def _recommend_gemma_sidecar_models(resources: Dict[str, float]) -> Tuple[str, str]:
    """Return recommended (analysis_model, reasoning_model) for current hardware.

    Analysis runs over sampled video frames and should stay relatively light.
    Reasoning runs once at the end and can use a larger long-thinking model.
    """
    vram = resources.get("vram_gb", 0.0)
    free_vram = resources.get("free_vram_gb", vram)
    ram = resources.get("ram_gb", 0.0)

    if free_vram >= 64 or vram >= 80:
        return "gemma4:26b", "deepseek-r1:32b"
    if free_vram >= 32 or vram >= 48:
        return "gemma4:12b", "qwen3:30b"
    if free_vram >= 18 or vram >= 24:
        return "gemma4:4b", "deepseek-r1:14b"
    if free_vram >= 10 or vram >= 16:
        return "gemma4:e4b", "deepseek-r1:14b"

    # CPU / mixed RAM-heavy fallback. Keep analysis lighter; spend RAM on the final audit.
    if ram >= 96:
        return "gemma4:12b", "deepseek-r1:32b"
    if ram >= 64:
        return "gemma4:4b", "deepseek-r1:14b"
    if ram >= 32:
        return "gemma4:e4b", "qwen3:8b"
    return "gemma3:1b", "gemma4:e4b"


def _resolve_ollama_model_with_preferences(
    api_url: str,
    configured_model: str,
    *,
    preference_order: List[str],
    family_prefixes: Tuple[str, ...],
    label: str,
) -> str:
    """Resolve a requested Ollama model against the instance model list."""
    available = _list_ollama_models(api_url)
    if not available:
        return configured_model
    if configured_model in available:
        return configured_model
    for preferred in preference_order:
        if preferred in available:
            _log.warning(
                "  %s model '%s' not found in Ollama; auto-selected '%s'. Pull the desired model with: ollama pull %s",
                label, configured_model, preferred, configured_model,
            )
            return preferred
    family_models = [m for m in available if m.startswith(family_prefixes)]
    if family_models:
        chosen = family_models[0]
        _log.warning(
            "  %s model '%s' not found; using first available family match: '%s'",
            label, configured_model, chosen,
        )
        return chosen
    return configured_model


def _resolve_ollama_gemma_model(api_url: str, configured_model: str) -> str:
    """Return the best available Gemma model for *api_url*.

    1. If *configured_model* is present in Ollama → use it.
    2. Otherwise scan available models and return the lightest Gemma by
       ``_GEMMA_PREFERENCE_ORDER``, or the first gemma* found.
    3. Falls back to *configured_model* (caller will get a 404 and fail clearly).
    """
    resolved = _resolve_ollama_model_with_preferences(
        api_url,
        configured_model,
        preference_order=_GEMMA_PREFERENCE_ORDER,
        family_prefixes=("gemma",),
        label="Gemma analysis",
    )
    if resolved == configured_model:
        available = _list_ollama_models(api_url)
        if available and not any(m.startswith("gemma") for m in available):
            _log.error(
                "No Gemma model found in Ollama. Pull one with: ollama pull gemma4:e4b\n"
                "Available models: %s", available,
            )
    return resolved


def _resolve_ollama_reasoning_model(api_url: str, configured_model: str) -> str:
    """Resolve the final reasoning model against Ollama availability."""
    return _resolve_ollama_model_with_preferences(
        api_url,
        configured_model,
        preference_order=_REASONING_PREFERENCE_ORDER,
        family_prefixes=("deepseek", "qwen", "llama", "gemma"),
        label="Reasoning",
    )


def _unload_ollama_model(api_url: str, model: str) -> bool:
    """Ask Ollama to evict *model* from VRAM by setting keep_alive=0.

    Only works when *api_url* points to an Ollama server (the /api/generate
    endpoint is Ollama-specific; vLLM will return 404 and we silently ignore
    that).  Returns True if the model was successfully unloaded.

    Typical VRAM freed: ~11–12 GiB for a 7B-param model, giving Florence-2
    (~1.5 GiB FP16) plenty of room to load locally.  Ollama auto-reloads the
    model on the next inference request (step R), so no explicit warmup needed.
    """
    try:
        import httpx
    except ImportError:
        return False
    base = api_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        resp = httpx.post(
            f"{base}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=15.0,
        )
        if resp.status_code == 200:
            _log.info("  Ollama: '%s' unloaded from VRAM", model)
            return True
        _log.debug("  Ollama unload returned HTTP %d — may be vLLM (ignored)", resp.status_code)
    except Exception as exc:
        _log.debug("  Could not contact Ollama for unload: %s", exc)
    return False


def _unload_known_sidecars(pairs: List[Tuple[str, str]]) -> None:
    """Unload all known Ollama sidecars from prior steps/runs when possible."""
    seen: set[Tuple[str, str]] = set()
    for url, model in pairs:
        if not url or not model:
            continue
        key = (url, model)
        if key in seen:
            continue
        seen.add(key)
        _unload_ollama_model(url, model)


# ── Gemma analysis ────────────────────────────────────────────────────────────

def _gemma_analyse_frame_via_api(
    fp: str,
    api_url: str,
    model: str,
    timeout: float,
) -> str:
    """Send a single frame to a Gemma Ollama/vLLM sidecar and return its description."""
    import base64
    import io

    try:
        import httpx
    except ImportError:
        return ""

    try:
        img = Image.open(fp).convert("RGB")
        img.thumbnail((768, 768))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",
                     "text": (
                         "Analyse this frame from aerial/robotics mission video. "
                         "Describe in 2-3 sentences: scene type, visible objects, "
                         "terrain, any notable features or anomalies. "
                         "Be concise and factual."
                     )},
                ],
            }],
            # 600 tokens: thinking models (gemma4:e4b) consume ~300-400 on reasoning
            # before writing the final answer into content.
            "max_tokens": 600,
            "temperature": 0.1,
        }
        endpoint = f"{api_url.rstrip('/')}/chat/completions"
        resp = httpx.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content")
        # Thinking models (gemma4:e4b) place the answer in content and
        # the chain-of-thought in reasoning.  If content is still empty
        # (budget exhausted on reasoning), use the last sentence of reasoning.
        if not content:
            reasoning = msg.get("reasoning") or msg.get("thinking") or ""
            if reasoning:
                # Take last non-empty sentence as a best-effort summary
                sentences = [s.strip() for s in reasoning.replace("\n", " ").split(".") if s.strip()]
                content = sentences[-1] if sentences else reasoning[-200:]
        # content may be a list of parts (some backends)
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        return (content or "").strip()
    except Exception as exc:
        _log.debug("  [Gemma API] frame analysis failed for %s: %s", Path(fp).name, exc)
        return ""


def step_gemma_analysis(
    video_path: Path,
    video_id: str,
    video_name: str,
    video_dir: Path,
    frame_list: List[Tuple[str, float]],
    models: Dict[str, Any],
    gemma_api_url: str = "",
    gemma_api_model: str = "",
) -> Dict[str, Any]:
    """Step J: Gemma open-weight multimodal video analysis.

    Uses the local GemmaEmbedder for embedding-based analysis and, when a
    Gemma Ollama/vLLM sidecar is configured (GEMMA_API_URL), also runs
    generative per-frame scene description.

    Analyses performed:
    1. Generative frame descriptions via Gemma sidecar (if GEMMA_API_URL set)
    2. Scene change detection — consecutive-frame cosine distance
    3. Semantic scene clustering — greedy cosine-based grouping
    4. Zero-shot scene classification — text probe vs frame embedding scores
    5. Cross-modal text -> frame retrieval — nearest-neighbour in Gemma space
    6. Temporal video embedding — mean-pool of all frame embeddings
    7. Gemma vs CLIP embedding comparison — MNN@k and mean pairwise cosine sim
    8. Gemma vs DINOv3 embedding comparison — MNN@k and mean pairwise cosine sim

    Skipped gracefully when GemmaEmbedder is unavailable and no sidecar is set.
    Writes ``gemma_analysis.md`` to *video_dir*.
    """
    import numpy as np
    from .steps_report import write_gemma_analysis_md, _write_gemma_captions_md

    result: Dict[str, Any] = {"skipped": True, "reason": ""}

    effective_api_url   = gemma_api_url or settings.GEMMA_API_URL
    effective_api_model = gemma_api_model or settings.GEMMA_API_MODEL
    effective_timeout   = float(settings.GEMMA_API_TIMEOUT_SEC)

    # Use local GemmaEmbedder when available; otherwise fall back to whatever
    # embedder is loaded (OpenCLIP) — all embedding analyses still run, just
    # powered by a different backbone.
    _clip_model = models.get("clip")
    has_local  = _clip_model is not None
    _embedder_name = (
        "GemmaEmbedder" if (_HAS_GEMMA and isinstance(_clip_model, GemmaEmbedder))
        else type(_clip_model).__name__ if _clip_model is not None
        else "none"
    )
    has_sidecar = bool(effective_api_url)

    if not has_local and not has_sidecar:
        result["reason"] = "No embedder available and GEMMA_API_URL not set"
        _log.info("  Gemma analysis skipped: %s", result["reason"])
        return result

    t0 = time.time()

    # Sample frames evenly
    n_avail  = len(frame_list)
    n_sample = min(_GEMMA_ANALYSIS_SAMPLE_N, n_avail)
    step     = max(1, n_avail // n_sample)
    sample_frames = frame_list[::step][:n_sample]
    sample_images = [_open_frame_image(fp) for fp, _ in sample_frames]
    sample_paths  = [fp for fp, _ in sample_frames]
    sample_ts     = [t for _, t in sample_frames]
    n = len(sample_images)
    _log.info("  Gemma analysis: %d sampled frames (from %d total)", n, n_avail)

    task_results: Dict[str, Any] = {}

    # 1. Generative per-frame analysis via Ollama/vLLM sidecar
    gemma_captions: List[Dict[str, Any]] = []
    if has_sidecar:
        _log.info(
            "  [Gemma] Generative scene analysis via sidecar (url=%s  model=%s  frames=%d) ...",
            effective_api_url, effective_api_model, n,
        )
        for idx, (fp, t_sec) in enumerate(sample_frames):
            desc = _gemma_analyse_frame_via_api(
                fp, effective_api_url, effective_api_model, effective_timeout,
            )
            gemma_captions.append({"frame_path": fp, "t_sec": t_sec, "description": desc})
            if (idx + 1) % 10 == 0:
                _log.info("    ... %d/%d frames analysed via Gemma sidecar", idx + 1, n)
        described = sum(1 for c in gemma_captions if c.get("description"))
        _log.info("  [Gemma] Generative descriptions: %d/%d frames", described, n)
        task_results["generative_descriptions"] = {
            "description": "Per-frame scene description generated by Gemma sidecar",
            "n_frames": n,
            "described_count": described,
            "model": effective_api_model,
            "captions": gemma_captions,
        }
        _write_gemma_captions_md(
            video_dir / "gemma_captions.md",
            video_name, effective_api_model, gemma_captions,
        )
    else:
        task_results["generative_descriptions"] = {
            "description": "Skipped — GEMMA_API_URL not configured",
            "skipped": True,
        }

    # For the remaining embedding-based analyses we need a local embedder.
    if not has_local:
        _log.info("  [Gemma] Skipping embedding analyses (no embedder loaded)")
        text_query_results: List[Dict[str, Any]] = []
        dino_comparison: Dict[str, Any] = {"available": False, "reason": "no embedder loaded"}
        clip_comparison: Dict[str, Any] = {"available": False, "reason": "GemmaEmbedder not loaded"}
        elapsed = time.time() - t0
        write_gemma_analysis_md(
            video_dir / "gemma_analysis.md",
            video_name, effective_api_model or settings.GEMMA_MODEL_ID,
            n, task_results, dino_comparison, text_query_results, elapsed,
            clip_comparison=clip_comparison,
        )
        result.update({
            "skipped": False, "n_frames": n,
            "task_results": task_results,
            "dino_comparison": dino_comparison,
            "clip_comparison": clip_comparison,
            "elapsed_sec": elapsed,
        })
        return result

    # Use whichever embedder is loaded (GemmaEmbedder preferred, CLIP fallback).
    gemma: GemmaEmbedder = models["clip"]  # type: ignore[assignment]
    _log.info("  [Gemma] Embedding analyses using %s", _embedder_name)

    # 2. Scene change detection via consecutive-frame cosine distance
    gemma_embeds: Optional[np.ndarray] = None
    try:
        _log.info("  [Gemma] Scene change detection ...")
        gemma_embeds = gemma.encode_images(sample_images)
        changes = []
        for i in range(1, n):
            cos_sim  = float(np.dot(gemma_embeds[i - 1], gemma_embeds[i]))
            distance = 1.0 - cos_sim
            if distance >= _SCENE_CHANGE_THRESH:
                changes.append({"frame_idx": i, "t_sec": sample_ts[i], "distance": distance})
        task_results["scene_change_detection"] = {
            "description": "Consecutive-frame cosine distance > threshold",
            "n_changes": len(changes),
            "threshold": _SCENE_CHANGE_THRESH,
            "changes": changes,
        }
        _log.info("  [Gemma] Scene changes detected: %d", len(changes))
    except Exception as exc:
        task_results["scene_change_detection"] = {"error": str(exc)}
        _log.warning("  [Gemma] Scene change detection failed: %s", exc)

    # 3. Greedy cosine-based scene clustering
    try:
        _log.info("  [Gemma] Semantic scene clustering ...")
        cl_embeds = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
        sim_mat   = np.dot(cl_embeds, cl_embeds.T)
        labels    = [-1] * n
        cluster_id = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            labels[i] = cluster_id
            for j in range(i + 1, n):
                if labels[j] == -1 and sim_mat[i, j] > (1.0 - _SCENE_CHANGE_THRESH):
                    labels[j] = cluster_id
            cluster_id += 1
        task_results["scene_clustering"] = {
            "description": "Greedy scene grouping by cosine similarity",
            "n_clusters": cluster_id,
            "n_frames": n,
            "mean_cluster_size": round(n / max(1, cluster_id), 2),
        }
        _log.info("  [Gemma] Scene clusters: %d from %d frames", cluster_id, n)
    except Exception as exc:
        task_results["scene_clustering"] = {"error": str(exc)}
        _log.warning("  [Gemma] Scene clustering failed: %s", exc)

    # 4. Zero-shot scene classification via text probe matching
    try:
        _log.info("  [Gemma] Zero-shot scene classification (%d probes) ...", len(_GEMMA_TEXT_PROBES))
        clf_frame  = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
        clf_text   = gemma.encode_texts(_GEMMA_TEXT_PROBES)
        clf_scores = np.dot(clf_frame, clf_text.T)  # (n_frames, n_categories)
        from collections import Counter
        top_cats: List[str] = [
            _GEMMA_TEXT_PROBES[int(np.argmax(clf_scores[i]))] for i in range(n)
        ]
        cat_dist = dict(Counter(top_cats).most_common(5))
        task_results["scene_classification"] = {
            "description": "Zero-shot classification against %d scene categories" % len(_GEMMA_TEXT_PROBES),
            "n_frames": n,
            "category_distribution": cat_dist,
        }
        _log.info("  [Gemma] Top category: %s", next(iter(cat_dist)) if cat_dist else "---")
    except Exception as exc:
        task_results["scene_classification"] = {"error": str(exc)}
        _log.warning("  [Gemma] Zero-shot classification failed: %s", exc)

    # 5. Cross-modal text -> frame retrieval
    text_query_results = []
    try:
        _log.info("  [Gemma] Cross-modal text->frame retrieval (%d probes) ...", len(_GEMMA_TEXT_PROBES))
        doc_embeds   = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
        query_embeds = gemma.encode_texts(_GEMMA_TEXT_PROBES)
        tq_scores    = np.dot(query_embeds, doc_embeds.T)  # (n_queries, n_frames)
        for q_idx, query in enumerate(_GEMMA_TEXT_PROBES):
            top_idxs = list(np.argsort(-tq_scores[q_idx])[:3])
            text_query_results.append({
                "query": query,
                "top_results": [
                    {"frame_path": sample_paths[i], "t_sec": sample_ts[i],
                     "score": float(tq_scores[q_idx, i])}
                    for i in top_idxs
                ],
            })
        task_results["cross_modal_retrieval"] = {
            "description": "Text probes matched against Gemma frame embeddings",
            "n_queries": len(_GEMMA_TEXT_PROBES),
        }
    except Exception as exc:
        task_results["cross_modal_retrieval"] = {"error": str(exc)}
        _log.warning("  [Gemma] Cross-modal retrieval failed: %s", exc)

    # 6. Temporal video embedding (mean-pool all frames)
    try:
        _log.info("  [Gemma] Temporal video embedding ...")
        vid_embed = gemma.encode_images_temporal(sample_images)
        task_results["temporal_embedding"] = {
            "description": "Mean-pool of %d frame embeddings -> single video-level vector" % n,
            "dim": int(vid_embed.shape[1]),
            "n_frames": n,
        }
        _log.info("  [Gemma] Temporal embedding dim=%d", vid_embed.shape[1])
    except Exception as exc:
        task_results["temporal_embedding"] = {"error": str(exc)}
        _log.warning("  [Gemma] Temporal embedding failed: %s", exc)

    # 7. Gemma vs CLIP comparison — skip when the main embedder IS CLIP (trivial)
    clip_comparison: Dict[str, Any] = {"available": False}
    from models.openclip_model import OpenCLIPEmbedder as _CLIPModel
    _main_is_clip = isinstance(gemma, _CLIPModel)
    if _main_is_clip:
        clip_comparison = {"available": False, "reason": "main embedder is OpenCLIP — comparison skipped (self vs self)"}
        _log.info("  [Gemma vs CLIP] Skipped — main embedder is already OpenCLIP")
    else:
        try:
            _log.info("  [Gemma vs CLIP] Loading temporary OpenCLIP ViT-B-16 ...")
            temp_clip  = _CLIPModel()
            clip_frame = temp_clip.encode_images(sample_images)
            g_e = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
            g_sim_c = np.dot(g_e, g_e.T)
            c_sim   = np.dot(clip_frame, clip_frame.T)
            mask_c  = ~np.eye(n, dtype=bool)
            mean_cossim_gemma_c = float(np.mean(g_sim_c[mask_c]))
            mean_cossim_clip    = float(np.mean(c_sim[mask_c]))
            k_c = min(5, n - 1)
            mnn_c = 0
            for i in range(n):
                gr = g_sim_c[i].copy(); gr[i] = -2.0
                cr = c_sim[i].copy();   cr[i] = -2.0
                mnn_c += len(set(np.argsort(-gr)[:k_c].tolist()) & set(np.argsort(-cr)[:k_c].tolist()))
            mnn_rate_c = mnn_c / (n * k_c)
            clip_comparison = {
                "available": True,
                "n_frames": n,
                "k": k_c,
                "mnn_rate": mnn_rate_c,
                "mean_cossim_gemma": mean_cossim_gemma_c,
                "mean_cossim_clip": mean_cossim_clip,
            }
            _log.info(
                "  [Gemma vs CLIP] MNN@%d=%.3f  mean_cossim: Gemma=%.4f  CLIP=%.4f",
                k_c, mnn_rate_c, mean_cossim_gemma_c, mean_cossim_clip,
            )
            try:
                import torch as _t
                _bb = getattr(temp_clip, "model", None)
                if _bb is not None:
                    _bb.cpu()
            except Exception:
                pass
            del temp_clip, clip_frame
        except Exception as exc:
            clip_comparison = {"available": False, "reason": str(exc)}
            _log.warning("  [Gemma vs CLIP] comparison failed: %s", exc)

    # 8. Gemma vs DINOv3 comparison
    dino_comparison: Dict[str, Any] = {"available": False}
    if _HAS_DINO and n > 1:
        try:
            _log.info("  [Gemma vs DINOv3] Loading temporary DINOv3 ViT-B/14 ...")
            temp_dino   = DINOEmbedder("dinov3_vitb14")
            dino_embeds = temp_dino.encode_images(sample_images)
            g_e = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
            g_sim_d = np.dot(g_e, g_e.T)
            d_sim   = np.dot(dino_embeds, dino_embeds.T)
            mask_d  = ~np.eye(n, dtype=bool)
            mean_cossim_gemma_d = float(np.mean(g_sim_d[mask_d]))
            mean_cossim_dino    = float(np.mean(d_sim[mask_d]))
            k_d = min(5, n - 1)
            mnn_d = 0
            for i in range(n):
                gr = g_sim_d[i].copy(); gr[i] = -2.0
                dr = d_sim[i].copy();   dr[i] = -2.0
                mnn_d += len(set(np.argsort(-gr)[:k_d].tolist()) & set(np.argsort(-dr)[:k_d].tolist()))
            mnn_rate_d = mnn_d / (n * k_d)
            dino_comparison = {
                "available": True,
                "n_frames": n,
                "k": k_d,
                "mnn_rate": mnn_rate_d,
                "mean_cossim_gemma": mean_cossim_gemma_d,
                "mean_cossim_dino": mean_cossim_dino,
            }
            _log.info(
                "  [Gemma vs DINOv3] MNN@%d=%.3f  mean_cossim: Gemma=%.4f  DINO=%.4f",
                k_d, mnn_rate_d, mean_cossim_gemma_d, mean_cossim_dino,
            )
            try:
                import torch as _t
                _bb = getattr(temp_dino, "model", None)
                if _bb is not None:
                    _bb.cpu()
            except Exception:
                pass
            del temp_dino, dino_embeds
        except Exception as exc:
            dino_comparison = {"available": False, "reason": str(exc)}
            _log.warning("  [Gemma vs DINOv3] comparison failed: %s", exc)
    else:
        dino_comparison["reason"] = (
            "DINOv3 not available" if not _HAS_DINO else "too few frames for comparison"
        )

    elapsed = time.time() - t0

    write_gemma_analysis_md(
        video_dir / "gemma_analysis.md",
        video_name, effective_api_model or settings.GEMMA_MODEL_ID,
        n, task_results, dino_comparison, text_query_results, elapsed,
        clip_comparison=clip_comparison,
    )

    result.update({
        "skipped": False,
        "n_frames": n,
        "task_results": task_results,
        "dino_comparison": dino_comparison,
        "clip_comparison": clip_comparison,
        "elapsed_sec": elapsed,
    })
    return result


# ── Florence / Qwen captioning ────────────────────────────────────────────────

def _caption_via_florence_api(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    api_url: str,
    model: str,
    domain_hint: str = "",
) -> Dict[str, Any]:
    """Caption frames via a vLLM endpoint serving Florence-2-large.

    vLLM serves Florence-2 with ``--task generate --trust-remote-code``.
    The ``<MORE_DETAILED_CAPTION>`` task token is passed as a text message
    alongside the base64-encoded image; the response is the plain caption string.

    This path consumes zero local VRAM — all inference runs inside the vLLM
    process, which can be on a separate GPU or port from Ollama.
    """
    import base64
    import io
    from .steps_report import write_scene_captions_md

    try:
        import httpx
    except ImportError:
        _log.warning("  httpx unavailable — cannot use Florence API")
        return {"skipped": True, "reason": "httpx not installed", "captions": []}

    _log.info(
        "  Florence-2 via vLLM API (url=%s  model=%s  frames=%d)",
        api_url, model, len(frame_list),
    )
    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    caption_results: List[Dict[str, Any]] = []
    t0 = time.time()

    for idx, (fp, t_sec) in enumerate(frame_list):
        caption = ""
        try:
            img = Image.open(fp).convert("RGB")
            img.thumbnail((768, 768))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": (
                            f"[Context: {domain_hint}] <MORE_DETAILED_CAPTION>"
                            if domain_hint else "<MORE_DETAILED_CAPTION>"
                        )},
                    ],
                }],
                "max_tokens": 256,
                "temperature": 0.0,
            }
            resp = httpx.post(endpoint, json=payload, timeout=60.0)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # Florence-2 sometimes echoes the task token; strip it
            if raw.startswith("<MORE_DETAILED_CAPTION>"):
                raw = raw[len("<MORE_DETAILED_CAPTION>"):].strip()
            caption = raw
        except Exception as exc:
            _log.debug("  Florence API error for %s: %s", Path(fp).name, exc)

        caption_results.append({
            "frame_path": fp, "t_sec": t_sec,
            "caption": caption,
            "caption_confidence": 0.75 if caption else 0.0,
        })
        if (idx + 1) % 20 == 0:
            _log.info("    ... %d/%d frames captioned via Florence API", idx + 1, len(frame_list))

    elapsed   = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info("  ✓ Florence API captions: %d/%d frames in %.1fs", captioned, len(frame_list), elapsed)
    out_md = video_dir / "scene_captions.md"
    write_scene_captions_md(out_md, video_name, caption_results, elapsed)
    return {
        "skipped": False, "captions": caption_results,
        "captioned_count": captioned, "elapsed_sec": elapsed, "backend": "florence_api",
    }


def _caption_via_qwen_api(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    api_url: str,
    model: str,
    domain_hint: str = "",
) -> Dict[str, Any]:
    """Caption frames via an OpenAI-compatible VLM endpoint (Ollama / vLLM).

    Used as a fallback when Florence-2 cannot load due to OOM.  Sends one
    ``/chat/completions`` request per frame with the image embedded as a base64
    data-URI.  Images are downscaled to 512 px on the longest side before
    encoding to keep latency reasonable.
    """
    import base64
    import io
    from .steps_report import write_scene_captions_md

    try:
        import httpx
    except ImportError:
        _log.warning("  httpx unavailable — cannot use Qwen API for captioning")
        return {"skipped": True, "reason": "httpx not installed", "captions": []}

    _log.info(
        "  Florence-2 OOM — falling back to Qwen API captioning "
        "(url=%s  model=%s  frames=%d)",
        api_url, model, len(frame_list),
    )
    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    caption_results: List[Dict[str, Any]] = []
    t0 = time.time()
    _MAX_CONSECUTIVE_FAILURES = 3

    # Pre-flight: verify the endpoint is responsive before iterating all frames.
    try:
        probe = httpx.post(
            endpoint,
            json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 1},
            timeout=15.0,
        )
        if probe.status_code >= 500:
            _log.warning(
                "  Qwen API pre-flight failed (HTTP %d) — skipping captioning",
                probe.status_code,
            )
            return {"skipped": True, "reason": f"Qwen API returned {probe.status_code}", "captions": []}
    except Exception as exc:
        _log.warning("  Qwen API pre-flight error (%s) — skipping captioning", exc)
        return {"skipped": True, "reason": str(exc), "captions": []}

    consecutive_failures = 0
    for idx, (fp, t_sec) in enumerate(frame_list):
        caption = ""
        try:
            img = Image.open(fp).convert("RGB")
            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text",
                         "text": (
                             (f"[Context: {domain_hint}]\n" if domain_hint else "")
                             + "Describe this image in one or two sentences. "
                             "Focus on the scene type, visible objects, and environment."
                         )},
                    ],
                }],
                "max_tokens": 150,
                "temperature": 0.1,
            }
            resp = httpx.post(endpoint, json=payload, timeout=30.0)
            resp.raise_for_status()
            caption = resp.json()["choices"][0]["message"]["content"].strip()
            consecutive_failures = 0
        except Exception as exc:
            _log.debug("  Qwen caption error for %s: %s", Path(fp).name, exc)
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                _log.warning(
                    "  Qwen API: %d consecutive failures — aborting captioning early "
                    "(%d/%d frames done)",
                    consecutive_failures, idx + 1, len(frame_list),
                )
                # Fill remaining frames with empty captions and break
                for fp2, t2 in frame_list[idx + 1:]:
                    caption_results.append({"frame_path": fp2, "t_sec": t2,
                                            "caption": "", "caption_confidence": 0.0})
                break

        caption_results.append({
            "frame_path": fp, "t_sec": t_sec,
            "caption": caption,
            "caption_confidence": 0.7 if caption else 0.0,
        })
        if (idx + 1) % 50 == 0:
            _log.info("    ... %d/%d frames captioned via Qwen API", idx + 1, len(frame_list))

    elapsed   = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info("  ✓ Qwen API captions: %d/%d frames in %.1fs", captioned, len(frame_list), elapsed)
    out_md = video_dir / "scene_captions.md"
    write_scene_captions_md(out_md, video_name, caption_results, elapsed)
    return {
        "skipped": False, "captions": caption_results,
        "captioned_count": captioned, "elapsed_sec": elapsed, "backend": "qwen_api",
    }


def step_scene_captioning(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
    models: Optional[Dict[str, Any]] = None,
    qwen_api_url: str = "",
    qwen_model: str = "",
    florence_api_url: str = "",
    florence_model: str = "",
    domain_hint: str = "",
) -> Dict[str, Any]:
    """Step L: Florence-2 scene captioning with memory management and API support.

    Memory strategy (CUDA only):
      1. If ``florence_api_url`` is set: call Florence-2 via vLLM API — no local
         weights loaded, zero VRAM consumed.  Use this when another process
         (e.g. Ollama) already occupies most of VRAM.
      2. Otherwise load Florence-2 locally:
         a. Offload CLIP+DINO to CPU to free ~1.7 GiB.
         b. If ``qwen_api_url`` looks like Ollama (port 11434): send keep_alive=0
            to evict the VLM (~11-12 GiB freed), giving Florence plenty of room.
            Ollama auto-reloads on the next request (step R).
         c. If Florence still OOMs and ``qwen_api_url`` + ``qwen_model`` are set:
            fall back to Qwen API captioning.
    """
    from .steps_report import write_scene_captions_md

    # ── API route: vLLM serving Florence-2 ────────────────────────────────────
    effective_florence_api_url = florence_api_url or settings.FLORENCE_API_URL
    effective_florence_model   = florence_model or settings.FLORENCE_MODEL
    if effective_florence_api_url:
        _log.info("  Florence-2 via vLLM API at %s", effective_florence_api_url)
        _log_vram_snapshot("before Florence API captioning")
        # Offload CLIP+DINO while API captions run (they aren't needed until step C)
        if models and device == "cuda":
            _offload_models_to_cpu(models)
        result = _caption_via_florence_api(
            frame_list, video_name, video_dir,
            effective_florence_api_url, effective_florence_model,
            domain_hint=domain_hint,
        )
        _log_vram_snapshot("after Florence API captioning")
        return result

    # ── Local route: load Florence-2 weights into this process ────────────────
    out_md = video_dir / "scene_captions.md"
    try:
        from pipeline.vision.florence import FlorenceModel
    except ImportError as exc:
        _log.warning("  Florence-2 unavailable (%s) — skipping captioning", exc)
        return {"skipped": True, "reason": str(exc), "captions": []}

    # Step 1: offload CLIP+DINO to free ~1.7 GiB
    if models and device == "cuda":
        _offload_models_to_cpu(models)

    # Step 2: unload all Ollama models to free VRAM for Florence
    if device == "cuda":
        if qwen_api_url and qwen_model:
            _unload_ollama_model(qwen_api_url, qwen_model)
        # Also unload Gemma sidecar if configured (may still be resident from step J)
        _gemma_url_cap = settings.GEMMA_API_URL
        _gemma_model_cap = settings.GEMMA_API_MODEL
        if _gemma_url_cap and _gemma_model_cap and _gemma_model_cap != qwen_model:
            _unload_ollama_model(_gemma_url_cap, _gemma_model_cap)

    _log.info("Loading Florence-2-large on %s …", device)
    _log_vram_snapshot("before local Florence load")
    t0 = time.time()
    try:
        florence = FlorenceModel()
    except Exception as exc:
        if qwen_api_url and qwen_model:
            _log.warning("  Florence-2 load failed (%s) — using Qwen API fallback", exc)
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    _log.info("  CUDA cache cleared before Qwen fallback")
            except Exception:
                pass
            return _caption_via_qwen_api(frame_list, video_name, video_dir, qwen_api_url, qwen_model,
                                         domain_hint=domain_hint)
        _log.warning(
            "  Florence-2 load failed (%s) — skipping captioning "
            "(pass --qwen-api-url + --qwen to enable Qwen API fallback)",
            exc,
        )
        return {"skipped": True, "reason": str(exc), "captions": []}

    _log.info("  ✓ Florence-2-large loaded in %.1fs", time.time() - t0)
    _log.info("  Captioning %d frames …", len(frame_list))
    caption_results: List[Dict[str, Any]] = []
    batch_size = settings.FLORENCE_BATCH_SIZE
    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        pil_images = []
        for fp, _t in batch:
            try:
                pil_images.append(Image.open(fp).convert("RGB"))
            except Exception:
                pil_images.append(Image.new("RGB", (224, 224)))
        try:
            captions_and_confs = florence.caption_batch(pil_images)
        except Exception as exc:
            _log.warning("  Florence batch %d failed: %s", batch_start, exc)
            captions_and_confs = [("", 0.5)] * len(batch)
        for (fp, t_sec), (cap, conf) in zip(batch, captions_and_confs):
            caption_results.append({"frame_path": fp, "t_sec": t_sec,
                                    "caption": cap, "caption_confidence": conf})

    elapsed   = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info("  ✓ %d/%d frames captioned in %.1fs", captioned, len(frame_list), elapsed)
    _log_vram_snapshot("after local Florence captioning")
    write_scene_captions_md(out_md, video_name, caption_results, elapsed)
    florence.release()
    # VRAM freed — caller (_run_video_pipeline) decides when to restore CLIP+DINO

    return {"skipped": False, "captions": caption_results,
            "captioned_count": captioned, "elapsed_sec": elapsed}


def step_qwen_captioning(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    subtitle_map: Dict[float, str],
    ocr_results: List[Dict[str, Any]],
    clip_prescreen_fn=None,
    knowledge: Optional["VideoKnowledge"] = None,
) -> Dict[str, Any]:
    """Step R: Qwen VLM detailed scene captioning with full agentic context.

    When *knowledge* is provided, each frame's prompt is enriched with all
    prior observations: Florence caption, depth profile, detected objects,
    scene segment, ASR, OCR, and the previous frame's Qwen structured output.
    This lets Qwen reason about *what changed* rather than describing each
    frame in isolation.
    """
    from .steps_report import write_detailed_captions_md

    out_md = video_dir / "detailed_captions.md"
    result: Dict[str, Any] = {"skipped": True, "results": []}
    try:
        from pipeline.vision.qwen import QwenModel
    except ImportError as exc:
        _log.warning("  Qwen model unavailable (%s) — skipping", exc)
        return result
    qwen = QwenModel(clip_prescreen_fn=clip_prescreen_fn)
    _log_vram_snapshot("before Qwen sidecar use")
    if not qwen.is_enabled():
        _log.info("  Qwen disabled (QWEN_API_URL not set) — skipping detailed captioning")
        _log.info("  To enable: --qwen-api-url http://localhost:8010/v1  (or set QWEN_API_URL)")
        return result
    ocr_map: Dict[float, str] = {r["t_sec"]: r["ocr_text"]
                                  for r in ocr_results
                                  if r.get("t_sec") is not None and r.get("ocr_text")}

    domain = knowledge.domain_hint() if knowledge else ""
    if domain:
        _log.info("  Qwen domain hint: %s", domain)
    _log.info("Running Qwen detailed captioning on %d frames (model=%s  agentic=%s) …",
              len(frame_list), settings.QWEN_MODEL, "yes" if knowledge else "no")
    t0 = time.time()

    caption_results: List[Dict[str, Any]] = []

    def _batch_fn(batch: List[Tuple[str, float]], imgs: List) -> List[Dict[str, Any]]:
        extra_contexts = None
        if knowledge:
            extra_contexts = [knowledge.context_for_frame(t_sec) for _fp, t_sec in batch]
        results = qwen.extract_batch(
            imgs,
            subtitle_texts=[subtitle_map.get(t_sec) or None for _fp, t_sec in batch],
            ocr_texts=[ocr_map.get(t_sec) or None for _fp, t_sec in batch],
            extra_contexts=extra_contexts,
            domain_hint=domain or None,
        )
        # Feed each successful result back into knowledge as prior state
        if knowledge:
            for r in results:
                knowledge.update_qwen_state(r)
        return results

    batch_results = _run_batched_frame_inference(
        frame_list,
        batch_size=4,
        batch_fn=_batch_fn,
        warning_label="Qwen",
        error_result={"service_unavailable": True},
    )
    for r in batch_results:
        t_sec = r.get("t_sec", 0.0)
        caption_results.append({**r, "subtitle_text": subtitle_map.get(t_sec) or ""})
    elapsed = time.time() - t0
    ok             = sum(1 for r in caption_results
                         if not r.get("service_unavailable") and not r.get("skipped"))
    subtitle_used  = sum(1 for r in caption_results if r.get("subtitle_text"))
    _log.info("  ✓ Qwen: %d/%d frames captioned in %.1fs (%d with ASR  agentic=%s)",
              ok, len(frame_list), elapsed, subtitle_used, "yes" if knowledge else "no")
    _log_vram_snapshot("after Qwen sidecar use")
    write_detailed_captions_md(out_md, video_name, caption_results, elapsed, settings.QWEN_MODEL)
    result.update({"skipped": False, "results": caption_results,
                   "ok_count": ok, "subtitle_used": subtitle_used, "elapsed_sec": elapsed})
    return result


def step_unidrive_analysis(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    subtitle_map: Dict[float, str],
    ocr_results: List[Dict[str, Any]],
    knowledge: Optional["VideoKnowledge"] = None,
) -> Dict[str, Any]:
    """Step S: UniDriveVLA expert analysis on a sparse frame sample."""
    from .steps_report import write_unidrive_analysis_md

    out_md = video_dir / "unidrive_analysis.md"
    result: Dict[str, Any] = {"skipped": True, "results": []}
    try:
        from pipeline.vision.unidrive import UniDriveVLAModel
    except ImportError as exc:
        _log.warning("  UniDriveVLA client unavailable (%s) — skipping", exc)
        return result

    client = UniDriveVLAModel()
    _log_vram_snapshot("before UniDrive sidecar use")
    if not client.is_enabled():
        _log.info("  UniDriveVLA disabled (UNIDRIVE_API_URL not set) — skipping")
        _log.info("  To enable: --unidrive-api-url http://localhost:8030/v1  (or set UNIDRIVE_API_URL)")
        return result

    max_frames = max(1, int(getattr(settings, "UNIDRIVE_MAX_FRAMES", 24) or 24))
    sample_step = max(1, len(frame_list) // max_frames)
    sampled_frames = frame_list[::sample_step][:max_frames]
    ocr_map: Dict[float, str] = {
        r["t_sec"]: r["ocr_text"]
        for r in ocr_results
        if r.get("t_sec") is not None and r.get("ocr_text")
    }
    domain = knowledge.domain_hint() if knowledge else ""
    _log.info(
        "Running UniDriveVLA expert analysis on %d sampled frames (model=%s) …",
        len(sampled_frames), settings.UNIDRIVE_MODEL,
    )
    t0 = time.time()

    def _batch_fn(batch: List[Tuple[str, float]], imgs: List[Image.Image]) -> List[Dict[str, Any]]:
        extra_contexts = None
        if knowledge:
            extra_contexts = [knowledge.context_for_frame(t_sec) for _fp, t_sec in batch]
        return client.extract_batch(
            imgs,
            subtitle_texts=[subtitle_map.get(t_sec) or None for _fp, t_sec in batch],
            ocr_texts=[ocr_map.get(t_sec) or None for _fp, t_sec in batch],
            extra_contexts=extra_contexts,
            domain_hint=domain or None,
        )

    batch_results = _run_batched_frame_inference(
        sampled_frames,
        batch_size=2,
        batch_fn=_batch_fn,
        warning_label="UniDriveVLA",
        error_result={"service_unavailable": True},
    )
    elapsed = time.time() - t0
    ok = sum(1 for r in batch_results if not r.get("service_unavailable") and not r.get("parse_error"))
    _log.info("  ✓ UniDriveVLA: %d/%d sampled frames analysed in %.1fs", ok, len(batch_results), elapsed)
    _log_vram_snapshot("after UniDrive sidecar use")
    write_unidrive_analysis_md(out_md, video_name, batch_results, elapsed, settings.UNIDRIVE_MODEL)
    result.update({
        "skipped": False,
        "results": batch_results,
        "ok_count": ok,
        "elapsed_sec": elapsed,
        "sampled_frames": len(batch_results),
    })
    return result


def step_asr_transcription(
    video_path: Path,
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step M: extract audio, run Whisper ASR."""
    from datetime import datetime
    from ._common import _RUNNER_LABEL

    out_md = video_dir / "asr_subtitles.md"
    result: Dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}
    try:
        from pipeline.media.audio import extract_audio, map_subtitles_to_frames
        from pipeline.vision.asr import ASRModel
    except ImportError as exc:
        _log.warning("  ASR unavailable (%s) — skipping", exc)
        return result
    asr = ASRModel()
    _log_vram_snapshot("before ASR model use")
    if not asr.is_enabled():
        _log.info("  ASR disabled (ASR_ENABLED=false) — skipping")
        return result
    audio_dir = video_dir / "audio"; audio_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Extracting audio from %s …", video_path.name)
    wav_path = extract_audio(str(video_path), str(audio_dir))
    if not wav_path:
        _log.warning("  No audio stream found in %s — ASR skipped", video_path.name)
        return result
    _log.info("Transcribing audio with %s …", asr.model_id)
    t0       = time.time()
    segments = asr.transcribe(wav_path)
    elapsed  = time.time() - t0
    if not segments:
        _log.warning("  ASR returned no segments for %s", video_path.name)
        return result
    frame_timestamps = [t for _, t in frame_list]
    subtitle_map     = map_subtitles_to_frames(segments, frame_timestamps,
                                               window_sec=settings.ASR_SUBTITLE_WINDOW_SEC)
    covered = sum(1 for t in frame_timestamps if t in subtitle_map)
    _log.info("  ✓ ASR: %d segments → %d/%d frames have subtitles (%.1fs, model=%s)",
              len(segments), covered, len(frame_list), elapsed, asr.model_id)
    _log_vram_snapshot("after ASR model use")
    lines = [
        f"# ASR Subtitles — {video_name}", f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{asr.model_id}`",
        f"Segments: {len(segments)}  |  Frames with subtitles: {covered}/{len(frame_list)}",
        f"Elapsed: {elapsed:.1f}s", f"",
        f"## Subtitle Segments", f"",
        f"| Start (s) | End (s) | Text |",
        f"|-----------|---------|------|",
    ]
    for seg in segments:
        ts = seg.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
        start = float(ts[0]) if len(ts) > 0 and ts[0] is not None else 0.0
        end = float(ts[1]) if len(ts) > 1 and ts[1] is not None else start
        text = seg.get("text", "").strip().replace("|", "\\|")
        lines.append(f"| {start:.2f} | {end:.2f} | {text} |")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · ASR step M*"]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    result.update({"skipped": False, "subtitle_map": subtitle_map,
                   "segments": segments, "elapsed_sec": elapsed, "covered_frames": covered})
    return result


def step_ocr_extraction(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    caption_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Step N: visible text extraction per frame."""
    result: Dict[str, Any] = {"skipped": True, "ocr_results": []}
    try:
        from pipeline.vision.ocr import OCRModel
    except ImportError as exc:
        _log.warning("  OCR unavailable (%s) — skipping", exc)
        return result
    ocr = OCRModel()
    _log_vram_snapshot("before OCR model use")
    if not ocr.is_enabled():
        _log.info("  OCR disabled (OCR_ENABLED=false) — skipping"); return result
    _log.info("Running OCR on %d frames (model=%s) …", len(frame_list), ocr.model_id)
    t0 = time.time()
    threshold = settings.OCR_MIN_CAPTION_CONFIDENCE
    caption_conf_by_frame: Dict[str, float] = {}
    if caption_results:
        caption_conf_by_frame = {
            str(r.get("frame_path")): float(r.get("caption_confidence", 0.0) or 0.0)
            for r in caption_results
            if r.get("frame_path")
        }

    selected_frame_list: List[Tuple[str, float]] = []
    skipped_by_caption: Dict[str, Dict[str, Any]] = {}
    if threshold > 0.0 and caption_conf_by_frame:
        for fp, t_sec in frame_list:
            conf = caption_conf_by_frame.get(fp)
            if conf is not None and conf >= threshold:
                skipped_by_caption[fp] = {
                    "frame_path": fp,
                    "t_sec": t_sec,
                    "ocr_text": "",
                    "ocr_model": ocr.model_id,
                    "ocr_skipped_by_caption": True,
                }
            else:
                selected_frame_list.append((fp, t_sec))
        _log.info(
            "  OCR prescreen: %d/%d frames selected (caption_confidence < %.2f)",
            len(selected_frame_list),
            len(frame_list),
            threshold,
        )
    else:
        selected_frame_list = list(frame_list)

    processed_results = _run_batched_frame_inference(
        selected_frame_list,
        batch_size=settings.OCR_BATCH_SIZE,
        batch_fn=lambda _batch, imgs: ocr.extract_text_batch(imgs),
        warning_label="OCR",
        error_result={"ocr_text": "", "ocr_error": True},
    )
    processed_by_frame = {str(r["frame_path"]): r for r in processed_results}
    ocr_results: List[Dict[str, Any]] = []
    for fp, t_sec in frame_list:
        if fp in processed_by_frame:
            ocr_results.append(processed_by_frame[fp])
        else:
            ocr_results.append(
                skipped_by_caption.get(
                    fp,
                    {"frame_path": fp, "t_sec": t_sec, "ocr_text": "", "ocr_error": True},
                )
            )
    elapsed   = time.time() - t0
    non_empty = sum(1 for r in ocr_results if r.get("ocr_text"))
    _log.info("  ✓ OCR: %d/%d frames have text in %.1fs", non_empty, len(frame_list), elapsed)
    result.update({"skipped": False, "ocr_results": ocr_results,
                   "non_empty": non_empty, "elapsed_sec": elapsed})
    ocr.release()
    _log_vram_snapshot("after OCR model use")
    return result


def step_depth_estimation(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step O: depth estimation per frame."""
    result: Dict[str, Any] = {"skipped": True, "depth_results": []}
    try:
        from pipeline.vision.depth import DepthModel
    except ImportError as exc:
        _log.warning("  Depth model unavailable (%s) — skipping", exc)
        return result
    depth_model = DepthModel()
    _log_vram_snapshot("before depth model use")
    if not depth_model.is_enabled():
        _log.info("  Depth disabled (DEPTH_ENABLED=false) — skipping"); return result
    _log.info("Running depth estimation on %d frames (model=%s) …",
              len(frame_list), depth_model.model_id)
    t0 = time.time()
    depth_results = _run_batched_frame_inference(
        frame_list,
        batch_size=4,
        batch_fn=lambda _batch, imgs: depth_model.estimate_batch(imgs),
        warning_label="Depth",
        error_result={"depth_error": True},
    )
    elapsed = time.time() - t0
    ok = sum(
        1
        for r in depth_results
        if not r.get("depth_error")
        and not r.get("depth_unavailable")
        and not r.get("depth_disabled")
    )
    _log.info("  ✓ Depth: %d/%d frames estimated in %.1fs", ok, len(frame_list), elapsed)
    result.update({"skipped": False, "depth_results": depth_results,
                   "ok_count": ok, "elapsed_sec": elapsed})
    depth_model.release()
    _log_vram_snapshot("after depth model use")
    return result


def step_object_detection(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step P: object detection per frame."""
    result: Dict[str, Any] = {"skipped": True, "detection_results": []}
    try:
        from pipeline.vision.detection import DetectionModel
    except ImportError as exc:
        _log.warning("  Detection model unavailable (%s) — skipping", exc)
        return result
    det_model = DetectionModel()
    _log_vram_snapshot("before detection model use")
    if not det_model.is_enabled():
        _log.info("  Detection disabled (DETECTION_ENABLED=false) — skipping"); return result
    _log.info("Running object detection on %d frames (model=%s) …",
              len(frame_list), det_model.model_id)
    t0 = time.time()
    det_results = _run_batched_frame_inference(
        frame_list,
        batch_size=4,
        batch_fn=lambda _batch, imgs: det_model.detect_batch(imgs),
        warning_label="Detection",
        error_result={"detection_error": True},
    )
    elapsed     = time.time() - t0
    total_objs  = sum(len(r.get("detections", [])) for r in det_results)
    ok          = sum(
        1
        for r in det_results
        if not r.get("detection_error")
        and not r.get("detection_unavailable")
        and not r.get("detection_disabled")
    )
    _log.info("  ✓ Detection: %d objects across %d/%d frames in %.1fs",
              total_objs, ok, len(frame_list), elapsed)
    result.update({"skipped": False, "detection_results": det_results,
                   "total_objects": total_objs, "ok_count": ok, "elapsed_sec": elapsed})
    det_model.release()
    _log_vram_snapshot("after detection model use")
    return result


def step_world_model_pass(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step Q: world model video embeddings."""
    result: Dict[str, Any] = {"skipped": True, "world_results": []}
    try:
        from pipeline.vision.world import WorldModel
    except ImportError as exc:
        _log.warning("  World model unavailable (%s) — skipping", exc)
        return result
    wm = WorldModel()
    _log_vram_snapshot("before world model use")
    if not wm.is_enabled():
        _log.info("  World model disabled (WORLD_MODEL_ENABLED=false) — skipping"); return result
    clip_frames = settings.WORLD_MODEL_CLIP_FRAMES
    _log.info("Running world model on %d frames in clips of %d (model=%s) …",
              len(frame_list), clip_frames, wm.model_id)
    t0 = time.time()
    world_results: List[Dict[str, Any]] = []
    for clip_start in range(0, len(frame_list), clip_frames):
        clip = frame_list[clip_start : clip_start + clip_frames]
        imgs = _open_frame_batch(clip)
        try:
            clip_out = wm.process_clip(imgs)
        except Exception as exc:
            _log.warning("  World model clip %d failed: %s", clip_start, exc)
            clip_out = {"world_model_error": True}
        mid = clip_start + len(clip) // 2
        fp, t_sec = frame_list[mid]
        world_results.append({"frame_path": fp, "t_sec": t_sec, **clip_out})
    elapsed = time.time() - t0
    ok = sum(
        1
        for r in world_results
        if not r.get("world_model_error")
        and not r.get("world_model_unavailable")
        and not r.get("world_model_disabled")
    )
    _log.info("  ✓ World model: %d clips processed in %.1fs", ok, elapsed)
    result.update({"skipped": False, "world_results": world_results,
                   "ok_count": ok, "elapsed_sec": elapsed})
    wm.release()
    _log_vram_snapshot("after world model use")
    return result
