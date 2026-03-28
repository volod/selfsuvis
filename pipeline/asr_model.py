"""Automatic Speech Recognition (ASR) model wrapper using Whisper via HuggingFace.

Supports all Whisper variants (tiny → large-v3-turbo), distil-whisper, and
NVIDIA Canary-1B.  The active model is selected by ``settings.ASR_MODEL``
(``"auto"`` = GPU-aware auto-selection from the model registry).

Usage::

    from pipeline.asr_model import ASRModel
    from pipeline.audio_extractor import extract_audio

    asr = ASRModel()
    if asr.is_enabled():
        wav = extract_audio(video_path, tmp_dir)
        if wav:
            segments = asr.transcribe(wav)
            # segments: [{"text": "...", "timestamp": (start_s, end_s)}, ...]

Top-10 ASR models (small → large, override with ``ASR_MODEL`` env var):

  1. openai/whisper-tiny           39 M   ~0.1 GB   fastest, basic quality
  2. openai/whisper-base           74 M   ~0.2 GB   good speed/quality balance
  3. openai/whisper-small          244 M  ~0.5 GB   strong multilingual
  4. openai/whisper-medium         769 M  ~1.5 GB   near large-v2 accuracy
  5. distil-whisper/distil-large-v3 756 M ~1.5 GB  6× faster than large-v3
  6. openai/whisper-large-v3-turbo 809 M  ~1.6 GB   8× speed vs large-v3
  7. openai/whisper-large-v2       1.55 B ~3.0 GB   best pre-v3 accuracy
  8. openai/whisper-large-v3       1.55 B ~3.0 GB   best overall accuracy
  9. nvidia/canary-1b              1.0 B  ~2.0 GB   punctuation/caps output
 10. facebook/seamless-m4t-v2-large 2.3 B ~4.6 GB  100+ language speech-to-text

CLI override examples::

    ASR_MODEL=openai/whisper-tiny python worker/main.py
    ASR_MODEL=openai/whisper-large-v3 ASR_LANGUAGE=en python worker/main.py
    ASR_MODEL=distil-whisper/distil-large-v3 ASR_BATCH_SIZE=4 python worker/main.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pipeline.config import settings
from pipeline.logging_utils import get_logger
from pipeline.model_registry import auto_select, detect_resources

logger = get_logger(__name__)

# Whisper-family model IDs that are supported natively by the HuggingFace
# transformers automatic-speech-recognition pipeline.
_WHISPER_PREFIXES = (
    "openai/whisper-",
    "distil-whisper/",
    "openai/whisper",
)


def _resolve_model_id() -> str:
    """Return the model ID to load, resolving ``"auto"`` via GPU detection."""
    model_cfg = settings.ASR_MODEL.strip()
    if model_cfg and model_cfg.lower() != "auto":
        return model_cfg
    resources = detect_resources()
    return auto_select("asr", resources) or "openai/whisper-large-v3-turbo"


class ASRModel:
    """Whisper-based ASR wrapper.

    Lazily loads the underlying HuggingFace pipeline on first ``transcribe()``
    call to avoid GPU memory allocation when ASR is disabled.
    """

    def __init__(self) -> None:
        self._pipe: Optional[Any] = None
        self._model_id: Optional[str] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        """Return True when ``ASR_ENABLED=true`` in the environment."""
        return settings.ASR_ENABLED

    def transcribe(self, audio_path: str) -> List[Dict]:
        """Transcribe *audio_path* (WAV/MP3/etc.) to a list of timestamped segments.

        Returns a list of ``{"text": str, "timestamp": (start_sec, end_sec)}``
        dicts, compatible with :func:`pipeline.audio_extractor.map_subtitles_to_frames`.

        Returns an empty list on any error (ASR failure never crashes indexing).
        """
        if not self.is_enabled():
            return []
        pipe = self._get_pipe()
        if pipe is None:
            return []
        try:
            language = settings.ASR_LANGUAGE.strip() or None
            generate_kwargs: Dict[str, Any] = {}
            if language:
                generate_kwargs["language"] = language

            result = pipe(
                audio_path,
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
                batch_size=settings.ASR_BATCH_SIZE,
                chunk_length_s=settings.ASR_CHUNK_LENGTH_SEC,
            )
            chunks = result.get("chunks", [])
            logger.info(
                "ASR transcribed %s → %d segments (model=%s)",
                audio_path, len(chunks), self._model_id,
            )
            return chunks
        except Exception:
            logger.warning("ASR transcription failed for %s", audio_path, exc_info=True)
            return []

    @property
    def model_id(self) -> str:
        """Resolved model ID (loaded lazily on first use)."""
        if self._model_id is None:
            self._model_id = _resolve_model_id()
        return self._model_id

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_pipe(self):
        """Lazily initialise and return the HuggingFace ASR pipeline."""
        if self._pipe is not None:
            return self._pipe

        model_id = self.model_id
        logger.info("Loading ASR model: %s", model_id)

        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = _resolve_device()
            torch_dtype = torch.float16 if settings.USE_FP16 and device != "cpu" else torch.float32

            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=device,
                torch_dtype=torch_dtype,
            )
            logger.info("ASR model loaded: %s on %s (dtype=%s)", model_id, device, torch_dtype)
        except Exception:
            logger.warning(
                "Failed to load ASR model %s; ASR will be skipped for this run",
                model_id, exc_info=True,
            )
            self._pipe = None

        return self._pipe


def _resolve_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on availability and DEVICE setting."""
    cfg = settings.DEVICE.lower()
    try:
        import torch
        if cfg == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if cfg == "cuda" and torch.cuda.is_available():
            return "cuda"
        if cfg == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
