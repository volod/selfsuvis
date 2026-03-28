"""OCR model wrapper — DeepSeek-OCR-2 (local or vLLM sidecar), TrOCR, GOT-OCR2.

Extracts visible text from frame images.  The result is stored in
``frame_facts_json["ocr_text"]`` and in the dedicated ``ocr_text`` DB column.

Mode selection (``OCR_MODEL`` env var, ``"auto"`` = GPU-aware):

  Local transformers (default when ``OCR_API_URL`` is empty):
    - TrOCR family (tiny, base, large) — fast, low VRAM, printed-document OCR
    - GOT-OCR2_0 — scene text, formulas, tables
    - DeepSeek-OCR-2 — best quality; 3B params (~7 GB VRAM)
    - VLM family (Phi-3.5-vision, Qwen2.5-VL, LLaVA) — chat-based OCR via
      AutoProcessor + AutoModelForCausalLM with trust_remote_code

  vLLM/ollama sidecar (when ``OCR_API_URL`` is non-empty):
    - Any vision-capable model served at the URL (typically DeepSeek-OCR-2 or
      Qwen2.5-VL running in the same sidecar as Phase 2)
    - Uses the same OpenAI-compatible API as the Qwen sidecar

Top-10 OCR models (small → large, override with ``OCR_MODEL``):

  1. microsoft/trocr-base-printed          334 M  ~0.7 GB  fast, printed text
  2. microsoft/trocr-large-printed         558 M  ~1.2 GB  better accuracy
  3. ucaslcl/GOT-OCR2_0                    580 M  ~1.2 GB  scene text + tables
  4. microsoft/Florence-2-base             230 M  ~0.5 GB  already in pipeline
  5. microsoft/Florence-2-large            770 M  ~1.5 GB  already in pipeline
  6. Qwen/Qwen2.5-VL-3B-Instruct           3 B    ~6.0 GB  strong spatial OCR
  7. deepseek-ai/DeepSeek-OCR-2            3 B    ~6.8 GB  DeepEncoder V2, best layout
  8. Qwen/Qwen2.5-VL-7B-Instruct           7 B   ~14.0 GB  top quality VLM OCR
  9. microsoft/Phi-3.5-vision-instruct     4.2 B  ~8.5 GB  128K ctx doc understanding
 10. llava-hf/llava-1.5-13b-hf            13 B   ~26.0 GB  strong VLM OCR

CLI override examples::

    OCR_ENABLED=true OCR_MODEL=ucaslcl/GOT-OCR2_0 python worker/main.py
    OCR_ENABLED=true OCR_API_URL=http://localhost:8010/v1 python worker/main.py
    OCR_ENABLED=true OCR_MODEL=auto python worker/main.py   # GPU auto-select
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

from PIL import Image

from pipeline.config import settings
from pipeline.florence_model import _best_attn_impl
from pipeline.logging_utils import get_logger
from pipeline.model_registry import auto_select, detect_resources

logger = get_logger(__name__)

_OCR_PROMPT = (
    "Extract all text visible in this image. "
    "Return only the extracted text, preserving layout as much as possible. "
    "If no text is visible, return an empty string."
)

_TROCR_PREFIXES = ("microsoft/trocr-",)
_GOT_PREFIXES = ("ucaslcl/GOT-",)
_DEEPSEEK_PREFIXES = ("deepseek-ai/DeepSeek-OCR",)
# VLM models that load via AutoProcessor + AutoModelForCausalLM (chat-style inference)
_VLM_PREFIXES = (
    "Qwen/Qwen2.5-VL-",
    "deepseek-ai/DeepSeek-OCR",
    "microsoft/Phi-3.5-vision",
    "llava-hf/",
)


def _resolve_model_id() -> str:
    model_cfg = settings.OCR_MODEL.strip()
    if model_cfg and model_cfg.lower() != "auto":
        return model_cfg
    resources = detect_resources()
    return auto_select("ocr", resources) or "microsoft/trocr-base-printed"


class OCRModel:
    """Extracts visible text from frame images.

    Supports four backends depending on configuration:
    - ``vllm``: HTTP call to ``OCR_API_URL`` (OpenAI-compatible vision endpoint)
    - ``got``: Local GOT-OCR2_0 via transformers
    - ``florence``: Local Florence-2 via transformers
    - ``trocr``: Local TrOCR via transformers (fastest, printed text only)
    - ``vlm``: Local VLM (Phi-3.5-vision, Qwen2.5-VL, etc.) via AutoModelForCausalLM
    """

    def __init__(self) -> None:
        self._model_id: Optional[str] = None
        self._backend: Optional[str] = None
        self._processor = None
        self._model = None
        self._got_tokenizer = None
        # Permanent load-failure flag — prevents retrying a failed load on every frame
        self._load_failed: bool = False

    # ── Public interface ──────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return settings.OCR_ENABLED

    def extract_text_batch(
        self, images: List[Image.Image]
    ) -> List[Dict[str, Any]]:
        """Extract text from a list of PIL images.

        Returns a list of dicts (one per image):
        ``{"ocr_text": "...", "ocr_model": "...", "ocr_error": True/absent}``.
        Never raises; returns error dicts on failure.
        """
        if not self.is_enabled():
            return [{"ocr_text": None, "ocr_disabled": True}] * len(images)
        return [self._extract_one(img) for img in images]

    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text from a single PIL image."""
        if not self.is_enabled():
            return {"ocr_text": None, "ocr_disabled": True}
        return self._extract_one(image)

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self._model_id = _resolve_model_id()
        return self._model_id

    # ── Backend dispatch ──────────────────────────────────────────────────────

    def _extract_one(self, image: Image.Image) -> Dict[str, Any]:
        backend = self._get_backend()
        try:
            if backend == "vllm":
                text = self._extract_vllm(image)
            elif backend == "got":
                text = self._extract_got(image)
            elif backend == "florence":
                text = self._extract_florence(image)
            elif backend == "vlm":
                text = self._extract_vlm_local(image)
            else:
                text = self._extract_trocr(image)
            return {"ocr_text": text.strip() if text else "", "ocr_model": self.model_id}
        except Exception:
            logger.warning("OCR extraction failed", exc_info=True)
            return {"ocr_text": "", "ocr_error": True, "ocr_model": self.model_id}

    def _get_backend(self) -> str:
        if self._backend is not None:
            return self._backend

        if settings.OCR_API_URL:
            self._backend = "vllm"
        elif any(self.model_id.startswith(p) for p in _GOT_PREFIXES):
            self._backend = "got"
        elif self.model_id.startswith("microsoft/Florence"):
            self._backend = "florence"
        elif any(self.model_id.startswith(p) for p in _VLM_PREFIXES):
            self._backend = "vlm"
        else:
            # TrOCR family (microsoft/trocr-*)
            self._backend = "trocr"
        logger.info("OCR backend: %s (model=%s)", self._backend, self.model_id)
        return self._backend

    # ── vLLM sidecar ─────────────────────────────────────────────────────────

    def _extract_vllm(self, image: Image.Image) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key="EMPTY",
            base_url=settings.OCR_API_URL,
            timeout=settings.OCR_TIMEOUT_SEC,
        )
        b64 = _encode_b64(image)
        response = client.chat.completions.create(
            model=settings.OCR_MODEL if settings.OCR_MODEL.lower() not in ("auto", "") else self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": _OCR_PROMPT},
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    # ── GOT-OCR2_0 ───────────────────────────────────────────────────────────

    def _extract_got(self, image: Image.Image) -> str:
        if self._model is None and not self._load_failed:
            self._load_got()
        if self._model is None:
            return ""
        import torch
        import tempfile, os
        # GOT-OCR2_0 requires the image saved to a temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
        try:
            image.save(tmp_path, format="JPEG")
            result = self._model.chat(
                self._got_tokenizer, tmp_path, ocr_type="ocr"
            )
            return result or ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _load_got(self) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            device = _get_device()
            self._got_tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto" if device == "cuda" else None,
                use_safetensors=True,
                pad_token_id=self._got_tokenizer.eos_token_id,
            ).eval()
            if device != "cuda":
                self._model = self._model.to(device)
            logger.info("GOT-OCR2 loaded: %s on %s", self.model_id, device)
        except Exception:
            logger.warning(
                "Failed to load GOT-OCR2 model %s — run: python scripts/prepare_models.py --ocr",
                self.model_id, exc_info=True,
            )
            self._model = None
            self._load_failed = True

    # ── TrOCR ────────────────────────────────────────────────────────────────

    def _extract_trocr(self, image: Image.Image) -> str:
        if self._model is None and not self._load_failed:
            self._load_trocr()
        if self._model is None:
            return ""
        try:
            import torch
            pixel_values = self._processor(images=image, return_tensors="pt").pixel_values
            device = _get_device()
            pixel_values = pixel_values.to(device)
            if settings.USE_FP16 and device != "cpu":
                pixel_values = pixel_values.half()
            with torch.no_grad():
                generated_ids = self._model.generate(pixel_values, max_new_tokens=512)
            return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            logger.warning("TrOCR inference failed", exc_info=True)
            return ""

    def _load_trocr(self) -> None:
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            device = _get_device()
            self._processor = TrOCRProcessor.from_pretrained(self.model_id)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            self._model.eval()
            if settings.USE_FP16 and device != "cpu":
                self._model = self._model.half()
            self._model = self._model.to(device)
            logger.info("TrOCR loaded: %s on %s", self.model_id, device)
        except Exception:
            logger.warning(
                "Failed to load TrOCR model %s — run: python scripts/prepare_models.py --ocr",
                self.model_id, exc_info=True,
            )
            self._model = None
            self._load_failed = True

    # ── VLM local (Phi-3.5-vision, Qwen2.5-VL, DeepSeek-OCR-2, LLaVA) ───────

    def _extract_vlm_local(self, image: Image.Image) -> str:
        if self._load_failed:
            return ""
        if self._model is None:
            self._load_vlm_local()
        if self._model is None:
            return ""
        try:
            import torch
            device = _get_device()

            # Strategy 1: apply_chat_template with image+text message (modern VLMs)
            try:
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": _OCR_PROMPT},
                ]}]
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=text, images=[image], return_tensors="pt"
                ).to(device)
            except Exception:
                # Strategy 2: direct processor call (fallback for simpler processors)
                inputs = self._processor(
                    text=_OCR_PROMPT, images=image, return_tensors="pt"
                ).to(device)

            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)

            # Strip echoed input tokens — only decode the newly generated portion
            input_len = inputs["input_ids"].shape[1]
            return self._processor.decode(out[0, input_len:], skip_special_tokens=True)
        except Exception:
            logger.warning("VLM OCR inference failed for %s", self.model_id, exc_info=True)
            return ""

    def _load_vlm_local(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM

            device = _get_device()
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            dtype = torch.float16 if (settings.USE_FP16 and device != "cpu") else torch.float32
            load_kwargs: dict = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "attn_implementation": _best_attn_impl(),
            }
            # device_map requires accelerate; fall back to .to(device) when absent
            try:
                import accelerate  # noqa: F401
                if device == "cuda":
                    load_kwargs["device_map"] = "auto"
            except ImportError:
                pass

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **load_kwargs
            ).eval()
            if "device_map" not in load_kwargs:
                self._model = self._model.to(device)
            logger.info("VLM OCR loaded: %s on %s", self.model_id, device)
        except Exception:
            logger.warning(
                "Failed to load VLM OCR model %s — run: python scripts/prepare_models.py --ocr",
                self.model_id, exc_info=True,
            )
            self._model = None
            self._load_failed = True

    # ── Florence-2 OCR (reuse existing model instance externally) ────────────

    def _extract_florence(self, image: Image.Image) -> str:
        # Use Florence-2 in <OCR> task mode. We call it directly rather than
        # loading a second Florence instance — the caller should pass the shared
        # instance via florence_ocr() helper instead.
        if self._model is None and not self._load_failed:
            self._load_florence_ocr()
        if self._model is None:
            return ""
        try:
            inputs = self._processor(
                text="<OCR>", images=image, return_tensors="pt"
            ).to(_get_device())
            import torch
            with torch.no_grad():
                ids = self._model.generate(**inputs, max_new_tokens=512)
            result = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
            # Strip Florence-2 task token
            return result.replace("<OCR>", "").replace("</s>", "").strip()
        except Exception:
            logger.warning("Florence OCR inference failed", exc_info=True)
            return ""

    def _load_florence_ocr(self) -> None:
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            device = _get_device()
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            import torch
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if settings.USE_FP16 else torch.float32,
                trust_remote_code=True,
                attn_implementation=_best_attn_impl(),
            ).to(device).eval()
            logger.info("Florence OCR loaded: %s", self.model_id)
        except Exception:
            logger.warning(
                "Failed to load Florence OCR model %s — run: python scripts/prepare_models.py --ocr",
                self.model_id, exc_info=True,
            )
            self._model = None
            self._load_failed = True


# ── Shared helpers ────────────────────────────────────────────────────────────


def _encode_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _get_device() -> str:
    cfg = settings.DEVICE.lower()
    try:
        import torch
        if cfg == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if cfg == "cuda" and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
