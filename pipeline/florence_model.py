"""Florence-2-large image captioning model wrapper.

Loaded at VideoIndexer.__init__() — if weights are missing the worker crashes on
startup rather than failing mid-mission.

caption_batch() contract:
    - Input:  List[PIL.Image.Image]
    - Output: List[Tuple[str, float]]  — (caption_text, confidence)
    - len(result) == len(input) always (stable ordering)
    - Per-image exception → ("", 0.5) for that index; never crashes the batch
    - confidence: mean of softmax(scores[i])[generated_token_id] across all generated
      tokens, clamped to [0.0, 1.0]. Falls back to 0.5 when scores unavailable.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import torch
from PIL import Image

from pipeline.config import settings
from pipeline.logging_utils import get_logger

_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"
_MODEL_ID = "microsoft/Florence-2-large"
_MODEL_BASE_NAME = "florence-2-large"

logger = get_logger(__name__)


class FlorenceModel:
    """Florence-2-large captioner. Load once; call caption_batch() many times."""

    def __init__(self) -> None:
        # Import here so the module is importable even without transformers installed
        # (unit tests that don't touch this class won't need the heavy deps).
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.41 is required for Florence-2 captioning. "
                "Install it with: pip install transformers>=4.41"
            ) from exc

        self.device = self._resolve_device()
        logger.info("Loading Florence-2-large on %s …", self.device)

        # Clear any leftover VRAM fragmentation before loading the model.
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # dtype: FP16 on CUDA, FP32 on CPU
        torch_dtype = torch.float16 if (self.device == "cuda" and settings.USE_FP16) else torch.float32

        self._processor = AutoProcessor.from_pretrained(
            _MODEL_ID, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self._model.eval()

        logger.info(
            "Florence-2-large ready (device=%s, dtype=%s)",
            self.device,
            torch_dtype,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def caption_batch(
        self,
        images: List[Image.Image],
        batch_size: int | None = None,
    ) -> List[Tuple[str, float]]:
        """Caption a list of PIL images.

        Returns a list of (caption, confidence) tuples, one per input image.
        Stable order: result[i] corresponds to images[i].
        Any per-image failure returns ("", 0.5) for that index.
        """
        if not images:
            return []

        effective_batch = batch_size if batch_size is not None else settings.FLORENCE_BATCH_SIZE
        results: List[Tuple[str, float]] = []

        for batch_start in range(0, len(images), effective_batch):
            batch = images[batch_start : batch_start + effective_batch]
            batch_results = self._caption_batch_chunk(batch, effective_batch)
            results.extend(batch_results)

        return results

    @property
    def model_tag(self) -> str:
        """Structured provenance string stored in frames.caption_model.

        Format: "{model}:{prompt_version}:{precision}"
        Example: "florence-2-large:v1:fp16"

        Bump FLORENCE_PROMPT_VERSION (env var) whenever the task prompt or
        post-processing changes so that existing captions remain distinguishable
        from ones generated under a different prompt.
        """
        precision = "fp16" if (self.device == "cuda" and settings.USE_FP16) else "fp32"
        return f"{_MODEL_BASE_NAME}:{settings.FLORENCE_PROMPT_VERSION}:{precision}"

    # ── internals ─────────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        if settings.DEVICE == "cpu":
            return "cpu"
        if settings.DEVICE == "cuda":
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _caption_batch_chunk(
        self,
        images: List[Image.Image],
        batch_size: int,
    ) -> List[Tuple[str, float]]:
        """Caption one chunk, with OOM fallback to batch=1."""
        try:
            return self._run_inference(images)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch_size > 1:
                logger.warning(
                    "Florence OOM on batch_size=%d; falling back to batch=1", batch_size
                )
                torch.cuda.empty_cache()
                # Process images one by one
                return [self._caption_single(img) for img in images]
            raise

    def _run_inference(self, images: List[Image.Image]) -> List[Tuple[str, float]]:
        """Run Florence inference on a list of images (same batch)."""
        prompts = [_TASK_PROMPT] * len(images)

        inputs = self._processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=256,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        # Decode captions
        sequences = generated.sequences
        # Strip the input prompt tokens from each generated sequence
        input_ids_len = inputs["input_ids"].shape[1]
        generated_ids = sequences[:, input_ids_len:]

        decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        # Post-process removes task prompt prefix that Florence sometimes echoes
        captions = []
        for raw, img in zip(decoded, images):
            parsed = self._processor.post_process_generation(
                raw,
                task=_TASK_PROMPT,
                image_size=(img.width, img.height),
            )
            text = parsed.get(_TASK_PROMPT, raw)
            if isinstance(text, str):
                captions.append(text.strip())
            else:
                captions.append("")

        # Compute per-image confidence from generation scores
        confidences = _compute_confidences(generated.scores, generated_ids)

        return list(zip(captions, confidences))

    def _caption_single(self, image: Image.Image) -> Tuple[str, float]:
        """Caption one image, returning ("", 0.5) on any error."""
        try:
            results = self._run_inference([image])
            return results[0]
        except Exception:
            logger.warning("Florence failed on a single image; returning empty caption", exc_info=True)
            return ("", 0.5)


# ── confidence computation ────────────────────────────────────────────────────


def _compute_confidences(
    scores: tuple | None,
    generated_ids: torch.Tensor,
) -> List[float]:
    """Compute mean token probability for each sequence in the batch.

    scores: tuple of (vocab_size,) or (batch, vocab_size) tensors, one per step.
    generated_ids: (batch, seq_len) tensor of chosen token ids.

    Returns a list of floats in [0.0, 1.0], one per batch item.
    Falls back to 0.5 when scores are unavailable or seq_len is zero.
    """
    batch_size = generated_ids.shape[0]

    if scores is None or len(scores) == 0:
        return [0.5] * batch_size

    seq_len = generated_ids.shape[1]
    if seq_len == 0:
        return [0.5] * batch_size

    # Accumulate per-token probabilities for each sequence
    # sum_probs[b] sums the chosen-token probability across all non-padding positions
    sum_probs = [0.0] * batch_size
    count = [0] * batch_size

    for step_idx, step_logits in enumerate(scores):
        if step_idx >= seq_len:
            break
        # step_logits: (batch, vocab_size)
        if step_logits.dim() == 1:
            # single-item batch flattened — expand
            step_logits = step_logits.unsqueeze(0)

        probs = torch.softmax(step_logits.float(), dim=-1)  # (batch, vocab)
        chosen_ids = generated_ids[:, step_idx]            # (batch,)

        for b in range(batch_size):
            tok = chosen_ids[b].item()
            if tok == 1:  # padding / EOS token typically 1 for Florence
                continue
            prob = probs[b, tok].item()
            sum_probs[b] += prob
            count[b] += 1

    confidences = []
    for b in range(batch_size):
        if count[b] == 0:
            confidences.append(0.5)
        else:
            raw = sum_probs[b] / count[b]
            confidences.append(float(max(0.0, min(1.0, raw))))

    return confidences
