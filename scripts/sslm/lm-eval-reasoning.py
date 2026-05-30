#!/usr/bin/env python3
"""lm-eval entry-point wrapper for reasoning models.

Patches applied before lm-eval starts:

1. LocalChatCompletion._create_payload: injects per-model request overrides read
   from env vars set by the orchestrator. This is the dict that is POSTed
   verbatim, so it works for both sync and async transports:
     - SSLM_DISABLE_THINKING=1   -> chat_template_kwargs={"enable_thinking": False}
       (Qwen3 non-thinking mode; vLLM forwards this to the chat template).
     - SSLM_FORCE_TEMPERATURE=<f>, SSLM_FORCE_TOP_P=<f>, SSLM_FORCE_TOP_K=<n>
       -> sampling overrides for models whose recommended decoding differs
       from lm-eval task defaults.

2. LocalChatCompletion.parse_generations: content fallback chain
   content -> reasoning_content -> reasoning (the field name varies by vLLM
   version) plus light normalisation (strip markdown $ * #, unwrap \\boxed{})
   so the task answer-extraction regexes see clean text.

3. Noisy-warning filters for expected, benign lm-eval WARNINGs.
"""
import logging
import os
import re
import sys

_logger = logging.getLogger(__name__)

# Per-process request counter; resets when a new lm_eval subprocess starts.
_req_count = [0]

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _normalize_content(text: str) -> str:
    """Strip LaTeX/markdown noise that breaks task answer-extraction regexes.

    gsm8k flexible-extract grabs the last number-like token; LaTeX wrappers like
    `$$ ... $$` and `\\boxed{18}` otherwise hide or mismatch the real number.
    """
    if not text:
        return text
    text = _BOXED_RE.sub(r"\1", text)   # \boxed{18} -> 18
    text = text.replace("$", " ")        # drop $ / $$ math delimiters
    text = text.replace("*", "")         # markdown bold/italic
    text = text.replace("#", " ")        # markdown headers / stray hashes
    return text


def _patch_create_payload() -> None:
    from lm_eval.models.openai_completions import LocalChatCompletion

    _orig = LocalChatCompletion._create_payload

    def _create_payload(self, messages, **kwargs):
        payload = _orig(self, messages, **kwargs)
        force_temp = os.environ.get("SSLM_FORCE_TEMPERATURE")
        if force_temp is not None:
            payload["temperature"] = float(force_temp)
        force_top_p = os.environ.get("SSLM_FORCE_TOP_P")
        if force_top_p is not None:
            payload["top_p"] = float(force_top_p)
        force_top_k = os.environ.get("SSLM_FORCE_TOP_K")
        if force_top_k is not None:
            payload["top_k"] = int(force_top_k)
        if os.environ.get("SSLM_DISABLE_THINKING"):
            # vLLM reads chat_template_kwargs and forwards to the chat template.
            ctk = dict(payload.get("chat_template_kwargs") or {})
            ctk["enable_thinking"] = False
            payload["chat_template_kwargs"] = ctk
        return payload

    LocalChatCompletion._create_payload = _create_payload


def _patch_tiktoken() -> None:
    """Fall back to cl100k_base for any model name tiktoken doesn't recognise."""
    import tiktoken.model as _tm

    _original = _tm.encoding_name_for_model

    def _patched(model_name: str) -> str:
        try:
            return _original(model_name)
        except KeyError:
            return "cl100k_base"

    _tm.encoding_name_for_model = _patched


def _patch_parse_generations() -> None:
    from lm_eval.models.openai_completions import LocalChatCompletion

    @staticmethod  # type: ignore[misc]
    def parse_generations(outputs, **kwargs):
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                tmp = [None] * len(out["choices"])
                for choice in out["choices"]:
                    _req_count[0] += 1
                    n = _req_count[0]
                    msg = choice["message"]
                    content = msg.get("content")
                    if content is None:
                        # vLLM field name for reasoning output varies by version.
                        content = msg.get("reasoning_content") or msg.get("reasoning")
                        if content is None:
                            other = {k: repr(v)[:120] for k, v in msg.items()
                                     if k not in ("content", "reasoning_content", "reasoning")}
                            _logger.warning(
                                f"[req {n}] content=null reasoning=null; "
                                f"finish_reason={choice.get('finish_reason')!r} "
                                f"msg_keys={list(msg.keys())} other={other}"
                            )
                        else:
                            _logger.info(
                                f"[req {n}] content=null; using reasoning fallback "
                                f"({len(content)} chars finish={choice.get('finish_reason')!r})"
                            )
                    if content is not None:
                        content = _normalize_content(content)
                    tmp[choice["index"]] = content
            except Exception as exc:
                _logger.warning(f"Could not parse generations: {exc}")
                tmp = [""]
            res += tmp
        return res

    LocalChatCompletion.parse_generations = parse_generations


def _suppress_noisy_warnings() -> None:
    """Filter expected benign lm-eval WARNINGs that add no diagnostic value."""
    benign = [
        ("lm_eval.models.api_models", "API returned null content"),
        ("lm_eval.models.api_models", "Batch size > 1 detected"),
        ("lm_eval.models.openai_completions", "does not support batching"),
        ("lm_eval.evaluator", "Overwriting default num_fewshot"),
        ("lm_eval.evaluator", "generation_kwargs"),
    ]
    for logger_name, fragment in benign:
        logging.getLogger(logger_name).addFilter(
            lambda r, f=fragment: f not in r.getMessage()
        )


_patch_create_payload()
_patch_tiktoken()
_patch_parse_generations()
_suppress_noisy_warnings()

# Suppress the "SHOULD ONLY BE USED FOR TESTING" warning emitted whenever --limit
# is set; expected for quick/dev runs.
logging.getLogger("lm_eval.config.evaluate_config").addFilter(
    lambda r: "SHOULD ONLY BE USED FOR TESTING" not in r.getMessage()
)

from lm_eval.__main__ import cli_evaluate  # noqa: E402

sys.exit(cli_evaluate())
