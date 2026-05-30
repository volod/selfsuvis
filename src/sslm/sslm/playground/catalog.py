from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sslm.playground.constants import (
    DEFAULT_LM_EVAL_NUM_FEWSHOT,
    DEFAULT_MODEL_DTYPE,
    DEFAULT_MODEL_GPU_MEMORY_UTILIZATION,
    DEFAULT_MODEL_MAX_LEN,
    DEFAULT_MODEL_MAX_NUM_SEQS,
    DEFAULT_MODEL_MIN_GPU_GB,
    DEFAULT_MODEL_QUANTIZATION,
    DEFAULT_VLLM_IMAGE,
    QUICK_SUITE_LIMIT,
    QUICK_SUITE_NUM_FEWSHOT,
    QWEN3_4B_FP8_GPU_MEMORY_UTILIZATION,
    QWEN3_4B_FP8_MAX_MODEL_LEN,
    QWEN3_4B_FP8_MAX_NUM_SEQS,
    QWEN3_4B_FP8_MIN_GPU_GB,
    QWEN3_4B_FP8_PORT,
    QWEN3_8B_GPU_MEMORY_UTILIZATION,
    QWEN3_8B_MAX_MODEL_LEN,
    QWEN3_8B_MIN_GPU_GB,
    QWEN3_8B_PORT,
    VLLM_CONTAINER_PORT,
    VLLM_HOST,
    ZAYA_GPU_MEMORY_UTILIZATION,
    ZAYA_MAX_MODEL_LEN,
    ZAYA_MIN_GPU_GB,
    ZAYA_PORT,
    ZAYA_TEMPERATURE,
    ZAYA_TOP_K,
    ZAYA_TOP_P,
)


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    tasks: tuple[str, ...]
    num_fewshot: int = DEFAULT_LM_EVAL_NUM_FEWSHOT
    limit: int | None = None
    # Extra lm-eval gen_kwargs string (e.g. "max_gen_toks=<tokens>") passed via --gen_kwargs.
    # For reasoning models, cap max_gen_toks so <think> completes within budget and
    # the final answer ends up in content (not lost to an exhausted token limit).
    gen_kwargs: str | None = None
    log_samples: bool = False
    notes: str = ""


@dataclass(frozen=True)
class ModelProfile:
    key: str
    model_id: str
    family: str
    modality: str
    port: int
    image: str = DEFAULT_VLLM_IMAGE
    build_context: str | None = None
    dockerfile: str | None = None
    dtype: str = DEFAULT_MODEL_DTYPE
    # fp8 quantization halves weight memory for dense 8B models. Set to None for
    # models where dynamic fp8 is unsupported or produces invalid generations.
    quantization: str | None = DEFAULT_MODEL_QUANTIZATION
    min_gpu_gb: int = DEFAULT_MODEL_MIN_GPU_GB
    max_model_len: int = DEFAULT_MODEL_MAX_LEN
    gpu_memory_utilization: float = DEFAULT_MODEL_GPU_MEMORY_UTILIZATION
    max_num_seqs: int = DEFAULT_MODEL_MAX_NUM_SEQS
    extra_vllm_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    # git URL@branch for the vllm fork used in a custom build; enables wheel
    # cache invalidation in sidecars.py when the branch HEAD changes.
    vllm_source: str | None = None
    # Force this temperature for every lm-eval request, overriding task-level
    # defaults (e.g. gsm8k hardcodes temperature=0). Required for Mamba-based
    # models (ZAYA) where greedy decoding causes repetition loops.
    force_temperature: float | None = None
    force_top_p: float | None = None
    force_top_k: int | None = None
    fallback_key: str | None = None
    # Inject enable_thinking=False into every API request so the model answers
    # directly without a thinking chain. Prevents token-budget exhaustion for
    # reasoning models where the <think> block fills max_gen_toks before the
    # final answer is produced.
    disable_thinking: bool = False

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def vllm_command(self) -> list[str]:
        args = [
            "serve",
            self.model_id,
            "--host",
            VLLM_HOST,
            "--port",
            str(VLLM_CONTAINER_PORT),
            "--dtype",
            self.dtype,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--trust-remote-code",
        ]
        if self.quantization:
            args += ["--quantization", self.quantization]
        args.extend(self.extra_vllm_args)
        return args


MODEL_CATALOG: dict[str, ModelProfile] = {
    "zaya1-8b": ModelProfile(
        key="zaya1-8b",
        model_id="Zyphra/ZAYA1-8B",
        family="zaya",
        modality="reasoning_llm",
        port=ZAYA_PORT,
        image="sslm/zaya-vllm:latest",
        build_context=".",
        dockerfile="scripts/sslm/docker/Dockerfile.zaya-vllm",
        vllm_source="https://github.com/Zyphra/vllm.git@zaya1-pr",
        max_model_len=ZAYA_MAX_MODEL_LEN,
        gpu_memory_utilization=ZAYA_GPU_MEMORY_UTILIZATION,
        quantization=None,
        min_gpu_gb=ZAYA_MIN_GPU_GB,
        extra_vllm_args=(
            "--mamba-cache-dtype",
            "float32",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "zaya_xml",
        ),
        env={
            # zaya1-pr wheel pins flashinfer 0.6.8 while the vllm base image ships
            # flashinfer 0.6.11; the version guard raises on startup. Bypass it
            # because the functional APIs used by this model are compatible.
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
        force_temperature=ZAYA_TEMPERATURE,
        force_top_p=ZAYA_TOP_P,
        force_top_k=ZAYA_TOP_K,
        notes=(
            "Zyphra reasoning MoE model. Requires Zyphra vLLM fork. "
            "Use bf16 serving; fp8 produced invalid token output in local smoke runs."
        ),
        fallback_key="qwen3-4b-fp8",
    ),
    "qwen3-4b-fp8": ModelProfile(
        key="qwen3-4b-fp8",
        model_id="Qwen/Qwen3-4B-FP8",
        family="qwen3",
        modality="reasoning_llm",
        port=QWEN3_4B_FP8_PORT,
        max_model_len=QWEN3_4B_FP8_MAX_MODEL_LEN,
        gpu_memory_utilization=QWEN3_4B_FP8_GPU_MEMORY_UTILIZATION,
        max_num_seqs=QWEN3_4B_FP8_MAX_NUM_SEQS,
        # Pre-quantized FP8 checkpoint; do not add dynamic --quantization.
        quantization=None,
        min_gpu_gb=QWEN3_4B_FP8_MIN_GPU_GB,
        extra_vllm_args=(
            "--reasoning-parser",
            "deepseek_r1",
        ),
        disable_thinking=True,
        notes=(
            "Compact Qwen3 FP8 fallback for GPUs that cannot run ZAYA1 BF16. "
            f"Keep serving concurrency conservative for {QUICK_SUITE_NUM_FEWSHOT}-shot quick prompts."
        ),
    ),
    "qwen3-8b": ModelProfile(
        key="qwen3-8b",
        model_id="Qwen/Qwen3-8B",
        family="qwen3",
        modality="reasoning_llm",
        port=QWEN3_8B_PORT,
        max_model_len=QWEN3_8B_MAX_MODEL_LEN,
        gpu_memory_utilization=QWEN3_8B_GPU_MEMORY_UTILIZATION,
        quantization=DEFAULT_MODEL_QUANTIZATION,
        min_gpu_gb=QWEN3_8B_MIN_GPU_GB,
        extra_vllm_args=(
            "--reasoning-parser",
            "deepseek_r1",
        ),
        disable_thinking=True,
        notes=(
            "Dense 8B reasoning baseline with thinking/non-thinking modes. "
            f"fp8 for {QWEN3_8B_MIN_GPU_GB} GB GPU."
        ),
    ),
}

DEFAULT_MODEL_PAIR = ("zaya1-8b", "qwen3-8b")

BENCHMARK_SUITES: dict[str, BenchmarkSuite] = {
    "smoke": BenchmarkSuite(
        name="smoke",
        tasks=(),
        notes="Local prompt smoke test; no external datasets.",
    ),
    # Open LLM Leaderboard v2 (HuggingFace) — the canonical public reasoning benchmark.
    # Task names are lm-evaluation-harness leaderboard task group identifiers.
    "open_llm_v2": BenchmarkSuite(
        name="open_llm_v2",
        tasks=(
            "leaderboard_ifeval",
            "leaderboard_bbh",
            "leaderboard_math_hard",
            "leaderboard_gpqa",
            "leaderboard_musr",
            "leaderboard_mmlu_pro",
        ),
        num_fewshot=DEFAULT_LM_EVAL_NUM_FEWSHOT,
        notes=(
            "Open LLM Leaderboard v2: IFEval, BBH, MATH Lvl5, GPQA Diamond, MuSR, MMLU-Pro. "
            "~4-8 h per 8B model on a single GPU."
        ),
    ),
    # Quick sanity check: fast tasks that finish in under 30 min per model.
    # Only generate_until tasks are used here -- loglikelihood (multiple_choice)
    # is not supported by the chat-completions endpoint.
    "reasoning_quick": BenchmarkSuite(
        name="reasoning_quick",
        # Custom *_sslm variants (scripts/sslm/lm-eval-tasks) with scoring robust
        # to chatty reasoning-model output. The built-in chat tasks score 0 here:
        # arc remove_whitespace can't pull a letter from a sentence, nq exact-match
        # never matches a sentence to a short entity, gsm8k flexible-extract grabs
        # "$$" from LaTeX. Each YAML carries its own generation_kwargs (stop
        # sequences + max_gen_toks); we deliberately do NOT set a suite-level
        # gen_kwargs override, since overriding 'until' would clobber the per-task
        # stop sequences that keep answers extractable.
        tasks=("gsm8k_sslm", "arc_challenge_sslm", "nq_open_sslm"),
        num_fewshot=QUICK_SUITE_NUM_FEWSHOT,
        limit=QUICK_SUITE_LIMIT,
        gen_kwargs=None,
        log_samples=True,
        notes=(
            f"Fast {QUICK_SUITE_NUM_FEWSHOT}-shot sanity check: "
            "GSM8K + ARC-Challenge + NQ-Open (sslm variants). "
            f"{QUICK_SUITE_LIMIT} samples/task."
        ),
    ),
    "reasoning_core": BenchmarkSuite(
        name="reasoning_core",
        tasks=("gsm8k", "gpqa_diamond", "mmlu_pro", "ifeval"),
        num_fewshot=DEFAULT_LM_EVAL_NUM_FEWSHOT,
        notes="Core reasoning tasks: math, graduate science, knowledge, instruction following.",
    ),
    "math_code": BenchmarkSuite(
        name="math_code",
        tasks=("gsm8k", "minerva_math", "humaneval"),
        num_fewshot=DEFAULT_LM_EVAL_NUM_FEWSHOT,
        notes="Math and code: GSM8K, Minerva, HumanEval.",
    ),
}


def get_model(key: str) -> ModelProfile:
    try:
        return MODEL_CATALOG[key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_CATALOG))
        raise SystemExit(f"Unknown model key {key!r}. Known models: {known}") from exc


def model_table() -> list[dict[str, Any]]:
    return [
        {
            "key": model.key,
            "model_id": model.model_id,
            "family": model.family,
            "modality": model.modality,
            "port": model.port,
            "base_url": model.base_url,
            "notes": model.notes,
        }
        for model in MODEL_CATALOG.values()
    ]
