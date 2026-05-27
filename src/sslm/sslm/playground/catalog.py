from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    tasks: tuple[str, ...]
    num_fewshot: int = 0
    notes: str = ""


@dataclass(frozen=True)
class ModelProfile:
    key: str
    model_id: str
    family: str
    modality: str
    port: int
    image: str = "vllm/vllm-openai:latest"
    build_context: str | None = None
    dockerfile: str | None = None
    dtype: str = "bfloat16"
    # fp8 quantization halves weight memory: 8B params x 1 byte = 8 GB, fitting a 12 GB GPU.
    # Set to None to disable (requires >=16 GB VRAM for 8B bf16 models).
    quantization: str | None = "fp8"
    min_gpu_gb: int = 12
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.88
    max_num_seqs: int = 4
    extra_vllm_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    # git URL@branch for the vllm fork used in a custom build; enables wheel
    # cache invalidation in sidecars.py when the branch HEAD changes.
    vllm_source: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def vllm_command(self) -> list[str]:
        args = [
            "serve",
            self.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
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
        port=8010,
        image="sslm/zaya-vllm:latest",
        build_context=".",
        dockerfile="scripts/sslm/docker/Dockerfile.zaya-vllm",
        vllm_source="https://github.com/Zyphra/vllm.git@zaya1-pr",
        max_model_len=8192,
        gpu_memory_utilization=0.75,
        quantization="fp8",
        min_gpu_gb=12,
        extra_vllm_args=(
            "--mamba-cache-dtype",
            "float32",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "zaya_xml",
        ),
        notes=(
            "Zyphra reasoning MoE model. Requires Zyphra vLLM fork. "
            "temperature=1.0/top_p=0.95 recommended. fp8 for 12 GB GPU."
        ),
    ),
    "qwen3-8b": ModelProfile(
        key="qwen3-8b",
        model_id="Qwen/Qwen3-8B",
        family="qwen3",
        modality="reasoning_llm",
        port=8011,
        max_model_len=8192,
        gpu_memory_utilization=0.75,
        quantization="fp8",
        min_gpu_gb=12,
        extra_vllm_args=(
            "--enable-reasoning",
            "--reasoning-parser",
            "deepseek_r1",
        ),
        notes="Dense 8B reasoning baseline with thinking/non-thinking modes. fp8 for 12 GB GPU.",
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
        num_fewshot=0,
        notes=(
            "Open LLM Leaderboard v2: IFEval, BBH, MATH Lvl5, GPQA Diamond, MuSR, MMLU-Pro. "
            "~4-8 h per 8B model on a single 12 GB GPU."
        ),
    ),
    # Quick sanity check: fast tasks that finish in under 30 min per model.
    "reasoning_quick": BenchmarkSuite(
        name="reasoning_quick",
        tasks=("gsm8k", "arc_challenge", "hellaswag"),
        num_fewshot=0,
        notes="Fast iteration suite: GSM8K + ARC-Challenge + HellaSwag. ~20 min per model.",
    ),
    "reasoning_core": BenchmarkSuite(
        name="reasoning_core",
        tasks=("gsm8k", "gpqa_diamond", "mmlu_pro", "ifeval"),
        num_fewshot=0,
        notes="Core reasoning tasks: math, graduate science, knowledge, instruction following.",
    ),
    "math_code": BenchmarkSuite(
        name="math_code",
        tasks=("gsm8k", "minerva_math", "humaneval"),
        num_fewshot=0,
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
