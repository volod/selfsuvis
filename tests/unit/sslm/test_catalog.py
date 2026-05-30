from sslm.playground.catalog import BENCHMARK_SUITES, DEFAULT_MODEL_PAIR, MODEL_CATALOG
from sslm.playground.constants import (
    QUICK_SUITE_NUM_FEWSHOT,
    QWEN3_4B_FP8_GPU_MEMORY_UTILIZATION,
    QWEN3_4B_FP8_MAX_MODEL_LEN,
    QWEN3_4B_FP8_MAX_NUM_SEQS,
    QWEN3_4B_FP8_MIN_GPU_GB,
    QWEN3_8B_MIN_GPU_GB,
    ZAYA_MIN_GPU_GB,
    ZAYA_TEMPERATURE,
    ZAYA_TOP_K,
    ZAYA_TOP_P,
)


def test_default_model_pair_is_configured() -> None:
    assert DEFAULT_MODEL_PAIR == ("zaya1-8b", "qwen3-8b")
    assert MODEL_CATALOG["zaya1-8b"].model_id == "Zyphra/ZAYA1-8B"
    assert MODEL_CATALOG["qwen3-8b"].model_id == "Qwen/Qwen3-8B"


def test_zaya_uses_required_reasoning_args() -> None:
    command = MODEL_CATALOG["zaya1-8b"].vllm_command()
    assert "--reasoning-parser" in command
    assert "qwen3" in command
    assert "--tool-call-parser" in command
    assert "zaya_xml" in command


def test_zaya_uses_recommended_sampling_overrides() -> None:
    model = MODEL_CATALOG["zaya1-8b"]
    assert model.force_temperature == ZAYA_TEMPERATURE
    assert model.force_top_p == ZAYA_TOP_P
    assert model.force_top_k == ZAYA_TOP_K


def test_zaya_defaults_to_bf16_serving() -> None:
    model = MODEL_CATALOG["zaya1-8b"]
    assert model.quantization is None
    assert model.min_gpu_gb == ZAYA_MIN_GPU_GB
    assert model.fallback_key == "qwen3-4b-fp8"


def test_qwen3_4b_fp8_is_compact_fallback() -> None:
    model = MODEL_CATALOG["qwen3-4b-fp8"]
    assert model.model_id == "Qwen/Qwen3-4B-FP8"
    assert model.quantization is None
    assert model.min_gpu_gb == QWEN3_4B_FP8_MIN_GPU_GB
    assert model.max_model_len == QWEN3_4B_FP8_MAX_MODEL_LEN
    assert model.gpu_memory_utilization == QWEN3_4B_FP8_GPU_MEMORY_UTILIZATION
    assert model.max_num_seqs == QWEN3_4B_FP8_MAX_NUM_SEQS
    assert model.disable_thinking is True


def test_qwen_remains_12gb_profile() -> None:
    model = MODEL_CATALOG["qwen3-8b"]
    assert model.quantization == "fp8"
    assert model.min_gpu_gb == QWEN3_8B_MIN_GPU_GB


def test_reasoning_quick_is_five_shot() -> None:
    assert BENCHMARK_SUITES["reasoning_quick"].num_fewshot == QUICK_SUITE_NUM_FEWSHOT
