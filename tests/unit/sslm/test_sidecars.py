import yaml
from sslm.playground.catalog import MODEL_CATALOG
from sslm.playground.constants import QWEN3_4B_FP8_PORT, QWEN3_8B_PORT, VLLM_CONTAINER_PORT
from sslm.playground.sidecars import render_compose


def test_render_compose_has_gpu_reservation() -> None:
    document = yaml.safe_load(render_compose([MODEL_CATALOG["qwen3-8b"]]))
    service = document["services"]["sslm_qwen3_8b"]
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert service["entrypoint"] == ["vllm"]
    assert service["command"][:2] == ["serve", "Qwen/Qwen3-8B"]
    assert "deepseek_r1" in service["command"]
    assert service["gpus"] == "all"
    assert devices[0]["driver"] == "nvidia"
    assert devices[0]["capabilities"] == ["gpu"]
    assert service["ports"] == [f"{QWEN3_8B_PORT}:{VLLM_CONTAINER_PORT}"]


def test_zaya_compose_uses_custom_build() -> None:
    document = yaml.safe_load(render_compose([MODEL_CATALOG["zaya1-8b"]]))
    service = document["services"]["sslm_zaya1_8b"]
    assert service["build"]["dockerfile"] == "scripts/sslm/docker/Dockerfile.zaya-vllm"
    assert service["build"]["context"] == "${SSLM_PROJECT_ROOT:-.}"


def test_qwen3_4b_fp8_compose_uses_prequantized_checkpoint() -> None:
    document = yaml.safe_load(render_compose([MODEL_CATALOG["qwen3-4b-fp8"]]))
    service = document["services"]["sslm_qwen3_4b_fp8"]
    assert service["command"][:2] == ["serve", "Qwen/Qwen3-4B-FP8"]
    assert "--quantization" not in service["command"]
    assert service["ports"] == [f"{QWEN3_4B_FP8_PORT}:{VLLM_CONTAINER_PORT}"]
