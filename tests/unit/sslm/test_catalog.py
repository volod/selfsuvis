from sslm.playground.catalog import DEFAULT_MODEL_PAIR, MODEL_CATALOG


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
