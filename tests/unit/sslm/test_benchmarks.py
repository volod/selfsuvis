from pathlib import Path

from sslm.playground.benchmarks import build_lm_eval_command


def test_lm_eval_command_targets_openai_compatible_endpoint() -> None:
    command = build_lm_eval_command(
        base_url="http://localhost:8010/v1",
        model_id="Zyphra/ZAYA1-8B",
        tasks=["gsm8k", "ifeval"],
        output_path=Path(".data/sslm/results"),
    )
    assert "local-chat-completions" in command
    model_args = command[command.index("--model_args") + 1]
    assert "model=Zyphra/ZAYA1-8B" in model_args
    assert "base_url=http://localhost:8010/v1" in model_args
    assert "--apply_chat_template" in command
