from pathlib import Path

TASK_DIR = Path("scripts/sslm/lm-eval-tasks")


def test_quick_task_yamls_are_five_shot() -> None:
    for task in ("gsm8k_sslm", "arc_challenge_sslm", "nq_open_sslm"):
        text = (TASK_DIR / f"{task}.yaml").read_text(encoding="utf-8")
        assert "num_fewshot: 5" in text
