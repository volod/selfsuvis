import pandas as pd
from sslm.dashboard.app import _leaderboard


def test_leaderboard_uses_latest_display_task_score() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "qwen3-8b",
                "task": "gsm8k",
                "task_display": "GSM8K",
                "score": 0.0,
                "date": 10,
            },
            {
                "model": "qwen3-8b",
                "task": "gsm8k_sslm",
                "task_display": "GSM8K",
                "score": 0.8,
                "date": 20,
            },
            {
                "model": "qwen3-8b",
                "task": "arc_challenge_sslm",
                "task_display": "ARC-C",
                "score": 0.7,
                "date": 20,
            },
        ]
    )

    board = _leaderboard(df)

    assert board.loc["qwen3-8b", "GSM8K"] == 80.0
    assert board.loc["qwen3-8b", "ARC-C"] == 70.0
    assert board.loc["qwen3-8b", "Avg"] == 75.0
