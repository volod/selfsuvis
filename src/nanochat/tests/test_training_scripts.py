import os
import subprocess
import sys
from pathlib import Path

from nanochat.training.schedule import eval_step_count, should_eval_base, should_eval_sft

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_help(module_name: str) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_base_train_help_exposes_eval_controls():
    help_text = run_help("scripts.base_train")

    assert "Pretrain base model" in help_text
    assert "--eval-at-start" in help_text
    assert "--eval-tokens" in help_text
    assert "--loader-buffer-size" in help_text


def test_chat_sft_help_exposes_eval_and_resume_controls():
    help_text = run_help("scripts.chat_sft")

    assert "Supervised fine-tuning (SFT) the model" in help_text
    assert "--eval-at-start" in help_text
    assert "--eval-tokens" in help_text
    assert "--resume-from-step" in help_text
    assert "--save-every" in help_text


def test_base_eval_schedule_skips_step_zero_by_default():
    assert not should_eval_base(step=0, eval_every=250, eval_at_start=False, last_step=False)
    assert should_eval_base(step=0, eval_every=250, eval_at_start=True, last_step=False)
    assert should_eval_base(step=250, eval_every=250, eval_at_start=False, last_step=False)
    assert should_eval_base(step=499, eval_every=250, eval_at_start=False, last_step=True)
    assert not should_eval_base(step=499, eval_every=-1, eval_at_start=False, last_step=True)


def test_sft_eval_schedule_keeps_final_eval_when_periodic_disabled():
    assert not should_eval_sft(step=0, eval_every=200, eval_at_start=False, last_step=False)
    assert should_eval_sft(step=0, eval_every=200, eval_at_start=True, last_step=False)
    assert should_eval_sft(step=200, eval_every=200, eval_at_start=False, last_step=False)
    assert should_eval_sft(step=999, eval_every=-1, eval_at_start=False, last_step=True)


def test_eval_step_count_never_drops_below_one():
    assert (
        eval_step_count(
            eval_tokens=128,
            device_batch_size=4,
            max_seq_len=1024,
            ddp_world_size=1,
        )
        == 1
    )
    assert (
        eval_step_count(
            eval_tokens=4 * 1024 * 8,
            device_batch_size=4,
            max_seq_len=1024,
            ddp_world_size=2,
        )
        == 4
    )
