import os
import subprocess
import sys
from pathlib import Path

from nanochat.training.report import Report
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
    assert "--load-optimizer" in help_text
    assert "recommended for clean SFT" in help_text
    assert "--resume-from-step" in help_text
    assert "--save-every" in help_text


def test_report_generate_accepts_local_run_sections(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "header.md").write_text(
        "# nanochat training report\n\n"
        "### Bloat\n"
        "- Lines: 10\n\n"
        "Run started: 2026-06-08 12:00:00\n\n",
        encoding="utf-8",
    )
    for file_name, content in {
        "tokenizer-training.md": "## Tokenizer training\ntimestamp: 2026-06-08 12:01:00\n\n",
        "tokenizer-evaluation.md": "## Tokenizer evaluation\ntimestamp: 2026-06-08 12:02:00\n\n",
        "base-model-training.md": "## Base model training\ntimestamp: 2026-06-08 13:00:00\n\n",
        "base-model-evaluation.md": (
            "## Base model evaluation\n"
            "timestamp: 2026-06-08 14:00:00\n\n"
            "- CORE: 0.1234\n"
        ),
        "sft.md": "## SFT\ntimestamp: 2026-06-08 15:00:00\n\n",
        "chat-evaluation-sft.md": (
            "## Chat evaluation sft\n"
            "timestamp: 2026-06-08 16:00:00\n\n"
            "- ARC-Easy: 0.4000\n"
            "- ChatCORE metric: 0.2500\n"
        ),
    }.items():
        (report_dir / file_name).write_text(content, encoding="utf-8")

    Report(str(report_dir)).generate()

    stdout = capsys.readouterr().out
    assert "does not exist" not in stdout
    report = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "## SFT" in report
    assert "| Metric" in report
    assert "RL" not in report.split("## Summary", 1)[1].splitlines()[4]
    assert (tmp_path / "report.md").exists()


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
