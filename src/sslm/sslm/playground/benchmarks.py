from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sslm.playground.client import ChatRequest, OpenAICompatibleClient

# Primary metric key (as written by lm-eval in results.json) for each task.
# Used by the dashboard to extract a single representative score per task.
TASK_PRIMARY_METRIC: dict[str, str] = {
    # Open LLM Leaderboard v2
    "leaderboard_ifeval": "prompt_level_strict_acc,none",
    "leaderboard_bbh": "acc_norm,none",
    "leaderboard_math_hard": "exact_match,none",
    "leaderboard_gpqa": "acc_norm,none",
    "leaderboard_musr": "acc_norm,none",
    "leaderboard_mmlu_pro": "acc,none",
    # Standalone tasks
    "gsm8k": "exact_match,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "gpqa_diamond": "acc_norm,none",
    "mmlu_pro": "acc,none",
    "ifeval": "prompt_level_strict_acc,none",
    "minerva_math": "exact_match,none",
    "humaneval": "pass@1,none",
    "mbpp": "pass@1,none",
}

# Short display name for each task (used in the dashboard).
TASK_DISPLAY_NAME: dict[str, str] = {
    "leaderboard_ifeval": "IFEval",
    "leaderboard_bbh": "BBH",
    "leaderboard_math_hard": "MATH",
    "leaderboard_gpqa": "GPQA*",
    "leaderboard_musr": "MuSR",
    "leaderboard_mmlu_pro": "MMLU-Pro",
    "gsm8k": "GSM8K",
    "arc_challenge": "ARC-C",
    "hellaswag": "HellaSwag",
    "gpqa_diamond": "GPQA*",
    "mmlu_pro": "MMLU-Pro",
    "ifeval": "IFEval",
    "minerva_math": "Minerva",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
}


@dataclass(frozen=True)
class SmokePrompt:
    name: str
    prompt: str
    expected_hint: str | None = None


SMOKE_PROMPTS = (
    SmokePrompt(
        name="arithmetic_reasoning",
        prompt="A train travels 60 miles in 1.5 hours, then 40 miles in 1 hour. What is the average speed for the whole trip?",
        expected_hint="40",
    ),
    SmokePrompt(
        name="code_reasoning",
        prompt="Write a Python expression that returns the sorted unique values from [3, 1, 3, 2].",
        expected_hint="sorted",
    ),
    SmokePrompt(
        name="planning",
        prompt="Give three concrete checks before launching a GPU benchmark sidecar on a shared workstation.",
        expected_hint="GPU",
    ),
)


@dataclass
class SmokeResult:
    model: str
    prompt_name: str
    latency_s: float
    ok: bool
    response_text: str


def extract_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def run_smoke(
    *,
    base_url: str,
    model_id: str,
    output_path: Path,
    max_tokens: int = 512,
) -> list[SmokeResult]:
    client = OpenAICompatibleClient(base_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[SmokeResult] = []
    for prompt in SMOKE_PROMPTS:
        started = time.perf_counter()
        response = client.chat(ChatRequest(model=model_id, prompt=prompt.prompt, max_tokens=max_tokens))
        latency_s = time.perf_counter() - started
        text = extract_text(response)
        ok = bool(text.strip())
        if prompt.expected_hint:
            ok = ok and prompt.expected_hint.lower() in text.lower()
        results.append(
            SmokeResult(
                model=model_id,
                prompt_name=prompt.name,
                latency_s=latency_s,
                ok=ok,
                response_text=text,
            )
        )
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return results


def build_lm_eval_command(
    *,
    base_url: str,
    model_id: str,
    tasks: list[str],
    output_path: Path,
    num_fewshot: int = 0,
    batch_size: str = "4",
) -> list[str]:
    model_args = f"model={model_id},base_url={base_url},tokenizer_backend=huggingface"
    return [
        "lm_eval",
        "--model",
        "local-chat-completions",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        batch_size,
        "--output_path",
        str(output_path),
        "--apply_chat_template",
    ]


def run_lm_eval(
    *,
    base_url: str,
    model_id: str,
    tasks: list[str],
    output_path: Path,
    num_fewshot: int = 0,
    batch_size: str = "4",
) -> int:
    output_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    command = build_lm_eval_command(
        base_url=base_url,
        model_id=model_id,
        tasks=tasks,
        output_path=output_path,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )
    return subprocess.call(command, env=env)


def _best_score(task_metrics: dict) -> tuple[str, float] | tuple[None, None]:
    """Return (metric_key, value) using TASK_PRIMARY_METRIC then common fallbacks."""
    for key in ("acc_norm,none", "exact_match,none", "acc,none", "pass@1,none"):
        if key in task_metrics:
            return key, float(task_metrics[key])
    return None, None


def parse_lm_eval_results(results_dir: Path) -> list[dict]:
    """Scan results_dir recursively for lm-eval results.json files.

    Returns a flat list of dicts with keys:
        model, task, task_display, metric, score, date, result_file
    """
    rows: list[dict] = []
    for json_file in sorted(results_dir.rglob("results.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "results" not in data:
            continue

        # Infer the model key from the directory structure under results_dir.
        try:
            rel_parts = json_file.relative_to(results_dir).parts
            model_key = rel_parts[0]
        except ValueError:
            model_key = json_file.parent.parent.name

        date_str = data.get("date", "") or data.get("config_general", {}).get("date", "")

        for task_name, task_metrics in data["results"].items():
            primary_key = TASK_PRIMARY_METRIC.get(task_name)
            if primary_key and primary_key in task_metrics:
                metric_key = primary_key
                score = float(task_metrics[primary_key])
            else:
                metric_key, score = _best_score(task_metrics)

            if score is None:
                continue

            rows.append(
                {
                    "model": model_key,
                    "task": task_name,
                    "task_display": TASK_DISPLAY_NAME.get(task_name, task_name),
                    "metric": metric_key,
                    "score": score,
                    "date": date_str,
                    "result_file": str(json_file),
                }
            )
    return rows
