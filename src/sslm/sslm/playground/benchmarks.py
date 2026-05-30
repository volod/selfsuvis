from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sslm.playground.client import ChatRequest, OpenAICompatibleClient
from sslm.playground.constants import (
    DEFAULT_LM_EVAL_BATCH_SIZE,
    DEFAULT_LM_EVAL_MAX_GEN_TOKS,
    DEFAULT_LM_EVAL_MAX_LENGTH,
    DEFAULT_LM_EVAL_NUM_FEWSHOT,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_SMOKE_MAX_TOKENS,
    LM_EVAL_TERMINATE_TIMEOUT_S,
)

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
    # Standalone tasks — keys must match the lm-eval results.json format:
    # "{metric},{filter}" where filter is the post-processing step applied.
    "gsm8k": "exact_match,flexible-extract",   # flexible-extract finds numeric answer
    "gsm8k_sslm": "exact_match,flexible-extract",
    "arc_challenge": "acc_norm,none",
    "arc_challenge_chat": "exact_match,remove_whitespace",
    "arc_challenge_sslm": "exact_match,none",
    "nq_open": "exact_match,remove_whitespace",
    "nq_open_sslm": "contains,none",
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
    "gsm8k_sslm": "GSM8K",
    "arc_challenge": "ARC-C",
    "arc_challenge_chat": "ARC-C",
    "arc_challenge_sslm": "ARC-C",
    "nq_open": "NQ-Open",
    "nq_open_sslm": "NQ-Open",
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
    max_tokens: int = DEFAULT_SMOKE_MAX_TOKENS,
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


# System instruction injected into every lm_eval request.
# Goals: short output that matches each task's expected format exactly.
# Must not contain commas -- lm_eval splits model_args on "," without quoting.
# Format rules:
#   arc_challenge_chat  expects exactly the letter:  "C"
#   gsm8k flexible-extract expects "#### <number>" at the end
#   nq_open remove_whitespace expects the bare entity name
REASONING_SYSTEM_INSTRUCTION = (
    "Answer concisely. "
    "For multiple choice output only the letter. "
    "For math end with #### followed by the numeric answer. "
    "For factual questions answer with only the key fact or name."
)


def build_lm_eval_command(
    *,
    base_url: str,
    model_id: str,
    tasks: list[str],
    output_path: Path,
    num_fewshot: int = DEFAULT_LM_EVAL_NUM_FEWSHOT,
    batch_size: str = DEFAULT_LM_EVAL_BATCH_SIZE,
    max_gen_toks: int = DEFAULT_LM_EVAL_MAX_GEN_TOKS,
    max_length: int = DEFAULT_LM_EVAL_MAX_LENGTH,
    system_instruction: str = REASONING_SYSTEM_INSTRUCTION,
    limit: int | None = None,
    gen_kwargs: str | None = None,
    log_samples: bool = False,
) -> list[str]:
    # Use the reasoning wrapper so that reasoning_content is read as a fallback
    # when content is null (common for Qwen3/Zaya via vLLM --reasoning-parser).
    scripts_sslm = Path(__file__).parent.parent.parent.parent.parent / "scripts" / "sslm"
    wrapper = str(scripts_sslm / "lm-eval-reasoning.py")
    # Custom task variants (gsm8k_sslm, arc_challenge_sslm, nq_open_sslm) live
    # here; they carry scoring robust to chatty reasoning-model output.
    include_path = str(scripts_sslm / "lm-eval-tasks")
    # lm_eval posts directly to base_url with no path appended, so it must be
    # the full chat completions endpoint, not just the /v1 prefix.
    chat_url = base_url.rstrip("/") + "/chat/completions"
    # tokenizer_backend=none: ZAYA1/Qwen3 use server-side chat templates (vLLM
    # applies the correct format); we don't need a local tokenizer for context
    # checking and avoid torch/model-type warnings entirely.
    # eos_string: with no tokenizer we can't auto-detect EOS; set it explicitly
    # so lm-eval adds <|im_end|> to every request's stop list.
    model_args = (
        f"model={model_id},base_url={chat_url},tokenizer_backend=none,"
        f"eos_string=<|im_end|>,"
        f"max_gen_toks={max_gen_toks},max_length={max_length},"
        f"system_instruction={system_instruction}"
    )
    cmd = [
        sys.executable,
        wrapper,
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
        "--include_path",
        include_path,
        "--apply_chat_template",
        "--verbosity", "WARNING",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if gen_kwargs:
        cmd += ["--gen_kwargs", gen_kwargs]
    if log_samples:
        cmd += ["--log_samples"]
    return cmd


def run_lm_eval(
    *,
    base_url: str,
    model_id: str,
    tasks: list[str],
    output_path: Path,
    num_fewshot: int = DEFAULT_LM_EVAL_NUM_FEWSHOT,
    batch_size: str = DEFAULT_LM_EVAL_BATCH_SIZE,
    limit: int | None = None,
    gen_kwargs: str | None = None,
    log_samples: bool = False,
    extra_env: dict | None = None,
) -> int:
    output_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY)
    if extra_env:
        env.update(extra_env)
    command = build_lm_eval_command(
        base_url=base_url,
        model_id=model_id,
        tasks=tasks,
        output_path=output_path,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        gen_kwargs=gen_kwargs,
        log_samples=log_samples,
    )
    # start_new_session isolates lm_eval from the terminal SIGINT so Ctrl+C
    # doesn't trigger its own traceback — our orchestrator handles teardown.
    proc = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\n[sslm] Interrupted -- stopping lm_eval ...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=LM_EVAL_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise


def _best_score(task_metrics: dict) -> tuple[str, float] | tuple[None, None]:
    """Return (metric_key, value) using TASK_PRIMARY_METRIC then common fallbacks.

    lm-eval stores metrics as "{metric},{filter}" (e.g. "exact_match,flexible-extract").
    Fallbacks cover both unfiltered ("*,none") and the most common filter variants.
    """
    for key in (
        "acc_norm,none", "acc,none", "pass@1,none",
        "exact_match,flexible-extract",
        "exact_match,remove_whitespace",
        "exact_match,none",
        "exact_match,strict-match",
        "contains,none",
        "prompt_level_strict_acc,none",
    ):
        if key in task_metrics:
            return key, float(task_metrics[key])
    return None, None


def parse_lm_eval_results(results_dir: Path) -> list[dict]:
    """Scan results_dir recursively for lm-eval results.json files.

    Returns a flat list of dicts with keys:
        model, task, task_display, metric, score, date, result_file
    """
    rows: list[dict] = []
    for json_file in sorted(results_dir.rglob("results*.json")):
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
