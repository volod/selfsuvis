from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sslm.playground.client import ChatRequest, OpenAICompatibleClient


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
    batch_size: str = "auto",
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
    batch_size: str = "auto",
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
