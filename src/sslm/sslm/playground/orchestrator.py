from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sslm.playground.benchmarks import run_lm_eval, run_smoke
from sslm.playground.catalog import BENCHMARK_SUITES, MODEL_CATALOG, ModelProfile
from sslm.playground.constants import (
    CLI_INTERRUPT_EXIT_CODE,
    DOCKER_LOG_TAIL,
    GPU_QUERY_TIMEOUT_S,
    NVIDIA_SMI_MIB_PER_GIB,
)
from sslm.playground.sidecars import DockerComposeSidecar, render_compose


@dataclass(frozen=True)
class SequentialRunConfig:
    models: list[ModelProfile]
    results_dir: Path
    compose_file: Path
    suite: str = "smoke"
    build: bool = False
    keep_running: bool = False
    limit: int | None = None


def write_compose_file(models: list[ModelProfile], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_compose(models), encoding="utf-8")
    return output


def models_with_fallbacks(models: list[ModelProfile]) -> list[ModelProfile]:
    expanded = list(models)
    seen = {model.key for model in expanded}
    idx = 0
    while idx < len(expanded):
        model = expanded[idx]
        idx += 1
        if model.fallback_key and model.fallback_key not in seen:
            fallback = MODEL_CATALOG[model.fallback_key]
            expanded.append(fallback)
            seen.add(fallback.key)
    return expanded


def detected_gpu_total_gb() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=GPU_QUERY_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    totals: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            totals.append(float(line) / NVIDIA_SMI_MIB_PER_GIB)
        except ValueError:
            continue
    return max(totals) if totals else None


def run_sequential(config: SequentialRunConfig) -> None:
    write_compose_file(models_with_fallbacks(config.models), config.compose_file)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    suite = BENCHMARK_SUITES[config.suite]
    gpu_total_gb = detected_gpu_total_gb()
    ran_any = False

    pending = list(config.models)
    queued_keys = {model.key for model in pending}

    while pending:
        model = pending.pop(0)
        if gpu_total_gb is not None and gpu_total_gb < model.min_gpu_gb:
            print(
                f"[sslm] {model.key}: skipping; requires >= {model.min_gpu_gb} GB GPU "
                f"but detected {gpu_total_gb:.1f} GB.",
                flush=True,
            )
            if model.fallback_key and model.fallback_key not in queued_keys:
                fallback = MODEL_CATALOG[model.fallback_key]
                pending.append(fallback)
                queued_keys.add(fallback.key)
                print(
                    f"[sslm] {model.key}: queued fallback model {fallback.key}.",
                    flush=True,
                )
            continue
        ran_any = True
        sidecar = DockerComposeSidecar(config.compose_file, model)
        try:
            sidecar.prefetch()
            sidecar.up(build=config.build)
            sidecar.wait_ready()
            model_dir = config.results_dir / model.key
            if config.suite == "smoke":
                print(f"[sslm] Running smoke suite for {model.key} ...", flush=True)
                t0 = time.monotonic()
                run_smoke(
                    base_url=model.base_url,
                    model_id=model.model_id,
                    output_path=model_dir / "smoke.jsonl",
                )
                print(f"[sslm] Smoke complete in {time.monotonic() - t0:.0f}s", flush=True)
            else:
                tasks_str = ", ".join(suite.tasks)
                print(f"[sslm] Running lm_eval suite '{config.suite}' tasks=[{tasks_str}] for {model.key} ...", flush=True)
                t0 = time.monotonic()
                extra_env: dict[str, str] = {}
                if model.disable_thinking:
                    extra_env["SSLM_DISABLE_THINKING"] = "1"
                if model.force_temperature is not None:
                    extra_env["SSLM_FORCE_TEMPERATURE"] = str(model.force_temperature)
                if model.force_top_p is not None:
                    extra_env["SSLM_FORCE_TOP_P"] = str(model.force_top_p)
                if model.force_top_k is not None:
                    extra_env["SSLM_FORCE_TOP_K"] = str(model.force_top_k)
                code = run_lm_eval(
                    base_url=model.base_url,
                    model_id=model.model_id,
                    tasks=list(suite.tasks),
                    output_path=model_dir / "lm-eval",
                    num_fewshot=suite.num_fewshot,
                    limit=config.limit if config.limit is not None else suite.limit,
                    gen_kwargs=suite.gen_kwargs,
                    log_samples=suite.log_samples,
                    extra_env=extra_env or None,
                )
                elapsed = time.monotonic() - t0
                if code != 0:
                    print(f"[sslm] lm_eval exited with code {code} after {elapsed:.0f}s", flush=True)
                    raise SystemExit(code)
                print(f"[sslm] lm_eval complete in {elapsed:.0f}s -> {model_dir / 'lm-eval'}", flush=True)
        except TimeoutError as exc:
            print(f"\n[sslm] {exc}", file=sys.stderr)
            print(f"[sslm] Container logs for {model.key}:", file=sys.stderr)
            sidecar.dump_logs(tail=DOCKER_LOG_TAIL)
            raise
        except KeyboardInterrupt:
            print(f"\n[sslm] Interrupted -- tearing down {model.key} ...", file=sys.stderr, flush=True)
            sys.exit(CLI_INTERRUPT_EXIT_CODE)  # finally block below still runs and calls sidecar.down()
        finally:
            if not config.keep_running:
                sidecar.down()

    if not ran_any:
        raise SystemExit("[sslm] No selected models can run on the detected GPU.")
