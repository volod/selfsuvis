from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sslm.playground.benchmarks import run_lm_eval, run_smoke
from sslm.playground.catalog import BENCHMARK_SUITES, ModelProfile
from sslm.playground.sidecars import DockerComposeSidecar, render_compose


@dataclass(frozen=True)
class SequentialRunConfig:
    models: list[ModelProfile]
    results_dir: Path
    compose_file: Path
    suite: str = "smoke"
    build: bool = False
    keep_running: bool = False


def write_compose_file(models: list[ModelProfile], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_compose(models), encoding="utf-8")
    return output


def run_sequential(config: SequentialRunConfig) -> None:
    write_compose_file(config.models, config.compose_file)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    suite = BENCHMARK_SUITES[config.suite]

    for model in config.models:
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
                code = run_lm_eval(
                    base_url=model.base_url,
                    model_id=model.model_id,
                    tasks=list(suite.tasks),
                    output_path=model_dir / "lm-eval",
                    num_fewshot=suite.num_fewshot,
                )
                elapsed = time.monotonic() - t0
                if code != 0:
                    print(f"[sslm] lm_eval exited with code {code} after {elapsed:.0f}s", flush=True)
                    raise SystemExit(code)
                print(f"[sslm] lm_eval complete in {elapsed:.0f}s -> {model_dir / 'lm-eval'}", flush=True)
        except TimeoutError as exc:
            print(f"\n[sslm] {exc}", file=sys.stderr)
            print(f"[sslm] Container logs for {model.key}:", file=sys.stderr)
            sidecar.dump_logs(tail=100)
            raise
        except KeyboardInterrupt:
            print(f"\n[sslm] Interrupted -- tearing down {model.key} ...", file=sys.stderr, flush=True)
            sys.exit(130)  # finally block below still runs and calls sidecar.down()
        finally:
            if not config.keep_running:
                sidecar.down()
