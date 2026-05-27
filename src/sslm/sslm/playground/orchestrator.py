from __future__ import annotations

import sys
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
                run_smoke(
                    base_url=model.base_url,
                    model_id=model.model_id,
                    output_path=model_dir / "smoke.jsonl",
                )
            else:
                code = run_lm_eval(
                    base_url=model.base_url,
                    model_id=model.model_id,
                    tasks=list(suite.tasks),
                    output_path=model_dir / "lm-eval",
                    num_fewshot=suite.num_fewshot,
                )
                if code != 0:
                    raise SystemExit(code)
        except KeyboardInterrupt:
            print(f"\nInterrupted -- tearing down {model.key} ...", file=sys.stderr)
            raise
        finally:
            if not config.keep_running:
                sidecar.down()
