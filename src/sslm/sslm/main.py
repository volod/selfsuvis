from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sslm.playground.benchmarks import build_lm_eval_command, run_lm_eval, run_smoke
from sslm.playground.catalog import BENCHMARK_SUITES, DEFAULT_MODEL_PAIR, get_model, model_table
from sslm.playground.finetune import write_qlora_recipe
from sslm.playground.orchestrator import SequentialRunConfig, run_sequential, write_compose_file


def parse_model_keys(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SSLM sidecar benchmark playground")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="List configured model sidecars")

    render = sub.add_parser("render-compose", help="Render a Docker Compose file for selected sidecars")
    render.add_argument("--models", default=",".join(DEFAULT_MODEL_PAIR))
    render.add_argument("--output", type=Path, default=Path(".data/sslm/docker-compose.generated.yml"))

    smoke = sub.add_parser("smoke", help="Run local smoke prompts against an existing endpoint")
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--base-url", required=True)
    smoke.add_argument("--output", type=Path, default=Path(".data/sslm/results/smoke.jsonl"))
    smoke.add_argument("--max-tokens", type=int, default=512)

    lm_eval = sub.add_parser("lm-eval", help="Run lm-evaluation-harness against an existing endpoint")
    lm_eval.add_argument("--model", required=True)
    lm_eval.add_argument("--base-url", required=True)
    lm_eval.add_argument("--tasks", default="gsm8k")
    lm_eval.add_argument("--output", type=Path, default=Path(".data/sslm/results/lm-eval"))
    lm_eval.add_argument("--num-fewshot", type=int, default=0)
    lm_eval.add_argument("--batch-size", default="auto")
    lm_eval.add_argument("--dry-run", action="store_true")

    sequential = sub.add_parser("sequential", help="Start sidecars one at a time and benchmark them")
    sequential.add_argument("--models", default=",".join(DEFAULT_MODEL_PAIR))
    sequential.add_argument("--suite", choices=sorted(BENCHMARK_SUITES), default="smoke")
    sequential.add_argument("--results-dir", type=Path, default=Path(".data/sslm/results"))
    sequential.add_argument("--compose-file", type=Path, default=Path(".data/sslm/docker-compose.generated.yml"))
    sequential.add_argument("--build", action="store_true")
    sequential.add_argument("--keep-running", action="store_true")

    finetune = sub.add_parser("write-finetune-config", help="Write a starter QLoRA/SFT config")
    finetune.add_argument("--base-model", default="Qwen/Qwen3-8B")
    finetune.add_argument("--dataset", default="jsonl://.data/reasoning_sft.jsonl")
    finetune.add_argument("--output", type=Path, default=Path(".data/sslm/finetune/qlora.yaml"))

    dashboard = sub.add_parser("dashboard", help="Launch Streamlit leaderboard dashboard")
    dashboard.add_argument(
        "--results-dir",
        type=Path,
        default=Path(".data/sslm/results"),
        help="Directory containing lm-eval results (default: .data/sslm/results)",
    )
    dashboard.add_argument("--port", type=int, default=8501)

    args = parser.parse_args(argv)

    if args.command == "list-models":
        print(json.dumps(model_table(), indent=2))
        return 0

    if args.command == "render-compose":
        models = [get_model(key) for key in parse_model_keys(args.models)]
        write_compose_file(models, args.output)
        print(args.output)
        return 0

    if args.command == "smoke":
        run_smoke(
            base_url=args.base_url,
            model_id=args.model,
            output_path=args.output,
            max_tokens=args.max_tokens,
        )
        print(args.output)
        return 0

    if args.command == "lm-eval":
        tasks = parse_model_keys(args.tasks)
        command = build_lm_eval_command(
            base_url=args.base_url,
            model_id=args.model,
            tasks=tasks,
            output_path=args.output,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
        )
        if args.dry_run:
            print(" ".join(command))
            return 0
        return run_lm_eval(
            base_url=args.base_url,
            model_id=args.model,
            tasks=tasks,
            output_path=args.output,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
        )

    if args.command == "sequential":
        models = [get_model(key) for key in parse_model_keys(args.models)]
        run_sequential(
            SequentialRunConfig(
                models=models,
                results_dir=args.results_dir,
                compose_file=args.compose_file,
                suite=args.suite,
                build=args.build,
                keep_running=args.keep_running,
            )
        )
        return 0

    if args.command == "write-finetune-config":
        write_qlora_recipe(args.output, base_model=args.base_model, dataset=args.dataset)
        print(args.output)
        return 0

    if args.command == "dashboard":
        app_path = Path(__file__).parent / "dashboard" / "app.py"
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", str(args.port),
            "--",
            "--results-dir", str(args.results_dir),
        ]
        print(f"Dashboard: http://localhost:{args.port}")
        return subprocess.call(cmd)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
