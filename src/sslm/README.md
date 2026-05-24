# SSLM

Sidecar-based benchmark playground for reasoning LLMs, VLMs, and future
world-model inference backends.

The first configured comparison is:

- `Zyphra/ZAYA1-8B`
- `Qwen/Qwen3-8B`

The playground keeps inference sidecars decoupled from benchmark code. You can run the
same smoke tests and lm-evaluation-harness jobs against Docker sidecars on this machine
or against OpenAI-compatible endpoints hosted on another node or cluster.

## Layout

The package root is intentionally small:

- `pyproject.toml` defines the standalone package and optional extras.
- `main.py` is a thin local entry-point wrapper.
- `sslm/main.py` contains the installed console entry point.
- `sslm/playground/` contains implementation modules.

Operational wrappers and Docker assets live under `scripts/sslm/`.

## Setup

Create the separate SSLM virtual environment from the project root:

```bash
scripts/sslm/setup-venv.sh
```

For a minimal environment without benchmark extras:

```bash
scripts/sslm/setup-venv.sh none
```

The environment is created at `.venv-sslm`, separate from the main project `.venv`.

## Inspect Models

List configured model profiles:

```bash
.venv-sslm/bin/sslm list-models
```

The current profiles expose local OpenAI-compatible endpoints on:

- ZAYA1-8B: `http://localhost:8010/v1`
- Qwen3-8B: `http://localhost:8011/v1`

## Render Docker Sidecars

Render a Compose file for the configured sidecars:

```bash
scripts/sslm/render-compose.sh --output scripts/sslm/docker-compose.generated.yml
```

The generated sidecars reserve NVIDIA GPU access and mount the Hugging Face cache at
`.data/sslm/hf-cache`. Set `HUGGING_FACE_HUB_TOKEN` in your shell if a model requires
authenticated download.

## Run Sequential Local Benchmarks

Run the ZAYA1-8B versus Qwen3-8B smoke benchmark one sidecar at a time:

```bash
scripts/sslm/run-zaya-qwen-smoke.sh --build
```

This starts one inference container, waits for `/health`, runs the selected benchmark,
removes that sidecar, and then moves to the next model. This avoids loading both 8B
models into GPU memory at the same time.

Results are written under:

```bash
.data/sslm/results/
```

## Run Against Existing Endpoints

If inference is already running locally or on a cluster, skip Docker and point the
benchmark client at the endpoint:

```bash
.venv-sslm/bin/sslm smoke \
  --model Zyphra/ZAYA1-8B \
  --base-url http://localhost:8010/v1 \
  --output .data/sslm/results/zaya-smoke.jsonl
```

Run existing lm-evaluation-harness tasks against any OpenAI-compatible endpoint:

```bash
.venv-sslm/bin/sslm lm-eval \
  --model Zyphra/ZAYA1-8B \
  --base-url http://localhost:8010/v1 \
  --tasks gsm8k,gpqa_diamond,mmlu_pro,ifeval
```

Preview the command without running it:

```bash
.venv-sslm/bin/sslm lm-eval \
  --model Zyphra/ZAYA1-8B \
  --base-url http://localhost:8010/v1 \
  --tasks gsm8k,ifeval \
  --dry-run
```

## Fine-Tuning Scaffold

Generate a starter QLoRA/SFT recipe for reasoning-model adaptation:

```bash
.venv-sslm/bin/sslm write-finetune-config \
  --base-model Qwen/Qwen3-8B \
  --dataset jsonl://data/reasoning_sft.jsonl
```

The generated config is a reviewable starting point, not an automatic training launch.
