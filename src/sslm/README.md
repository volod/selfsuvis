# SSLM -- Reasoning LLM Benchmark Playground

Sidecar-first benchmark harness for comparing reasoning LLMs on a single machine.
One vLLM Docker container runs at a time, is benchmarked, then torn down before the
next model starts. This keeps GPU memory usage to one model at a time.

Current test model pair: **ZAYA1-8B** (Zyphra Mamba MoE) vs **Qwen3-8B** (dense baseline).

Minimum hardware: 64 GB RAM and a 24 GB GPU for the default ZAYA1/Qwen run.
Qwen can run on smaller GPUs with fp8; ZAYA defaults to BF16 because fp8 produced
invalid token output in local smoke runs.

```bash
make sslm          # Open LLM Leaderboard v2 — full suite, ~8-16 h total
make sslm-quick    # GSM8K + ARC-C + NQ-Open — ~40 min total, good for first run
```

Both commands run the full pipeline end-to-end: environment setup, sidecar image
build, benchmarks on all models sequentially, then the results dashboard. No prior
setup required.

---

## Pipeline overview

```
  SETUP
  -----
  scripts/sslm/setup-venv.sh          create .venv-sslm (eval + dashboard)
  export HUGGING_FACE_HUB_TOKEN=...   for gated models

  FOR EACH MODEL (sequential, one GPU at a time)
  -----------------------------------------------

    sslm render-compose
         |
         v
    docker-compose.generated.yml
         |
         v
    docker compose up  <sslm_{model_key}>
         |
         v  (vLLM loads model weights into GPU; per-model quantization settings apply)
         |
         +-- healthcheck loop  GET /health  (up to 15 min)
         |
         v  ready
         |
    lm_eval --model local-chat-completions          <-- lm-evaluation-harness
             --model_args model=...,base_url=...
             --tasks leaderboard_ifeval,...
             --output_path .data/sslm/results/{key}/lm-eval/
         |
         v
    results.json  written per model
         |
         v
    docker compose rm -sf  <sslm_{model_key}>       teardown; GPU freed

  RESULTS
  -------
  .data/sslm/results/
  +-- zaya1-8b/
  |   +-- lm-eval/
  |   |   +-- */results.json        lm-eval structured output
  |   +-- smoke.jsonl               optional smoke results
  +-- qwen3-8b/
      +-- lm-eval/
          +-- */results.json

  DASHBOARD
  ---------
  sslm dashboard                    Streamlit leaderboard at http://localhost:8501
```

---

## Requirements

| Resource | Minimum | Notes |
|---|---|---|
| RAM | 64 GB | lm-eval keeps eval data in host memory |
| GPU VRAM | 24 GB | ZAYA1 defaults to BF16; Qwen keeps fp8 enabled |
| GPU arch | sm_86+ (Ampere) | fp8 profiles require sm_89+ (Ada/Hopper/Blackwell) |
| Docker | 24+ | NVIDIA Container Toolkit required |
| Disk | 50 GB | HF model cache under `.data/sslm/hf-cache/` |

Set `HUGGING_FACE_HUB_TOKEN` for any gated model download.

---


## Single command: full pipeline

```bash
make sslm-quick    # ~40 min total — start here
make sslm          # ~8-16 h total — canonical Open LLM Leaderboard v2
```

Both create `.venv-sslm`, build the Zaya sidecar image, run benchmarks on all models
sequentially, then open the Streamlit leaderboard at `http://localhost:8501`.

`sslm-quick` uses the `reasoning_quick` suite (GSM8K, ARC-Challenge, NQ-Open) and is
the recommended first run to verify the pipeline end-to-end without a multi-hour wait.

Set `HUGGING_FACE_HUB_TOKEN` beforehand if any model requires authenticated download.

---

## Step-by-step: manual control

### 1. Create the isolated environment

```bash
scripts/sslm/setup-venv.sh          # installs eval + dashboard extras
```

The venv is created at `.venv-sslm`, separate from the main project `.venv`.

### 2. Inspect model profiles

```bash
.venv-sslm/bin/sslm list-models
```

Check ports, GPU memory utilization, and quantization settings before running.
Edit `src/sslm/sslm/playground/catalog.py` to add models or adjust settings.

### 3. Run benchmarks

**Standard run — Open LLM Leaderboard v2 (~4-8 h per model):**

```bash
make sslm-benchmark
```

Builds sidecar images automatically on first run via `--build`. Subsequent runs
skip the build if the image is already present.

**Fast sanity check (~20 min per model):**

```bash
make sslm-benchmark-quick
```

**Custom suite or model subset:**

```bash
scripts/sslm/run-benchmark.sh --suite reasoning_core --models qwen3-8b
```

### 4. View results

```bash
make sslm-dashboard
```

Opens `http://localhost:8501`. The leaderboard aggregates all `results.json` files
found under `.data/sslm/results/` and ranks models by average score.

---

## Benchmark suites

| Suite | Tasks | Avg time / model | Use for |
|---|---|---|---|
| `open_llm_v2` | IFEval, BBH, MATH Lvl5, GPQA Diamond, MuSR, MMLU-Pro | 4-8 h | canonical comparison |
| `reasoning_core` | GSM8K, GPQA Diamond, MMLU-Pro, IFEval | 1-2 h | focused reasoning check |
| `reasoning_quick` | GSM8K, ARC-Challenge, NQ-Open | ~20 min | fast iteration |
| `math_code` | GSM8K, Minerva Math, HumanEval | 1-2 h | math + code focus |
| `smoke` | 3 local prompts (no lm-eval) | <5 min | sanity / connectivity |

Task names are lm-evaluation-harness identifiers. The `open_llm_v2` suite uses the
`leaderboard_*` task group names that match the HuggingFace Open LLM Leaderboard v2.

---

## Pipeline internals

`sslm sequential` runs `orchestrator.run_sequential()`:

```
run_sequential(config)
  |
  +-- write_compose_file(models, compose_file)
  |     render_compose() -> YAML string -> docker-compose.generated.yml
  |
  for model in models:
    |
    +-- DockerComposeSidecar.up(build=False)
    |     docker compose up -d sslm_{key}
    |
    +-- DockerComposeSidecar.wait_ready(timeout=900s)
    |     OpenAICompatibleClient.health() polled every 5 s
    |
    +-- run_lm_eval(base_url, model_id, tasks, output_path)
    |     builds lm_eval CLI command
    |     sets OPENAI_API_KEY=EMPTY (required by lm-eval, unused by vLLM)
    |     subprocess.call(lm_eval ...)
    |     writes .data/sslm/results/{key}/lm-eval/*/results.json
    |
    +-- DockerComposeSidecar.down()
          docker compose rm -sf sslm_{key}
```

The compose service definition mounts `.data/sslm/hf-cache` into the container so
model weights are downloaded once and reused across runs.

---

## Model profiles

Each model is a `ModelProfile` in `catalog.py`. Key fields:

| Field | Default | Description |
|---|---|---|
| `image` | `vllm/vllm-openai:latest` | Docker image; override for custom forks |
| `build_context` / `dockerfile` | None | Set to enable `docker compose build` |
| `quantization` | per model | vLLM `--quantization` arg; None disables quantization |
| `min_gpu_gb` | per model | Documentation only; enforced by GPU util setting |
| `gpu_memory_utilization` | 0.88 | Fraction of VRAM for weights + KV cache |
| `max_model_len` | 32768 | Maximum sequence length |
| `extra_vllm_args` | `()` | Additional args passed to `vllm serve` |

Qwen uses fp8 by default because dense 8B bf16 models require ~16 GB VRAM.
ZAYA1 uses BF16 by default because fp8 produced invalid token output in local
smoke runs. Do not enable ZAYA fp8 unless you validate the generated samples.

---

## Adding a model

Add an entry to `MODEL_CATALOG` in `catalog.py`:

```python
"my-model-7b": ModelProfile(
    key="my-model-7b",
    model_id="org/my-model-7b",
    family="my-family",
    modality="reasoning_llm",
    port=8012,                        # unique port per model
    max_model_len=16384,
    gpu_memory_utilization=0.85,
    quantization="fp8",
    min_gpu_gb=12,
    extra_vllm_args=("--reasoning-parser", "deepseek_r1"),
),
```

Then run against it:

```bash
scripts/sslm/run-benchmark.sh --models my-model-7b --suite reasoning_quick
```

For a model that requires a custom vLLM fork, also set `image`, `build_context`,
and `dockerfile` (see the zaya1-8b entry as a reference).

---

## Remote endpoint mode

Skip Docker entirely when inference is already running on another machine:

```bash
# Smoke test
.venv-sslm/bin/sslm smoke \
  --model Qwen/Qwen3-8B \
  --base-url http://192.168.1.50:8011/v1 \
  --output .data/sslm/results/qwen3-8b/smoke.jsonl

# Full lm-eval
.venv-sslm/bin/sslm lm-eval \
  --model Qwen/Qwen3-8B \
  --base-url http://192.168.1.50:8011/v1 \
  --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_gpqa,leaderboard_musr,leaderboard_mmlu_pro \
  --output .data/sslm/results/qwen3-8b/lm-eval

# Preview the lm-eval command without running it
.venv-sslm/bin/sslm lm-eval \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8011/v1 \
  --tasks gsm8k,ifeval \
  --dry-run
```

---

## Fine-tuning scaffold

Generate a QLoRA/SFT recipe for reasoning-model adaptation:

```bash
.venv-sslm/bin/sslm write-finetune-config \
  --base-model Qwen/Qwen3-8B \
  --dataset jsonl://.data/reasoning_sft.jsonl \
  --output .data/sslm/finetune/qlora.yaml
```

The output is a reviewable YAML starting point. Install train extras to use it:

```bash
scripts/sslm/setup-venv.sh train
```

---

## CLI reference

```
sslm list-models           list configured model profiles as JSON
sslm render-compose        write docker-compose.generated.yml for selected models
sslm sequential            start sidecars one at a time and run benchmarks
sslm smoke                 run local smoke prompts against an existing endpoint
sslm lm-eval               run lm-evaluation-harness against an existing endpoint
sslm dashboard             launch Streamlit leaderboard at http://localhost:8501
sslm write-finetune-config write a QLoRA/SFT starter recipe
```

Run `sslm <subcommand> --help` for argument details.

---

## Troubleshooting

**Sidecar never becomes healthy (timeout after 900 s)**
Model weights are downloading for the first time into `.data/sslm/hf-cache/`.
Watch progress: `docker logs -f sslm_<model_key>`.
Set `HUGGING_FACE_HUB_TOKEN` if the model is gated.

**OOM during sidecar startup**
Reduce `gpu_memory_utilization` in the model profile. For dense baselines like Qwen,
confirm `quantization="fp8"` is set on smaller GPUs. ZAYA1 defaults to BF16 and
needs more VRAM for valid results. Check that no other process holds GPU memory:
`nvidia-smi`.

**lm-eval fails: OPENAI_API_KEY not set**
lm-eval requires the env var even though vLLM ignores it.
The runner sets it to `EMPTY` automatically; if running lm-eval manually,
export it: `export OPENAI_API_KEY=EMPTY`.

**Dashboard shows no results**
Results directory defaults to `.data/sslm/results/`. Confirm the path in the sidebar
matches where benchmarks wrote their output. Check for `results.json` files:
`find .data/sslm/results -name results.json`.
