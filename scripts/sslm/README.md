# SSLM Playground

Sidecar-first benchmark playground for reasoning LLMs, VLMs, and future world-model
backends. The first configured pair is:

- `Zyphra/ZAYA1-8B`
- `Qwen/Qwen3-8B`

Create the separate environment:

```bash
scripts/sslm/setup-venv.sh
```

Render sidecars:

```bash
scripts/sslm/render-compose.sh --output scripts/sslm/docker-compose.generated.yml
```

Run sequential local smoke tests, one GPU sidecar at a time:

```bash
scripts/sslm/run-zaya-qwen-smoke.sh --build
```

Run existing lm-evaluation-harness tasks against an already running endpoint:

```bash
.venv-sslm/bin/sslm lm-eval \
  --model Zyphra/ZAYA1-8B \
  --base-url http://localhost:8010/v1 \
  --tasks gsm8k,gpqa_diamond,mmlu_pro,ifeval
```

Generate a starter fine-tuning recipe:

```bash
.venv-sslm/bin/sslm write-finetune-config \
  --base-model Qwen/Qwen3-8B \
  --dataset jsonl://.data/reasoning_sft.jsonl
```
