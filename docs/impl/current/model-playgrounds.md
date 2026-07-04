# Model Playgrounds -- nanochat and sslm

Two self-contained experimentation packages that feed model understanding back
into the main stack. Neither is imported by production code.

## nanochat (`src/nanochat/`)

Karpathy's nanochat, refactored for local single-GPU training: tokenizer ->
pretrain -> SFT -> eval -> chat, with one complexity dial (`--depth`).

- Hardware-adaptive: `scripts/detect_hw.py` picks a profile by VRAM
  (`40g` d20/~430M ... `8g` d10/~65M, `cpu` d4/~10M) and drives dependency
  selection (`cpu`, `gpu-cu126`, `gpu-cu128`) and `MAX_JOBS`.
- Own Makefile and venv: `make venv`, `make install-fa` (flash-attn), `make hw-info`,
  `make train`, `make chat-cli`, `make chat-web` (port 8000).
- Artifacts under `.data/nanochat/` per the shared data-layout rule.

Role in the project: a controlled harness for understanding LLM training
mechanics that inform the reasoning-sidecar choices elsewhere.

## sslm (`src/sslm/`)

Sidecar-first reasoning-LLM benchmark harness: one vLLM container at a time is
rendered (`sslm render-compose` -> `docker-compose.generated.yml`), benchmarked,
and torn down, keeping GPU memory to a single model.

- Suites: `open_llm_v2` (full, ~8-16 h) and `reasoning_quick`
  (GSM8K + ARC-C + NQ-Open, ~40 min).
- Current pair: ZAYA1-8B (Mamba MoE, BF16) vs Qwen3-8B (dense; fp8-capable).
- Make targets: `sslm`, `sslm-quick`, `sslm-rebuild`, `sslm-benchmark[-quick]`,
  `sslm-dashboard` (Streamlit leaderboard on 8501); venv via
  `scripts/sslm/setup-venv.sh`; results under `.data/sslm/results`.
- Compiled sidecar wheels cache under `.data/wheels/` per the shared rule.

Role in the project: selects the reasoning backend behind `REASONING_API_URL`
(scene synthesis, Qwen3 audit step) with measured evidence instead of vibes.
