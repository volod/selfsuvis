# nanochat
[Andrej Karpathy](https://github.com/karpathy/nanochat)

nanochat is the simplest experimental harness for training LLMs from scratch on a single GPU.
The code is minimal and hackable, covering all major stages: tokenization, pretraining,
fine-tuning (SFT), evaluation, inference, and a chat UI. A single complexity dial (`--depth`,
the number of transformer layers) automatically derives all other hyperparameters so the
resulting model is compute-optimal. GPT-2 grade capability is approximately depth 26.

local single-GPU training - refactored

## Quick start — local single-GPU training

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU ≥ 8 GB, or CPU.
All artifacts (dataset, tokenizer, checkpoints, reports) go to `.data/nanochat/`.

```bash
make hw-info      # detect GPU + CUDA toolkit, print selected profile
make venv         # auto-detect GPU → install CUDA or CPU deps via uv
make install-fa   # (optional) build flash-attn for ~2x training speedup
make train        # full pipeline: tokenizer → pretrain → SFT
make chat-cli     # interactive CLI chat with the trained model
make chat-web     # streaming web UI at http://localhost:8000
```

`make venv` and `make train` both query `nvidia-smi` at runtime and select the right
dependencies and model size automatically — no manual configuration needed.

### Auto-selected training profiles

| VRAM | Profile | Depth | Seq | Micro-batch | Grad-accum | ~Model params |
|------|---------|-------|-----|-------------|------------|---------------|
| ≥ 40 GB | `40g` | 20 | 2048 | 16 | 16 | ~430 M |
| 24–39 GB | `24g` | 18 | 2048 | 8 | 32 | ~330 M |
| 16–23 GB | `16g` | 14 | 1024 | 8 | 64 | ~160 M |
| 12–15 GB | `12g` | 12 | 1024 | 4 | 128 | ~110 M |
| 8–11 GB | `8g` | 10 | 512 | 2 | 512 | ~65 M |
| CPU / none | `cpu` | 4 | 256 | 4 | 4 | ~10 M |

All GPU profiles use a 524 288-token logical batch.
Gradient accumulation compensates for the smaller micro-batch on a single device.

---

## Setup

### Environment

```bash
make venv        # GPU (CUDA) or CPU — auto-detected
make venv-dev    # same + dev extras: pytest, ruff, tensorboard, transformers
make clean-venv  # remove .venv entirely; re-run make venv to rebuild
```

Under the hood, `make venv` runs `uv sync --extra gpu` or `--extra cpu` based on
`nvidia-smi` output. The virtual environment is created at `src/nanochat/.venv`.

### Flash-attn (optional, ~2x speedup)

Flash-attn is compiled from source and cached under `.data/wheels/flash-attn_<key>/`.
The build is adaptive to the detected CUDA toolkit (nvcc):

| nvcc version | Max SM | flash-attn built |
|---|---|---|
| **≥ 12.8** | SM 12.0 (RTX 5000 / Blackwell Ultra) | latest |
| **12.0–12.7** | SM 9.0 (Hopper and below) | v2.7.4 |

```bash
make hw-info     # shows detected nvcc and which flash-attn version will be built
make install-fa  # compile and cache (~15–30 min, one-time per GPU/nvcc combo)
```

To enable SM 12.0 kernels for Blackwell Ultra / RTX 5000 series GPUs, install the
CUDA 12.8 toolkit (`cuda-toolkit-12-8` on Ubuntu), then re-run `make install-fa`.

The compiled wheel is cached — subsequent `make venv` / `make train` runs restore it
from `.data/wheels/` automatically in under a second.

---

## Make reference

### Commands

| Command | Description |
|---|---|
| `make hw-info` | GPU, VRAM, profile, nvcc version, flash-attn readiness |
| `make venv` | Install deps (auto-detects CUDA / CPU) |
| `make venv-dev` | Same + pytest, ruff, tensorboard, transformers |
| `make clean-venv` | Delete `.venv` (run `make venv` to rebuild) |
| `make install-fa` | Build flash-attn from source; version adapts to nvcc |
| `make train` | Full pipeline auto-selected for detected GPU |
| `make train-8g` | Force 8–11 GB profile  (depth=10 seq=512  bs=2) |
| `make train-12g` | Force 12–15 GB profile (depth=12 seq=1024 bs=4) |
| `make train-16g` | Force 16–23 GB profile (depth=14 seq=1024 bs=8) |
| `make train-24g` | Force 24–39 GB profile (depth=18 seq=2048 bs=8) |
| `make train-40g` | Force ≥ 40 GB profile  (depth=20 seq=2048 bs=16) |
| `make train-cpu` | Force CPU profile       (depth=4  seq=256  bs=4) |
| `make tok-train` | Train BPE tokenizer only (downloads 8 data shards) |
| `make tok-eval` | Evaluate tokenizer compression ratio |
| `make chat-cli` | Interactive CLI chat with the trained model |
| `make chat-web` | Streaming web chat UI at http://localhost:8000 |
| `make test` | Run unit tests with pytest |
| `make lint` | ruff check + format check |
| `make clean` | Remove `.data/nanochat/` (dataset, checkpoints, reports) |

### Environment variable overrides

```bash
PROFILE=16g make train                           # force a specific profile
RUN=myrun make train                             # name this run (enables TensorBoard)
NANOCHAT_BASE_DIR=/mnt/data/nanochat make train  # custom artifact directory
NANOCHAT_DTYPE=bfloat16 make train               # override compute precision
```

---

## TensorBoard

Training logs are written to `.data/nanochat/tb_logs/` whenever `RUN` is set.

```bash
# start a named run
RUN=myrun make train

# open the dashboard (in a second terminal)
tensorboard --logdir .data/nanochat/tb_logs

# or point at the full path
tensorboard --logdir /path/to/selfsuvis/.data/nanochat/tb_logs
```

Key metrics to watch:

| Metric | What it tells you |
|---|---|
| `val_bpb` | Validation loss (bits per byte — vocab-size-invariant) |
| `core_metric` | DCLM CORE score |
| `train/mfu` | Model FLOPS utilization |
| `train/tok_per_sec` | Training throughput |
| VRAM utilization | Whether you're leaving GPU headroom on the table |

See the Research section for an example of using TensorBoard in an iteration loop.

---

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Precision is managed through a single global
`COMPUTE_DTYPE` (defined in `nanochat/common.py`), auto-detected from the GPU:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, Ada RTX 4000/5000, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

Override with:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"
NANOCHAT_DTYPE=float16 make train   # enables GradScaler automatically
```

Model weights are stored in fp32 for optimizer precision; the custom `Linear` layer casts
to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE`.
`float16` training automatically enables `GradScaler` to prevent gradient underflow.

---

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a minimal example for CPU or Apple Silicon.
It trains a very small model (depth 4) for educational purposes; expect slow iteration.

---

## Research — iteration loop

For quick experiments (~5 min pretraining runs) use the `12g` or `16g` profile and a named run:

```bash
RUN=d12_baseline make train-12g     # baseline
# edit something in the code
RUN=d12_candidate make train-12g    # candidate
tensorboard --logdir .data/nanochat/tb_logs
```

Compare `val_bpb` curves across runs to decide if a change helps. Monitor
`train/mfu` and `train/tok_per_sec` to detect regressions in throughput.

nanochat is organized around a single dial of complexity — `--depth`. This integer
automatically determines all other hyperparameters (width, number of heads, learning rate
schedule, training horizon, weight decay, etc.) so the model is always compute-optimal.
Any principled change to the code should work for all depth settings.

For multi-GPU cluster speedruns see [runs/speedrun.sh](runs/speedrun.sh).

---
## Contributing

The goal of nanochat is to improve the state of the art in micro models accessible end-to-end
on budgets of < $1000. nanochat is not a configurable LLM framework — there are no giant
configuration objects, model factories, or if-then-else monsters. It is a single, cohesive,
minimal, readable, hackable codebase that produces a ChatGPT model you can talk to.

Currently the most interesting direction is speeding up time-to-GPT-2 in the pretraining stage.

**AI policy:** when submitting a PR, please declare any parts with substantial LLM contribution
that you have not written or do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
