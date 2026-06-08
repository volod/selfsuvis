.PHONY: help hw-info venv venv-dev clean-venv install-fa test lint \
        train train-8g train-12g train-16g train-24g train-40g train-cpu \
        tok-train tok-eval \
        chat-cli chat-web \
        clean

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Respect project-level DATA_DIR env var; fall back to ../../.data relative to this package.
_PROJECT_DATA := $(or $(DATA_DIR),$(abspath $(CURDIR)/../../.data))
# Runtime data: <PROJECT_DATA>/nanochat/ (project-wide convention: <data>/<module>/).
DATA_DIR := $(_PROJECT_DATA)/nanochat
# Shared compiled wheels: <PROJECT_DATA>/wheels/ (shared across all sub-packages).
WHEEL_CACHE := $(_PROJECT_DATA)/wheels

# Keep uv's cache on the same filesystem as .venv so uv hardlinks wheels instead of
# full-copying across volumes (the default ~/.cache/uv may be on a different mount).
# Anchored to the resolved project data dir (.data/uv-cache, alongside wheels/) — no
# hardcoded paths. Exported so every uv call (uv venv / uv sync / uv pip) picks it up.
export UV_CACHE_DIR := $(_PROJECT_DATA)/uv-cache

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo "================================================"
	@echo "  nanochat — local LLM training pipeline"
	@echo "================================================"
	@echo ""
	@echo "  Hardware"
	@echo "  --------"
	@echo "  make hw-info       Detect GPU, show selected profile and parameters"
	@echo ""
	@echo "  Setup"
	@echo "  -----"
	@echo "  make venv          Auto-detect GPU -> install CPU, gpu-cu126, or gpu-cu128 deps"
	@echo "  make venv-dev      Same + dev extras (pytest, ruff, tensorboard, transformers)"
	@echo "  make clean-venv    Remove .venv (re-run make venv afterwards)"
	@echo "  make install-fa    Build flash-attn; version adapts to detected nvcc:"
	@echo "                       nvcc >= 12.8 : flash-attn latest  (SM 8.9 + SM 12.0)"
	@echo "                       nvcc <  12.8 : flash-attn v2.7.4  (SM 8.9 only)"
	@echo "                     Install CUDA 12.8 toolkit for full Blackwell support."
	@echo ""
	@echo "  Training  (single GPU, auto-selects profile from VRAM)"
	@echo "  --------"
	@echo "  make train         Full pipeline: tokenizer → pretrain → SFT → chat"
	@echo "  make train-8g      Force  8–11 GB profile  (depth=10 seq=512  bs=2)"
	@echo "  make train-12g     Force 12–15 GB profile  (depth=12 seq=1024 bs=4)"
	@echo "  make train-16g     Force 16–23 GB profile  (depth=14 seq=1024 bs=8)"
	@echo "  make train-24g     Force 24–39 GB profile  (depth=18 seq=2048 bs=8)"
	@echo "  make train-40g     Force   ≥40 GB profile  (depth=20 seq=2048 bs=16)"
	@echo "  make train-cpu     Force CPU profile        (depth=4  seq=256  bs=4)"
	@echo ""
	@echo "  Tokenizer"
	@echo "  ---------"
	@echo "  make tok-train     Train the BPE tokenizer (downloads 8 data shards)"
	@echo "  make tok-eval      Evaluate tokenizer compression ratio"
	@echo ""
	@echo "  Inference"
	@echo "  ---------"
	@echo "  make chat-cli      Interactive CLI chat with the trained model"
	@echo "  make chat-web      Streaming web chat UI at http://localhost:8000"
	@echo ""
	@echo "  Quality"
	@echo "  -------"
	@echo "  make test          Run unit tests with pytest"
	@echo "  make lint          Run ruff check + format check"
	@echo ""
	@echo "  Maintenance"
	@echo "  -----------"
	@echo "  make clean         Remove .data/nanochat/ (dataset, checkpoints, reports)"
	@echo ""
	@echo "  Env-var overrides:"
	@echo "    RUN=<name>                  TensorBoard run name (default: auto timestamp)"
	@echo "    NANOCHAT_BASE_DIR=<path>    Artifact directory (default: ../.data/nanochat/)"
	@echo "    NANOCHAT_TORCH_CUDA=cu126   Force torch CUDA runtime (auto, cu126, cu128)"
	@echo ""
	@echo "  Run  make help  to show this again."

# ── Hardware detection ────────────────────────────────────────────────────────
hw-info:
	@echo ""
	@python3 scripts/detect_hw.py info
	@echo ""
	@$(PYTHON) scripts/bench_train.py || echo "  Benchmark : run 'make venv' to enable"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
# Detect hardware once via detect_hw.py 'shell' mode (stdlib only, no venv needed)
# and validate the driver + selected torch CUDA runtime before syncing. Sets $EXTRAS.
# Falls back to nvidia-smi (assume CUDA 12.6 wheels, broadest driver support) when
# python3 is unavailable. The calling recipe shell must run with `set -e`.
_DETECT_EXTRAS = \
  if command -v python3 >/dev/null 2>&1; then \
    eval "$$(python3 scripts/detect_hw.py shell)"; EXTRAS="$$HW_EXTRAS"; \
  else \
    { command -v nvidia-smi >/dev/null 2>&1 && EXTRAS=gpu-cu126 || EXTRAS=cpu; }; \
    HW_TORCH_DRIVER_OK=1; HW_TORCH_RUNTIME_OK=1; \
  fi; \
  echo "Detected extras: --extra $$EXTRAS"; \
  if [ "$$EXTRAS" != "cpu" ] && [ "$$HW_TORCH_DRIVER_OK" != "1" ]; then \
    echo "NVIDIA driver $${HW_DRIVER_VERSION:-unknown} is too old for torch+$${HW_TORCH_CUDA}; update the driver or set NANOCHAT_TORCH_CUDA=cu126 for pre-Blackwell GPUs."; exit 1; \
  fi; \
  if [ "$$EXTRAS" != "cpu" ] && [ "$$HW_TORCH_RUNTIME_OK" != "1" ]; then \
    echo "Selected torch+$${HW_TORCH_CUDA} is not compatible with this GPU; Blackwell / SM 10.0+ requires gpu-cu128."; exit 1; \
  fi

# flash-attn is an optional ~2x speedup, so a build failure must not fail `make venv`.
_FA_OPTIONAL_NOTE = flash-attn not built (optional ~2x speedup) - training will use the SDPA fallback. Run 'make install-fa' to retry.

venv:
	@command -v uv >/dev/null 2>&1 \
	  || (echo "uv not found — install from https://docs.astral.sh/uv/" && exit 1)
	@[ -d .venv ] || uv venv
	@set -e; $(_DETECT_EXTRAS); uv sync --extra "$$EXTRAS"
	@[ "$$(python3 scripts/detect_hw.py extras 2>/dev/null || echo cpu)" = "cpu" ] \
	  || $(MAKE) _install-fa-if-missing || echo "$(_FA_OPTIONAL_NOTE)"

venv-dev:
	@command -v uv >/dev/null 2>&1 \
	  || (echo "uv not found — install from https://docs.astral.sh/uv/" && exit 1)
	@[ -d .venv ] || uv venv
	@set -e; $(_DETECT_EXTRAS); \
	  echo "(plus dev group: pytest, ruff, tensorboard, transformers)"; \
	  uv sync --extra "$$EXTRAS" --group dev
	@[ "$$(python3 scripts/detect_hw.py extras 2>/dev/null || echo cpu)" = "cpu" ] \
	  || $(MAKE) _install-fa-if-missing || echo "$(_FA_OPTIONAL_NOTE)"

clean-venv:
	@echo "Removing .venv..."
	rm -rf .venv
	@echo "Done. Run 'make venv' to reinstall."

# Internal: install flash-attn only if not already present (idempotent).
_install-fa-if-missing:
	@$(PYTHON) -c "import flash_attn" 2>/dev/null \
	  && echo "flash-attn already installed ($(shell $(PYTHON) -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null))" \
	  || $(MAKE) install-fa

# ── Flash Attention 2 (cached source build) ───────────────────────────────────
# Wheel cache: .data/wheels/flash-attn_{key}/
# Cache key encodes torch, CUDA runtime, GPU SM list, and nvcc toolkit version.
#
# Version selection (adaptive to detected nvcc):
#   nvcc >= 12.8 → flash-attn latest  (compiles SM 12.0 Blackwell Ultra kernels)
#   nvcc <  12.8 → flash-attn v2.7.4  (SM 9.0 max; 2.8+ unconditionally adds
#                                       compute_120 which nvcc < 12.8 can't compile)
#
# Install CUDA 12.8 toolkit to unlock SM 12.0 (RTX 5000 / Blackwell Ultra) support.
# MAX_JOBS: min(max(1,(nproc-2)/2), max(1,RAM_GB/denom)) — denom=20 (<64GB), 16 (>=64GB)
install-fa:
	@$(PYTHON) -m ensurepip --upgrade >/dev/null 2>&1 || true
	@set -e; \
	\
	$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU — flash-attn is CUDA-only, skipping'; print('CUDA OK')" \
	  || { echo "No CUDA GPU detected — skipping flash-attn build."; exit 0; }; \
	\
	ARCHES=$$($(PYTHON) -c "import torch; caps=sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}); print(';'.join(f'{m}.{n}' for m,n in caps))"); \
	CUDA_HOME=$$(python3 scripts/detect_hw.py cuda_home); \
	if [ -n "$$CUDA_HOME" ]; then \
		export CUDA_HOME; \
		export PATH="$$CUDA_HOME/bin:$$PATH"; \
	fi; \
	NVCC_VER=$$(python3 scripts/detect_hw.py nvcc_ver); \
	COMPILABLE_ARCHES=$$(python3 scripts/detect_hw.py filtered_archs "$$ARCHES" "$$NVCC_VER"); \
	\
	echo "GPU architectures : $$ARCHES"; \
	if [ -n "$$CUDA_HOME" ]; then echo "CUDA_HOME         : $$CUDA_HOME"; fi; \
	echo "nvcc version      : $$NVCC_VER  (max SM: $$(python3 scripts/detect_hw.py max_sm $$NVCC_VER))"; \
	if [ "$$COMPILABLE_ARCHES" != "$$ARCHES" ]; then \
		echo "  Note: nvcc $$NVCC_VER skips archs not in: $$COMPILABLE_ARCHES"; \
		echo "        Install CUDA 12.8 toolkit for full SM support"; \
	fi; \
	if [ -z "$$COMPILABLE_ARCHES" ]; then \
		echo "No GPU architectures are compilable by nvcc $$NVCC_VER; skipping flash-attn build."; \
		echo "Install a CUDA toolkit that supports $$ARCHES, then re-run make install-fa."; \
		exit 0; \
	fi; \
	\
	FA_LATEST=$$($(PYTHON) -c "print(1 if tuple(int(x) for x in '$$NVCC_VER'.split('.')) >= (12,8) else 0)"); \
	if [ "$$FA_LATEST" = "1" ]; then \
		FA_SRC="flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git"; \
		FA_TAG="latest"; \
	else \
		FA_SRC="flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4"; \
		FA_TAG="v2.7.4  (nvcc $$NVCC_VER < 12.8 — SM 12.0 kernels need CUDA 12.8+)"; \
	fi; \
	echo "flash-attn        : $$FA_TAG"; \
	\
	FA_KEY=$$($(PYTHON) -c "import sys,torch; v=torch.__version__.split('+')[0]; c=(torch.version.cuda or 'cpu').replace('.',''); caps=sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}); sm='_'.join('sm'+''.join(str(x) for x in cap) for cap in caps); nv=sys.argv[1].replace('.',''); print(f'torch{v}_cu{c}_{sm}_nvcc{nv}')" "$$NVCC_VER"); \
	FA_DIR="$(WHEEL_CACHE)/flash-attn_$$FA_KEY"; \
	echo "Wheel cache       : $$FA_DIR"; \
	\
	if ls "$$FA_DIR"/flash_attn*.whl 1>/dev/null 2>&1; then \
		echo "Cached wheel found — installing (no compilation needed)..."; \
	else \
		JOBS=$$(python3 scripts/detect_hw.py max_jobs); \
		echo "No cached wheel — building from source (MAX_JOBS=$$JOBS, ~15-30 min)..."; \
		echo "  PyTorch : $$($(PYTHON) -c 'import torch; print(torch.__version__)')"; \
		echo "  Archs   : $$COMPILABLE_ARCHES"; \
		uv pip install --python "$(PYTHON)" wheel --quiet; \
		BUILD_TMP=$$(mktemp -d); \
		if TORCH_CUDA_ARCH_LIST="$$COMPILABLE_ARCHES" MAX_JOBS=$$JOBS $(PYTHON) -m pip wheel \
				"$$FA_SRC" \
				--no-build-isolation --no-cache-dir \
				--wheel-dir "$$BUILD_TMP"; then \
			mkdir -p "$$FA_DIR"; \
			mv "$$BUILD_TMP"/flash_attn*.whl "$$FA_DIR/"; \
			echo "Wheel built and cached at $$FA_DIR"; \
		else \
			rm -rf "$$BUILD_TMP"; \
			echo "Build failed — cache unchanged."; \
			exit 1; \
		fi; \
		rm -rf "$$BUILD_TMP"; \
	fi; \
	WHEEL_FILE=$$(ls "$$FA_DIR"/flash_attn*.whl | head -1); \
	uv pip install --python "$(PYTHON)" "$$WHEEL_FILE" --no-deps --quiet; \
	$(PYTHON) -c "import flash_attn; print('flash-attn ' + flash_attn.__version__ + ' ready')"

# ── Training ──────────────────────────────────────────────────────────────────
train:
	bash runs/runlocal.sh

train-8g:
	PROFILE=8g bash runs/runlocal.sh

train-12g:
	PROFILE=12g bash runs/runlocal.sh

train-16g:
	PROFILE=16g bash runs/runlocal.sh

train-24g:
	PROFILE=24g bash runs/runlocal.sh

train-40g:
	PROFILE=40g bash runs/runlocal.sh

train-cpu:
	PROFILE=cpu bash runs/runlocal.sh

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok-train:
	$(PYTHON) -m nanochat.data.dataset -n 8
	$(PYTHON) -m scripts.tok_train

tok-eval:
	$(PYTHON) -m scripts.tok_eval

# ── Inference ─────────────────────────────────────────────────────────────────
chat-cli:
	$(PYTHON) -m scripts.chat_cli

chat-web:
	$(PYTHON) -m scripts.chat_web

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check nanochat/ scripts/ tasks/ tests/
	$(PYTHON) -m ruff format --check nanochat/ scripts/ tasks/ tests/

# ── Maintenance ───────────────────────────────────────────────────────────────
clean:
	@echo "Removing $(DATA_DIR) (dataset, checkpoints, tokenizer, reports)..."
	rm -rf "$(DATA_DIR)"
	@echo "Done."
