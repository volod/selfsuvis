.PHONY: help hw-info venv venv-dev install-fa test lint \
        train train-8g train-12g train-16g train-24g train-40g train-cpu \
        tok-train tok-eval \
        chat-cli chat-web \
        clean

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Runtime data: .data/nanochat/ (project-wide convention: .data/<module>/).
DATA_DIR := $(abspath $(CURDIR)/../../.data/nanochat)
# Shared compiled wheels: .data/wheels/ (shared across all sub-packages).
WHEEL_CACHE := $(abspath $(CURDIR)/../../.data/wheels)

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
	@echo "  make venv          Auto-detect GPU → install CUDA or CPU deps via uv"
	@echo "  make venv-dev      Same + dev extras (pytest, ruff, tensorboard, transformers)"
	@echo "  make install-fa    Build flash-attn v2 from source (Ada/Ampere sm75+, ~10 min)"
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
# Uses system python3 (stdlib only) to detect GPU before venv exists.
# Falls back to nvidia-smi if python3 is unavailable.
_EXTRAS_CMD = python3 scripts/detect_hw.py extras 2>/dev/null \
              || (command -v nvidia-smi >/dev/null 2>&1 && echo gpu || echo cpu)

venv:
	@command -v uv >/dev/null 2>&1 \
	  || (echo "uv not found — install from https://docs.astral.sh/uv/" && exit 1)
	@[ -d .venv ] || uv venv
	$(eval EXTRAS := $(shell $(_EXTRAS_CMD)))
	@echo "Detected extras: --extra $(EXTRAS)"
	uv sync --extra $(EXTRAS)
	@$(PYTHON) -m ensurepip --upgrade
	@[ "$(EXTRAS)" = "gpu" ] && $(MAKE) _install-fa-if-missing || true

venv-dev:
	@command -v uv >/dev/null 2>&1 \
	  || (echo "uv not found — install from https://docs.astral.sh/uv/" && exit 1)
	@[ -d .venv ] || uv venv
	$(eval EXTRAS := $(shell $(_EXTRAS_CMD)))
	@echo "Detected extras: --extra $(EXTRAS) --group dev"
	uv sync --extra $(EXTRAS) --group dev
	@$(PYTHON) -m ensurepip --upgrade
	@[ "$(EXTRAS)" = "gpu" ] && $(MAKE) _install-fa-if-missing || true

# Internal: install flash-attn only if not already present (idempotent).
_install-fa-if-missing:
	@$(PYTHON) -c "import flash_attn" 2>/dev/null \
	  && echo "flash-attn already installed ($(shell $(PYTHON) -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null))" \
	  || $(MAKE) install-fa

# ── Flash Attention 2 (cached source build) ───────────────────────────────────
# Wheel cache lives in the shared .data/wheels/ directory (project-wide convention).
# Cache key: flash-attn_torch{version}_cu{cuda}  (e.g. flash-attn_torch2.9.1_cu128)
# First run: clone + compile (~30-60 min), pip wheel → cache dir, then install.
# Later runs: install directly from the cached .whl — no compilation.
# MAX_JOBS capped per CLAUDE.md: min(max(1,(nproc-2)/2), max(1,RAM_GB/12)).
install-fa:
	@$(PYTHON) -m ensurepip --upgrade >/dev/null 2>&1 || true
	@set -e; \
	\
	$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU — flash-attn is CUDA-only, skipping'; print('CUDA OK')" || { echo "No CUDA GPU detected — skipping flash-attn build."; exit 0; }; \
	\
	ARCHES=$$($(PYTHON) -c "import torch; caps=sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}); print(';'.join(f'{m}.{n}' for m,n in caps))"); \
	FA_KEY=$$($(PYTHON) -c "import torch; v=torch.__version__.split('+')[0]; c=(torch.version.cuda or 'cpu').replace('.',''); caps=sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}); sm='_'.join('sm'+''.join(str(x) for x in cap) for cap in caps); print(f'torch{v}_cu{c}_{sm}')"); \
	\
	FA_DIR="$(WHEEL_CACHE)/flash-attn_$$FA_KEY"; \
	echo "GPU architectures : $$ARCHES"; \
	echo "Wheel cache       : $$FA_DIR"; \
	\
	if ls "$$FA_DIR"/flash_attn*.whl 1>/dev/null 2>&1; then \
		echo "Cached wheel found — installing (no compilation needed)..."; \
		$(PYTHON) -m pip install "$$FA_DIR"/flash_attn*.whl --no-deps --quiet; \
	else \
		JOBS=$$(python3 scripts/detect_hw.py max_jobs); \
		echo "No cached wheel — building from source (MAX_JOBS=$$JOBS, ~15-30 min per arch)..."; \
		echo "  PyTorch : $$($(PYTHON) -c 'import torch; print(torch.__version__)')"; \
		echo "  Archs   : $$ARCHES (TORCH_CUDA_ARCH_LIST — only local GPU(s), avoids unsupported archs)"; \
		BUILD_TMP=$$(mktemp -d); \
		if TORCH_CUDA_ARCH_LIST="$$ARCHES" MAX_JOBS=$$JOBS $(PYTHON) -m pip wheel \
				"flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git" \
				--no-build-isolation --no-cache-dir \
				--wheel-dir "$$BUILD_TMP"; then \
			mkdir -p "$$FA_DIR"; \
			mv "$$BUILD_TMP"/flash_attn*.whl "$$FA_DIR/"; \
			echo "Wheel built and cached at $$FA_DIR"; \
			$(PYTHON) -m pip install "$$FA_DIR"/flash_attn*.whl --no-deps --quiet; \
		else \
			rm -rf "$$BUILD_TMP"; \
			echo "Build failed — cache unchanged."; \
			exit 1; \
		fi; \
		rm -rf "$$BUILD_TMP"; \
	fi; \
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
