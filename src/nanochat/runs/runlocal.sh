#!/bin/bash
# Local single-GPU training pipeline for nanochat.
#
# Auto-detects VRAM and selects depth, seq-len, batch size, and dataset shards.
# Override any detection by setting PROFILE explicitly.
#
# Usage:
#   bash runs/runlocal.sh                          # auto-detect GPU and train
#   PROFILE=16g bash runs/runlocal.sh              # force a specific profile
#   RUN=myrun bash runs/runlocal.sh                # enable TensorBoard logging
#   NANOCHAT_BASE_DIR=/mnt/data/nanochat bash runs/runlocal.sh  # custom artifact dir
#
# Profiles auto-selected by VRAM and GPU size:
#   40g  ≥40 GB  depth=20 seq=2048 bs=16  (A100/H100)
#   24g  24-39   depth=18 seq=2048 bs=8   (RTX 3090/4090)
#   16g  16-23   depth=14 seq=1024 bs=16  (compact depth=12 on low-SM GPUs)
#   12g  12-15   depth=12 seq=1024 bs=4   (RTX 4070/3080)
#   8g    8-11   depth=10 seq=512  bs=2   (RTX 3070/4060 8G)
#   cpu   none   depth=4  seq=256  bs=4   (CPU only, educational)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

# Load project-level .env so DATA_DIR and other shared vars are available.
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

_PROJECT_DATA="${DATA_DIR:-$PROJECT_ROOT/.data}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$_PROJECT_DATA/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

_NCPU=$(nproc 2>/dev/null || echo 4)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Hardware detection ────────────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    if [ -z "${PROFILE:-}" ]; then
        eval "$(python3 "$REPO_ROOT/scripts/detect_hw.py" shell)"
        PROFILE="$HW_PROFILE"
        log "Detected GPU: ${HW_GPU_NAME:-none}  (${HW_VRAM_MB:-0} MB VRAM, ${HW_SM_COUNT:-0} SMs) -> profile: $PROFILE"
    else
        eval "$(python3 "$REPO_ROOT/scripts/detect_hw.py" shell "$PROFILE")"
        log "Using explicit PROFILE=$PROFILE"
    fi
else
    echo "python3 is required for nanochat hardware detection" >&2
    exit 1
fi

# ── Profile parameters ────────────────────────────────────────────────────────
# total_batch = 524288 tokens for all GPU profiles (same logical batch as speedrun).
# grad_accum  = total_batch / (device_batch * seq_len)  — handled automatically.
# shards      = chinchilla-optimal dataset size (≈12 × params / 62M tokens/shard).
case "$PROFILE" in
    40g)
        DEPTH=20; SEQ_LEN=2048; DEVICE_BATCH=16; TOTAL_BATCH=524288; SHARDS=120
        DEVICE_TYPE="cuda"
        ;;
    24g)
        DEPTH=18; SEQ_LEN=2048; DEVICE_BATCH=8;  TOTAL_BATCH=524288; SHARDS=90
        DEVICE_TYPE="cuda"
        ;;
    16g)
        DEPTH=14; SEQ_LEN=1024; DEVICE_BATCH=16; TOTAL_BATCH=524288; SHARDS=50
        DEVICE_TYPE="cuda"
        ;;
    12g)
        DEPTH=12; SEQ_LEN=1024; DEVICE_BATCH=4;  TOTAL_BATCH=524288; SHARDS=35
        DEVICE_TYPE="cuda"
        ;;
    8g)
        DEPTH=10; SEQ_LEN=512;  DEVICE_BATCH=2;  TOTAL_BATCH=524288; SHARDS=25
        DEVICE_TYPE="cuda"
        ;;
    cpu)
        DEPTH=4;  SEQ_LEN=256;  DEVICE_BATCH=4;  TOTAL_BATCH=16384;  SHARDS=8
        DEVICE_TYPE="cpu"
        log "WARNING: CPU training is very slow and intended for testing only."
        ;;
    *)
        echo "Unknown PROFILE='$PROFILE'. Valid: cpu 8g 12g 16g 24g 40g" >&2
        exit 1
        ;;
esac

DEPTH="${NANOCHAT_DEPTH:-${HW_DEPTH:-$DEPTH}}"
SEQ_LEN="${NANOCHAT_SEQ_LEN:-${HW_SEQ_LEN:-$SEQ_LEN}}"
DEVICE_BATCH="${NANOCHAT_DEVICE_BATCH:-${HW_DEVICE_BATCH:-$DEVICE_BATCH}}"
TOTAL_BATCH="${NANOCHAT_TOTAL_BATCH:-${HW_TOTAL_BATCH:-$TOTAL_BATCH}}"
SHARDS="${NANOCHAT_SHARDS:-${HW_SHARDS:-$SHARDS}}"

GRAD_ACCUM=$(( TOTAL_BATCH / (DEVICE_BATCH * SEQ_LEN) ))
if [ $(( TOTAL_BATCH % (DEVICE_BATCH * SEQ_LEN) )) -ne 0 ]; then
    echo "TOTAL_BATCH=$TOTAL_BATCH must be divisible by DEVICE_BATCH*SEQ_LEN=$(( DEVICE_BATCH * SEQ_LEN ))" >&2
    exit 1
fi

RUN="${RUN:-}"
[ "$DEVICE_TYPE" = "cuda" ] && UV_EXTRAS="gpu" || UV_EXTRAS="cpu"
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    if [ "$DEVICE_TYPE" = "cuda" ]; then
        export OMP_NUM_THREADS=$(( _NCPU > 10 ? 8 : (_NCPU > 2 ? _NCPU - 2 : 1) ))
    else
        export OMP_NUM_THREADS=$(( _NCPU > 2 ? _NCPU - 2 : 1 ))
    fi
fi
TOKENIZER_THREADS="${NANOCHAT_TOKENIZER_THREADS:-$(( _NCPU >= 16 ? 8 : (_NCPU > 4 ? _NCPU / 2 : (_NCPU > 1 ? _NCPU - 1 : 1)) ))}"
LOADER_BUFFER_SIZE="${NANOCHAT_LOADER_BUFFER_SIZE:-4000}"

# SFT-specific parameters: prefer explicit env override, then hardware recommendation,
# then a safe fallback. HW_SFT_* values are emitted by detect_hw.py shell mode and
# already account for GPU architecture (Ada vs Blackwell vs Hopper) and VRAM tier.
SFT_BUFFER_SIZE="${NANOCHAT_SFT_BUFFER:-${HW_SFT_BUFFER:-1000}}"
SFT_EVAL_EVERY="${NANOCHAT_SFT_EVAL_EVERY:-${HW_SFT_EVAL_EVERY:-200}}"
SFT_CHATCORE_EVERY="${NANOCHAT_SFT_CHATCORE_EVERY:-${HW_SFT_CHATCORE_EVERY:-200}}"
SFT_EVAL_TOKENS="${NANOCHAT_SFT_EVAL_TOKENS:-${HW_SFT_EVAL_TOKENS:-20971520}}"

log "============================================================"
log "  nanochat local training"
log "  profile      : $PROFILE  arch=${HW_SM_ARCH:-unknown}  SM=${HW_COMPUTE_CAP:-?}"
log "  model        : depth=$DEPTH  seq=$SEQ_LEN  micro-batch=$DEVICE_BATCH"
log "  batch        : $TOTAL_BATCH tokens/step  ($GRAD_ACCUM grad-accum steps)"
log "  dataset      : $SHARDS shards  (~$(( SHARDS * 62 / 1000 ))B tokens on disk)"
log "  artifacts    : $NANOCHAT_BASE_DIR"
log "  run          : $RUN"
if [ -n "${RESUME_FROM_STEP:-}" ]; then
    log "  resume       : step=$RESUME_FROM_STEP"
fi
log "  cpu threads  : OMP_NUM_THREADS=$OMP_NUM_THREADS  (nproc=$_NCPU)"
log "  tokenizer    : threads=$TOKENIZER_THREADS  loader-buffer=$LOADER_BUFFER_SIZE"
log "  sft          : buffer=$SFT_BUFFER_SIZE  eval-every=$SFT_EVAL_EVERY  chatcore=$SFT_CHATCORE_EVERY"
log "============================================================"

# ── venv setup ───────────────────────────────────────────────────────────────
command -v uv &>/dev/null || { echo "uv not found - install from https://docs.astral.sh/uv/"; exit 1; }
cd "$REPO_ROOT"
[ -d ".venv" ] || uv venv
uv sync --extra "$UV_EXTRAS" --quiet
source .venv/bin/activate

# flash-attn is compiled from source and not in pyproject.toml, so uv sync removes it.
# Restore from the most-recently built wheel under .data/wheels/flash-attn_*/
_FA_WHEEL=$(ls -t "$_PROJECT_DATA/wheels/flash-attn_"*/flash_attn*.whl 2>/dev/null | head -1) || true
if [ -n "$_FA_WHEEL" ] && ! python -c "import flash_attn" 2>/dev/null; then
    uv pip install --python .venv/bin/python "$_FA_WHEEL" --no-deps --quiet 2>/dev/null && \
        log "flash-attn restored ($(basename "$(dirname "$_FA_WHEEL")"))" || true
fi

# ── Flash Attention check ─────────────────────────────────────────────────────
# flash-attn needs --no-build-isolation (compiles against the installed PyTorch).
# When not available, fall back to --window-pattern L which lets SDPA use its
# fast causal path instead of building an explicit O(T^2) sliding-window mask.
if [ "$DEVICE_TYPE" = "cuda" ] && python -c "import flash_attn" 2>/dev/null; then
    WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
    FA_STATUS="flash-attn $(python -c 'import flash_attn; print(flash_attn.__version__)') -- window-pattern=${WINDOW_PATTERN}"
else
    WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
    FA_STATUS="flash-attn not available -- falling back to SDPA (window-pattern=${WINDOW_PATTERN})"
    if [ "$DEVICE_TYPE" = "cuda" ]; then
        log "NOTE: Run 'make install-fa' to build flash-attn (~30-60 min, one-time)."
        log "      This enables sliding-window attention and ~2x training speedup."
    fi
fi
log "  flash-attn : $FA_STATUS"

# ── Training report reset ─────────────────────────────────────────────────────
python -m nanochat.training.report reset

# ── Dataset ───────────────────────────────────────────────────────────────────
log "Downloading first 8 shards for tokenizer training..."
python -m nanochat.data.dataset -n 8

log "Downloading remaining shards in background (total: $SHARDS)..."
python -m nanochat.data.dataset -n "$SHARDS" &
DATASET_PID=$!

# ── Tokenizer ─────────────────────────────────────────────────────────────────
log "Training tokenizer (vocab=32768)..."
python -m scripts.tok_train
python -m scripts.tok_eval

# ── Base pretrain ─────────────────────────────────────────────────────────────
log "Waiting for dataset download..."
wait "$DATASET_PID"

log "Pretraining base model (depth=$DEPTH)..."
BASE_TRAIN_ARGS=(
    --depth="$DEPTH" \
    --max-seq-len="$SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH" \
    --total-batch-size="$TOTAL_BATCH" \
    --device-type="$DEVICE_TYPE" \
    --window-pattern="$WINDOW_PATTERN" \
    --target-param-data-ratio=12 \
    --eval-every=250 \
    --core-metric-every=2000 \
    --core-metric-max-per-task=200 \
    --sample-every=500 \
    --save-every=1000 \
    --tokenizer-threads="$TOKENIZER_THREADS" \
    --loader-buffer-size="$LOADER_BUFFER_SIZE" \
    --run="$RUN"
)
if [ -n "${RESUME_FROM_STEP:-}" ]; then
    BASE_TRAIN_ARGS+=(--resume-from-step="$RESUME_FROM_STEP")
fi
python -m scripts.base_train "${BASE_TRAIN_ARGS[@]}"

# ── Base eval ─────────────────────────────────────────────────────────────────
if [ -d "$NANOCHAT_BASE_DIR/base_checkpoints" ]; then
    log "Evaluating base model..."
    python -m scripts.base_eval \
        --device-batch-size="$DEVICE_BATCH" \
        --split-tokens=131072 \
        --max-per-task=200
else
    log "No base checkpoint found - skipping eval."
    log "  (pass --save-every N to save checkpoints during training)"
fi

# ── Identity conversations ────────────────────────────────────────────────────
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_FILE" ]; then
    log "Downloading identity conversations..."
    curl -fsSL -o "$IDENTITY_FILE" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# ── SFT ───────────────────────────────────────────────────────────────────────
if [ -d "$NANOCHAT_BASE_DIR/base_checkpoints" ]; then
    log "Supervised fine-tuning (SFT)..."
    python -m scripts.chat_sft \
        --device-batch-size="$DEVICE_BATCH" \
        --loader-buffer-size="$SFT_BUFFER_SIZE" \
        --eval-every="$SFT_EVAL_EVERY" \
        --chatcore-every="$SFT_CHATCORE_EVERY" \
        --eval-tokens="$SFT_EVAL_TOKENS" \
        --run="$RUN"
else
    log "No base checkpoint found - skipping SFT."
fi

# ── SFT eval ──────────────────────────────────────────────────────────────────
if [ -d "$NANOCHAT_BASE_DIR/sft_checkpoints" ]; then
    log "Evaluating SFT model..."
    python -m scripts.chat_eval -i sft
else
    log "No SFT checkpoint found - skipping SFT eval."
fi

# ── Report ────────────────────────────────────────────────────────────────────
log "Generating training report..."
python -m nanochat.training.report generate

log "============================================================"
log "  Training complete!"
log "  Artifacts : $NANOCHAT_BASE_DIR"
log "  Report    : $NANOCHAT_BASE_DIR/report/report.md"
log ""
log "  Chat with your model:"
log "    python -m scripts.chat_cli"
log "    python -m scripts.chat_web   # http://localhost:8000"
log "============================================================"
