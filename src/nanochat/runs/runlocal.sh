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
#   NANOCHAT_BASE_DIR=/path/to/nanochat bash runs/runlocal.sh  # custom artifact dir
#   RESUME_FROM_STEP=5000 bash runs/runlocal.sh    # resume base-pretrain from step
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
# Resolve a relative data dir against the project root (not the CWD) so artifacts and
# caches always land in the same place regardless of where the script is invoked from.
case "$_PROJECT_DATA" in /*) ;; *) _PROJECT_DATA="$PROJECT_ROOT/${_PROJECT_DATA#./}" ;; esac
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$_PROJECT_DATA/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Keep uv's cache on the same filesystem as .venv so wheels hardlink instead of copy.
# Derived from the project data dir (honors UV_CACHE_DIR from .env); never hardcoded.
_UV_CACHE="${UV_CACHE_DIR:-$_PROJECT_DATA/uv-cache}"
case "$_UV_CACHE" in /*) ;; *) _UV_CACHE="$PROJECT_ROOT/${_UV_CACHE#./}" ;; esac
export UV_CACHE_DIR="$_UV_CACHE"
mkdir -p "$UV_CACHE_DIR"

_NCPU=$(nproc 2>/dev/null || echo 4)

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARNING: $*" >&2; }
err()  { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

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
    err "python3 is required for nanochat hardware detection"
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
        warn "CPU training is very slow and intended for testing only."
        ;;
    *)
        err "Unknown PROFILE='$PROFILE'. Valid: cpu 8g 12g 16g 24g 40g"
        exit 1
        ;;
esac

# Allow env-var overrides of individual parameters (NANOCHAT_* beats HW_* beats profile default)
DEPTH="${NANOCHAT_DEPTH:-${HW_DEPTH:-$DEPTH}}"
SEQ_LEN="${NANOCHAT_SEQ_LEN:-${HW_SEQ_LEN:-$SEQ_LEN}}"
DEVICE_BATCH="${NANOCHAT_DEVICE_BATCH:-${HW_DEVICE_BATCH:-$DEVICE_BATCH}}"
TOTAL_BATCH="${NANOCHAT_TOTAL_BATCH:-${HW_TOTAL_BATCH:-$TOTAL_BATCH}}"
SHARDS="${NANOCHAT_SHARDS:-${HW_SHARDS:-$SHARDS}}"

GRAD_ACCUM=$(( TOTAL_BATCH / (DEVICE_BATCH * SEQ_LEN) ))

RUN="${RUN:-}"
[ "$DEVICE_TYPE" = "cuda" ] && UV_EXTRAS="${HW_EXTRAS:-gpu-cu128}" || UV_EXTRAS="cpu"
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
SFT_SAVE_EVERY="${NANOCHAT_SFT_SAVE_EVERY:-${HW_SFT_SAVE_EVERY:-500}}"

# ── Pre-flight sanity checks ──────────────────────────────────────────────────
# Run all checks up-front so failures are caught before any GPU time is spent.
log "Running pre-flight checks..."
_preflight_ok=1

# 1. Batch size divisibility — base_train.py asserts this, but checking here
#    gives a much clearer message and stops before compilation.
if [ $(( TOTAL_BATCH % (DEVICE_BATCH * SEQ_LEN) )) -ne 0 ]; then
    err "TOTAL_BATCH=$TOTAL_BATCH is not divisible by DEVICE_BATCH*SEQ_LEN=$(( DEVICE_BATCH * SEQ_LEN ))."
    err "  Fix: adjust DEVICE_BATCH or TOTAL_BATCH so they divide evenly."
    err "  e.g. NANOCHAT_DEVICE_BATCH=8 bash runs/runlocal.sh"
    _preflight_ok=0
fi

# 2. VRAM check — compare detected VRAM against the minimum required by the profile.
#    HW_MIN_VRAM is emitted by detect_hw.py shell mode.
_min_vram="${HW_MIN_VRAM:-0}"
_cur_vram="${HW_VRAM_MB:-0}"
if [ "${HW_HAS_CUDA:-0}" = "1" ] && [ "$_min_vram" -gt 0 ] && [ "$_cur_vram" -lt "$_min_vram" ]; then
    err "Profile '$PROFILE' needs ≥${_min_vram} MB VRAM, but GPU has only ${_cur_vram} MB."
    err "  Fix: use a smaller profile, e.g. PROFILE=8g bash runs/runlocal.sh"
    _preflight_ok=0
fi

# 3. Disk space — rough estimate: ~5 GB per shard + ~10 GB for checkpoints/logs.
_needed_gb=$(( SHARDS * 5 + 10 ))
_avail_kb=$(df -k "$NANOCHAT_BASE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo 999999999)
_avail_gb=$(( _avail_kb / 1024 / 1024 ))
if [ "$_avail_gb" -lt "$_needed_gb" ]; then
    warn "Estimated disk need: ~${_needed_gb} GB, available: ~${_avail_gb} GB in $NANOCHAT_BASE_DIR."
    warn "  The run may fail mid-training due to disk full. Consider using a larger disk."
    warn "  (Continuing anyway — disk estimate is rough.)"
fi

# 4. CUDA availability matches device type
if [ "$DEVICE_TYPE" = "cuda" ] && [ "${HW_HAS_CUDA:-0}" != "1" ]; then
    err "Profile '$PROFILE' requires CUDA, but no CUDA GPU was detected."
    err "  Fix: use cpu profile or attach a GPU."
    _preflight_ok=0
fi

# 5. Driver supports the CUDA runtime used by the selected torch GPU wheel.
if [ "$DEVICE_TYPE" = "cuda" ] && [ "${HW_TORCH_DRIVER_OK:-0}" != "1" ]; then
    err "NVIDIA driver ${HW_DRIVER_VERSION:-unknown} is too old for torch+${HW_TORCH_CUDA:-unknown}."
    err "  Fix: update the Linux NVIDIA driver, use NANOCHAT_TORCH_CUDA=cu126 for pre-Blackwell GPUs, or run PROFILE=cpu."
    _preflight_ok=0
fi

# 6. Selected torch CUDA runtime supports the detected GPU architecture.
if [ "$DEVICE_TYPE" = "cuda" ] && [ "${HW_TORCH_RUNTIME_OK:-0}" != "1" ]; then
    err "Selected torch+${HW_TORCH_CUDA:-unknown} is not compatible with ${HW_SM_ARCH:-unknown} / SM ${HW_COMPUTE_CAP:-unknown}."
    err "  Fix: use NANOCHAT_TORCH_CUDA=cu128 for Blackwell / SM 10.0+, or unset the override."
    _preflight_ok=0
fi

# 7. Python environment health - ensure core imports work before spending time on venv sync
if ! python3 -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'" 2>/dev/null; then
    err "Python 3.10+ is required. Found: $(python3 --version 2>&1)"
    _preflight_ok=0
fi

if [ "$_preflight_ok" -ne 1 ]; then
    err "Pre-flight checks FAILED — fix the issues above and re-run."
    exit 1
fi
log "Pre-flight checks passed."

# ── Parameter summary ─────────────────────────────────────────────────────────
# Print everything that will be used so the user can review before the run starts.
log "============================================================"
log "  nanochat local training"
log "  profile      : $PROFILE  (${HW_SM_ARCH:-unknown} / SM ${HW_COMPUTE_CAP:-?})"
log "  GPU          : ${HW_GPU_NAME:-none}  (${_cur_vram} MB VRAM, ${HW_SM_COUNT:-?} SMs)"
log "  torch cuda   : ${HW_TORCH_CUDA:-cpu}  (--extra $UV_EXTRAS)"
log "  model        : depth=$DEPTH  seq=$SEQ_LEN  micro-batch=$DEVICE_BATCH"
log "  batch        : $TOTAL_BATCH tokens/step  ($GRAD_ACCUM grad-accum steps)"
log "  dataset      : $SHARDS shards  (~$(( SHARDS * 62 / 1000 ))B tokens on disk)"
log "  artifacts    : $NANOCHAT_BASE_DIR"
log "  run name     : ${RUN:-(auto timestamp)}"
if [ -n "${RESUME_FROM_STEP:-}" ]; then
    log "  base resume  : step=$RESUME_FROM_STEP"
fi
log "  cpu threads  : OMP_NUM_THREADS=$OMP_NUM_THREADS  (nproc=$_NCPU)"
log "  tokenizer    : threads=$TOKENIZER_THREADS  loader-buffer=$LOADER_BUFFER_SIZE"
log "  sft          : buffer=$SFT_BUFFER_SIZE  save-every=$SFT_SAVE_EVERY"
log "                 eval-every=$SFT_EVAL_EVERY  chatcore-every=$SFT_CHATCORE_EVERY"
log "  disk needed  : ~${_needed_gb} GB  (available: ~${_avail_gb} GB)"
log "============================================================"

# ── venv setup ───────────────────────────────────────────────────────────────
command -v uv &>/dev/null || { err "uv not found - install from https://docs.astral.sh/uv/"; exit 1; }
cd "$REPO_ROOT"
[ -d ".venv" ] || uv venv
uv sync --extra "$UV_EXTRAS" --quiet
source .venv/bin/activate

# flash-attn is compiled from source and not in pyproject.toml, so uv sync removes it.
# Restore only a wheel that matches the current torch CUDA runtime and GPU SM list.
_FA_KEY_PREFIX=""
if [ "$DEVICE_TYPE" = "cuda" ]; then
    _FA_KEY_PREFIX=$(python - <<'PY' 2>/dev/null || true
import torch

if not torch.cuda.is_available():
    raise SystemExit
torch_version = torch.__version__.split("+")[0]
cuda_version = (torch.version.cuda or "cpu").replace(".", "")
caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
sm = "_".join("sm" + "".join(str(x) for x in cap) for cap in caps)
print(f"torch{torch_version}_cu{cuda_version}_{sm}_")
PY
)
fi
if [ -n "$_FA_KEY_PREFIX" ]; then
    _FA_WHEEL=$(ls -t "$_PROJECT_DATA/wheels/flash-attn_${_FA_KEY_PREFIX}"*/flash_attn*.whl 2>/dev/null | head -1) || true
else
    _FA_WHEEL=""
fi
if [ -n "$_FA_WHEEL" ] && ! python -c "import flash_attn" 2>/dev/null; then
    uv pip install --python .venv/bin/python "$_FA_WHEEL" --no-deps --quiet 2>/dev/null && \
        log "flash-attn restored ($(basename "$(dirname "$_FA_WHEEL")"))" || true
elif [ "$DEVICE_TYPE" = "cuda" ] && [ -n "$_FA_KEY_PREFIX" ] && ! python -c "import flash_attn" 2>/dev/null; then
    log "No matching flash-attn cached wheel for ${_FA_KEY_PREFIX}*; run 'make install-fa' to build one."
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

# ── Helper: verify a checkpoint directory has at least one model file ─────────
_check_checkpoint() {
    local label="$1" dir="$2"
    if [ ! -d "$dir" ] || ! ls "$dir"/*/model_*.pt &>/dev/null 2>&1; then
        err "$label checkpoint not found in $dir"
        err "  The previous stage may have failed or been stopped before saving."
        return 1
    fi
    return 0
}

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

# Verify tokenizer was actually created before continuing
_TOK_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
if [ ! -f "$_TOK_FILE" ]; then
    err "Tokenizer file not found at $_TOK_FILE after tok_train."
    err "  tok_train may have failed — check the output above for errors."
    kill "$DATASET_PID" 2>/dev/null || true
    exit 1
fi
log "  Tokenizer OK: $_TOK_FILE"

# ── Base pretrain ─────────────────────────────────────────────────────────────
log "Waiting for dataset download..."
wait "$DATASET_PID"
DATASET_PID=""   # clear so we don't try to kill it in cleanup

log "Pretraining base model (depth=$DEPTH)..."
BASE_TRAIN_ARGS=(
    --depth="$DEPTH"
    --max-seq-len="$SEQ_LEN"
    --device-batch-size="$DEVICE_BATCH"
    --total-batch-size="$TOTAL_BATCH"
    --device-type="$DEVICE_TYPE"
    --window-pattern="$WINDOW_PATTERN"
    --target-param-data-ratio=12
    --eval-every="${NANOCHAT_BASE_EVAL_EVERY:-250}"
    --eval-tokens="${NANOCHAT_BASE_EVAL_TOKENS:-5242880}"
    --core-metric-every=2000
    --core-metric-max-per-task=200
    --sample-every=500
    --save-every=1000
    --tokenizer-threads="$TOKENIZER_THREADS"
    --loader-buffer-size="$LOADER_BUFFER_SIZE"
    --run="$RUN"
)
if [ -n "${RESUME_FROM_STEP:-}" ]; then
    BASE_TRAIN_ARGS+=(--resume-from-step="$RESUME_FROM_STEP")
fi
python -m scripts.base_train "${BASE_TRAIN_ARGS[@]}"

# Verify base checkpoint exists before proceeding to SFT
if ! _check_checkpoint "Base pretrain" "$NANOCHAT_BASE_DIR/base_checkpoints"; then
    err "base_train exited without saving a checkpoint."
    err "  If training was interrupted, resume with:"
    err "    RESUME_FROM_STEP=<last_step> bash runs/runlocal.sh"
    exit 1
fi

# ── Base eval ─────────────────────────────────────────────────────────────────
log "Evaluating base model..."
python -m scripts.base_eval \
    --device-batch-size="$DEVICE_BATCH" \
    --split-tokens=131072 \
    --max-per-task=200

# ── Identity conversations ────────────────────────────────────────────────────
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_FILE" ]; then
    log "Downloading identity conversations..."
    curl -fsSL -o "$IDENTITY_FILE" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    # Basic validation: file must be non-empty and look like JSON lines
    if [ ! -s "$IDENTITY_FILE" ] || ! python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    lines = [l.strip() for l in f if l.strip()]
assert lines, 'empty'
json.loads(lines[0])   # first line must parse as JSON
print(f'  {len(lines)} identity conversations OK')
" "$IDENTITY_FILE" 2>/dev/null; then
        err "Downloaded identity_conversations.jsonl appears empty or corrupt."
        err "  Remove it and re-run: rm '$IDENTITY_FILE'"
        exit 1
    fi
fi
log "  Identity conversations: $IDENTITY_FILE"

# ── SFT ───────────────────────────────────────────────────────────────────────
log "Supervised fine-tuning (SFT)..."
python -m scripts.chat_sft \
    --device-batch-size="$DEVICE_BATCH" \
    --loader-buffer-size="$SFT_BUFFER_SIZE" \
    --eval-every="$SFT_EVAL_EVERY" \
    --chatcore-every="$SFT_CHATCORE_EVERY" \
    --eval-tokens="$SFT_EVAL_TOKENS" \
    --save-every="$SFT_SAVE_EVERY" \
    --run="$RUN"

# Verify SFT checkpoint exists
if ! _check_checkpoint "SFT" "$NANOCHAT_BASE_DIR/chatsft_checkpoints"; then
    err "chat_sft exited without saving a checkpoint."
    err "  If training was interrupted, find the last saved step and resume with:"
    err "    python -m scripts.chat_sft --resume-from-step <step> \\"
    err "        --device-batch-size=$DEVICE_BATCH --save-every=$SFT_SAVE_EVERY"
    exit 1
fi

# ── SFT eval ──────────────────────────────────────────────────────────────────
log "Evaluating SFT model..."
python -m scripts.chat_eval -i sft

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
