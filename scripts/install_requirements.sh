#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_HELPER="${REPO_ROOT}/src/selfsuvis/scripts/shell_helpers.py"

DEPENDENCY_GROUPS=${1:-vision,dev}
VENV_PATH=${2:-.venv}
PYTHON_BIN="$VENV_PATH/bin/python"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found at $VENV_PATH" >&2
  exit 1
fi

# Ensure pip is available in the venv (uv-created venvs do not include it by default)
uv pip install --python "$VENV_PATH" pip

PACKAGE_SPEC="."
if [[ -n "$DEPENDENCY_GROUPS" ]]; then
  PACKAGE_SPEC=".[${DEPENDENCY_GROUPS}]"
fi

# Detect if the uv cache lives on a different filesystem than the venv.
# Cross-device hardlinks are not allowed, so we fall back to copy mode.
_detect_uv_link_mode() {
  local venv_dir="$1"
  local cache_dir="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"

  # Walk up to an existing ancestor if the cache dir itself doesn't exist yet.
  while [[ -n "$cache_dir" && ! -d "$cache_dir" ]]; do
    cache_dir="$(dirname "$cache_dir")"
  done

  local venv_dev cache_dev
  venv_dev=$(stat -c '%d' "$(dirname "$venv_dir")" 2>/dev/null || echo "")
  cache_dev=$(stat -c '%d' "$cache_dir" 2>/dev/null || echo "")

  if [[ -n "$venv_dev" && -n "$cache_dev" && "$venv_dev" != "$cache_dev" ]]; then
    echo "copy"
  fi
}

_UV_LINK_MODE=$(_detect_uv_link_mode "$VENV_PATH")
if [[ -n "$_UV_LINK_MODE" ]]; then
  echo "Cross-device uv cache detected (cache and .venv are on different disks) → using --link-mode=copy"
  export UV_LINK_MODE=copy
fi

detect_gpu_compute_capability() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | head -1 | tr -d ' '
  fi
}

# Returns the minimum torch CUDA index required for the given compute capability.
# e.g. "12.0" → "cu128" (Blackwell needs CUDA 12.8+ kernels).
# Returns "" if no special minimum applies (existing driver-based mapping suffices).
_min_torch_index_for_compute_cap() {
  local cc="$1"
  local major="${cc%%.*}"
  case "$major" in
    12) echo "cu128" ;;
    13) echo "cu130" ;;
    *) echo "" ;;
  esac
}

# True (exit 0) if cuda index $1 is strictly less than $2 (e.g. cu121 < cu128).
_cuda_index_lt() {
  local a="${1#cu}" b="${2#cu}"
  [[ "${a:-0}" -lt "${b:-0}" ]]
}

detect_cuda_version() {
  local cuda_version=""

  # Parse regular nvidia-smi text output (--query-gpu=cuda_version is not a valid field)
  if command -v nvidia-smi >/dev/null 2>&1; then
    cuda_version=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
  fi

  # Fallback: nvcc reports the installed toolkit version
  if [[ -z "$cuda_version" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_version=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
  fi

  # Fallback: /usr/local/cuda version files
  if [[ -z "$cuda_version" ]] && [[ -d /usr/local/cuda ]]; then
    if [[ -f /usr/local/cuda/version.json ]]; then
      cuda_version=$("$PYTHON_BIN" "$PY_HELPER" cuda-version-from-json --path /usr/local/cuda/version.json 2>/dev/null)
    fi
    if [[ -z "$cuda_version" ]] && [[ -f /usr/local/cuda/version.txt ]]; then
      cuda_version=$(sed -n 's/.*CUDA Version \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' /usr/local/cuda/version.txt | head -n1)
    fi
  fi

  echo "$cuda_version"
}

map_cuda_to_torch_index() {
  local cuda_version="$1"
  case "$cuda_version" in
    13.*) echo "cu126" ;;
    12.9*) echo "cu126" ;;
    12.8*) echo "cu126" ;;
    12.7*) echo "cu126" ;;
    12.6*) echo "cu126" ;;
    12.5*) echo "cu126" ;;
    12.4*) echo "cu124" ;;
    12.3*) echo "cu121" ;;
    12.2*) echo "cu121" ;;
    12.1*) echo "cu121" ;;
    12.0*) echo "cu121" ;;
    11.8*) echo "cu118" ;;
    11.7*) echo "cu118" ;;
    11.6*) echo "cu118" ;;
    *) echo "" ;;
  esac
}

# Install the package plus extras from pyproject.toml, which is the single
# source of truth for Python dependencies.
uv pip install --python "$VENV_PATH" -e "$PACKAGE_SPEC"

CUDA_VERSION=$(detect_cuda_version)

# Allow caller to force a specific CUDA index (e.g. FORCE_CUDA=cu126 or FORCE_CUDA=1 for auto-latest)
if [[ -n "${FORCE_CUDA:-}" ]]; then
  if [[ "$FORCE_CUDA" == "1" ]]; then
    TORCH_CUDA_INDEX="cu126"
  else
    TORCH_CUDA_INDEX="$FORCE_CUDA"
  fi
else
  TORCH_CUDA_INDEX=$(map_cuda_to_torch_index "$CUDA_VERSION")
fi

# Override torch index if the GPU's compute capability requires newer CUDA kernels
# than the driver-version mapping above selected.
# Example: CUDA 13.2 driver → cu126 by default, but Blackwell (sm_120) needs cu128.
_GPU_CC=$(detect_gpu_compute_capability)
if [[ -n "$_GPU_CC" ]]; then
  _MIN_FOR_CC=$(_min_torch_index_for_compute_cap "$_GPU_CC")
  if [[ -n "$_MIN_FOR_CC" ]]; then
    if [[ -z "$TORCH_CUDA_INDEX" ]] || _cuda_index_lt "$TORCH_CUDA_INDEX" "$_MIN_FOR_CC"; then
      echo "GPU compute capability ${_GPU_CC} (sm_${_GPU_CC/./}) requires ${_MIN_FOR_CC}+ kernels"
      echo "  driver-based index '${TORCH_CUDA_INDEX:-none}' does not include sm_${_GPU_CC/./} support"
      echo "  → upgrading TORCH_CUDA_INDEX to ${_MIN_FOR_CC}"
      TORCH_CUDA_INDEX="$_MIN_FOR_CC"
    fi
  fi
fi

if [[ -n "$TORCH_CUDA_INDEX" ]]; then
  echo "CUDA $CUDA_VERSION detected → installing torch with $TORCH_CUDA_INDEX wheels"
  if ! uv pip install --python "$VENV_PATH" --upgrade --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX" torch==2.10.0 torchvision==0.25.0; then
    echo "CUDA wheel install failed, falling back to CPU"
    uv pip install --python "$VENV_PATH" --upgrade --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
  fi
else
  echo "No CUDA detected → installing CPU-only torch"
  uv pip install --python "$VENV_PATH" --upgrade --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
fi

# flash-attn must be installed AFTER torch (its setup.py imports torch at build time).
# --no-build-isolation lets it find the torch headers already in the active environment.
# Skipped on CPU-only machines; non-fatal on failure (models fall back to sdpa attention).

# Compute safe ninja job count for flash-attn CUDA compilation.
#
# Each parallel CUDA compilation unit (translation unit) can peak at 10-14 GiB of
# RAM — flash-attn's backward-pass kernels are especially large.  Running too many
# jobs exhausts RAM and causes the kernel to swap or OOM-kill the compiler.
#
# Heuristic:
#   1. Read total and available RAM from /proc/meminfo.
#   2. Reserve RAM_RESERVE_FRAC of *total* RAM as a hard floor (protects the OS,
#      running models, and the desktop from being crowded out).
#   3. Divide usable available RAM by RAM_PER_JOB_GB to get memory-capped jobs.
#   4. Cap at (nproc - 2) / 2 — each ninja job forks ~1 compiler thread, so N
#      jobs saturate ~2N cores; reserve 2 cores for the OS.
#   5. Always emit at least 1 job.
#
# Tune RAM_PER_JOB_GB if your GPU arch produces larger TUs.  flash-attn backward
# kernels peak well above the average; 12 GiB per job is conservative enough to
# avoid swapping even at peak compile-time working set.  Use a lower value only
# if profiling confirms lower peak RSS during compilation:
#   RTX 3090 / 4090 (Ampere/Ada, large kernels) → 10-14 GiB per job
#   RTX 3070 / 4070 / A-series mid-range        →  8-12 GiB per job
_compute_flash_attn_jobs() {
  local ram_per_job_gb=${FLASH_ATTN_RAM_PER_JOB_GB:-12}
  local reserve_frac="0.20"   # keep 20 % of total RAM free as a hard floor

  # /proc/meminfo values are in kB
  local total_kb avail_kb
  total_kb=$(awk '/^MemTotal:/  { print $2 }' /proc/meminfo 2>/dev/null || echo "8388608")
  avail_kb=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo 2>/dev/null || echo "4194304")

  local cpu_cores
  cpu_cores=$(nproc 2>/dev/null || echo "4")

  "$PYTHON_BIN" "$PY_HELPER" compute-flash-attn-jobs \
    --total-kb "$total_kb" \
    --avail-kb "$avail_kb" \
    --cpu-cores "$cpu_cores" \
    --ram-per-job-gb "$ram_per_job_gb" \
    --reserve-frac "$reserve_frac"
}

# Ensure flash-attn and other JIT-compiled CUDA extensions are built with a toolkit
# that supports the current GPU.  On systems with multiple CUDA installations (e.g.
# /usr/bin/nvcc at 12.0 and /usr/local/cuda-13/bin/nvcc at 13.2), the system PATH
# may point at an older toolkit that doesn't support new architectures (Blackwell sm_120).
# Setting CUDA_HOME here overrides nvcc selection for the entire flash-attn build.
# On machines where the system nvcc already supports the GPU this is a no-op.
_GPU_CC=$(detect_gpu_compute_capability)
_CC_MAJOR="${_GPU_CC%%.*}"
_FORCE_SOURCE_BUILD=false

# Resolve the CUDA toolkit whose nvcc version matches torch.version.cuda.
# flash-attn's setup.py rejects a mismatch between nvcc and torch.version.cuda.
# The matching toolkit is also guaranteed to compile for the correct runtime ABI.
#
# On a standard machine (sm_80/sm_90) the system nvcc typically already matches
# torch — this block sets CUDA_HOME and exits quickly.
# On a Blackwell machine (sm_120) the system nvcc may be older than torch expects;
# the block locates the correct toolkit (e.g. /usr/local/cuda-12.8 after installing
# cuda-nvcc-12-8) and sets CUDA_HOME to it.
_TORCH_CUDA_VER=$("$PYTHON_BIN" -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
if [[ -n "$_TORCH_CUDA_VER" ]]; then
  _TORCH_CUDA_MAJOR="${_TORCH_CUDA_VER%%.*}"
  _TORCH_CUDA_MINOR="${_TORCH_CUDA_VER##*.}"
  # Check system nvcc first — if it already matches, use it.
  _sys_nvcc_ver=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
  if [[ "$_sys_nvcc_ver" != "$_TORCH_CUDA_VER" ]]; then
    # System nvcc doesn't match torch — search installed toolkits for a match.
    _found_home=""
    for _d in \
        "/usr/local/cuda-${_TORCH_CUDA_VER}" \
        "/usr/local/cuda-${_TORCH_CUDA_MAJOR}.${_TORCH_CUDA_MINOR}" \
        "/usr/local/cuda-${_TORCH_CUDA_MAJOR}-${_TORCH_CUDA_MINOR}"; do
      if [[ -x "$_d/bin/nvcc" ]]; then
        _found_home="$_d"; break
      fi
    done
    if [[ -n "$_found_home" ]]; then
      echo "CUDA_HOME → $_found_home  (nvcc ${_TORCH_CUDA_VER} matches torch)"
      export CUDA_HOME="$_found_home"
      export PATH="$_found_home/bin:$PATH"
    else
      echo "WARNING: system nvcc is ${_sys_nvcc_ver:-unknown}, torch expects ${_TORCH_CUDA_VER}."
      echo "  flash-attn may fail to build.  Fix: sudo apt-get install cuda-nvcc-${_TORCH_CUDA_MAJOR}-${_TORCH_CUDA_MINOR}"
      echo "  Then re-run: make venv"
    fi
  fi
fi

# For new GPU architectures (sm_12x Blackwell and later) that have no prebuilt
# flash-attn wheel, force a source build and include all common arch targets so
# the resulting wheel also runs on more powerful machines (sm_80/sm_90).
if [[ -n "$_CC_MAJOR" ]] && [[ "$_CC_MAJOR" -ge 12 ]]; then
  _INSTALLED_SO=$(find "${VENV_PATH}" -name "flash_attn_2_cuda*.so" 2>/dev/null | head -1)
  _NEEDS_REBUILD=true
  if [[ -n "$_INSTALLED_SO" ]] && command -v cuobjdump >/dev/null 2>&1; then
    if cuobjdump "$_INSTALLED_SO" 2>/dev/null | grep -q "sm_${_CC_MAJOR}"; then
      _NEEDS_REBUILD=false
      echo "flash-attn: sm_${_CC_MAJOR}0 already in installed binary — skipping rebuild"
    fi
  fi
  if $_NEEDS_REBUILD; then
    echo "flash-attn: building from source for sm_${_CC_MAJOR}0 (no prebuilt wheel available)"
    _FORCE_SOURCE_BUILD=true
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;9.0;12.0}"
    find ~/.cache/pip/wheels -name "flash_attn*.whl" -delete 2>/dev/null || true
    find ~/.cache/uv         -name "flash_attn*.whl" -delete 2>/dev/null || true
  fi
fi

if "$PYTHON_BIN" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  # ninja parallelises CUDA kernel compilation — without it flash-attn takes 10-30 min.
  # Check system ninja first, then fall back to the pip-bundled wheel.
  if command -v ninja >/dev/null 2>&1; then
    echo "ninja found at $(command -v ninja) — flash-attn compilation will be fast."
  else
    echo "ninja not found in PATH — installing via pip (speeds up flash-attn build) …"
    if "$PYTHON_BIN" -m pip install ninja -q; then
      echo "ninja (pip) installed."
    else
      echo "WARNING: ninja install failed. flash-attn will compile without it (slow, ~30 min)."
      echo "  To fix: sudo apt-get install -y ninja-build"
    fi
  fi

  # Determine safe parallelism
  read -r FLASH_JOBS _total _avail _usable _mem_cap _cpu_cap < <(_compute_flash_attn_jobs)
  echo ""
  echo "── flash-attn compilation resource budget ──────────────────────────────────"
  echo "  RAM total / available / usable : ${_total} GiB / ${_avail} GiB / ${_usable} GiB"
  echo "  RAM per job (FLASH_ATTN_RAM_PER_JOB_GB) : ${FLASH_ATTN_RAM_PER_JOB_GB:-12} GiB"
  echo "  CPU cores   : $(nproc)  →  limit: ${_cpu_cap} ((cores-2)/2, each job forks ~1 extra thread)"
  echo "  Memory jobs : ${_mem_cap}"
  echo "  → MAX_JOBS  : ${FLASH_JOBS}  (min of memory and CPU limits)"
  echo "  (Set FLASH_ATTN_RAM_PER_JOB_GB=N to override per-job RAM estimate)"
  echo "────────────────────────────────────────────────────────────────────────────"
  echo ""

  echo "Installing flash-attn with MAX_JOBS=${FLASH_JOBS} …"
  # torch/torchvision are already installed above from the selected PyTorch index.
  # Do not let pip re-resolve them here, or it may pull a newer torch from PyPI
  # that no longer matches the pinned torchvision CUDA wheel.
  _flash_attn_flags="--no-build-isolation --no-deps"
  if $_FORCE_SOURCE_BUILD; then
    # --no-binary  : build from source (skip any prebuilt wheel for wrong arch)
    # --force-reinstall : override an already-installed sm_80/sm_90 binary
    # --no-cache-dir   : prevent pip from serving the old cached wheel
    _flash_attn_flags="$_flash_attn_flags --no-binary flash-attn --force-reinstall --no-cache-dir"
  fi
  # shellcheck disable=SC2086
  if MAX_JOBS="$FLASH_JOBS" "$PYTHON_BIN" -m pip install flash-attn $_flash_attn_flags -q; then
    echo "flash-attn installed."
  else
    echo "WARNING: flash-attn build failed. Run later with:"
    echo "  python scripts/prepare_models.py --flash-attn"
    echo "Models will use sdpa (PyTorch built-in SDPA) until then."
  fi
else
  echo "No CUDA GPU detected — skipping flash-attn (CPU-only mode)."
fi

# ── TensorRT + onnxruntime-gpu ─────────────────────────────────────────────────
# onnxruntime-gpu replaces the CPU-only onnxruntime installed from pyproject.toml.
# It adds CUDAExecutionProvider and TensorrtExecutionProvider, giving the fastest
# ONNX inference on NVIDIA GPUs.  TensorRT must also be present at runtime for
# TensorrtExecutionProvider to activate; we install it from NVIDIA's PyPI index.
#
# TensorRT CUDA compatibility:
#   CUDA 12.x → TensorRT 10.x  (tensorrt>=10.0 from pypi.nvidia.com)
#   CUDA 11.8 → TensorRT 8.x   (tensorrt<9; use apt or older wheel)
#
# Non-fatal: falls back to CPU onnxruntime if GPU install fails.
if [[ -n "$TORCH_CUDA_INDEX" ]]; then
  echo ""
  echo "── onnxruntime-gpu + TensorRT ────────────────────────────────────────────────"

  # Uninstall CPU-only onnxruntime first to avoid package conflict.
  "$PYTHON_BIN" -m pip uninstall -y onnxruntime 2>/dev/null || true

  echo "Installing onnxruntime-gpu …"
  if "$PYTHON_BIN" -m pip install "onnxruntime-gpu>=1.18.0" -q; then
    echo "  onnxruntime-gpu installed (CUDAExecutionProvider enabled)."
  else
    echo "  WARNING: onnxruntime-gpu install failed — reinstalling CPU onnxruntime."
    "$PYTHON_BIN" -m pip install "onnxruntime>=1.18.0" -q || true
  fi

  # TensorRT wheels are published by NVIDIA at https://pypi.nvidia.com.
  # Version selection:
  #   CUDA 12.x → tensorrt>=10 (cu12 variant)
  #   CUDA 11.8 → tensorrt<9   (cu11 bindings are not on pypi.nvidia.com; use apt)
  echo "Installing TensorRT Python package (enables TensorrtExecutionProvider) …"
  if [[ "$TORCH_CUDA_INDEX" == cu118 ]]; then
    echo "  CUDA 11.8 detected — TensorRT 10+ is not available for CUDA 11."
    echo "  Install TensorRT 8.x via apt from the NVIDIA TensorRT apt repo, then"
    echo "  the TensorrtExecutionProvider will activate automatically."
    echo "  See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
  else
    # CUDA 12.x — use NVIDIA's PyPI index (pypi.nvidia.com publishes cu12 wheels).
    if "$PYTHON_BIN" -m pip install \
        "tensorrt>=10.0" \
        "tensorrt-cu12-bindings" \
        "tensorrt-cu12-libs" \
        --extra-index-url https://pypi.nvidia.com \
        -q; then
      echo "  TensorRT installed (TensorrtExecutionProvider will be active)."
    else
      echo "  WARNING: TensorRT pip install failed. Possible fixes:"
      echo "    pip install tensorrt --extra-index-url https://pypi.nvidia.com"
      echo "    sudo apt-get install -y tensorrt  (requires nvidia-tensorrt apt repo)"
      echo "    See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
      echo "  Without TensorRT, onnxruntime-gpu will use CUDAExecutionProvider only."
    fi
  fi
else
  echo ""
  echo "No CUDA detected — keeping CPU-only onnxruntime (no TensorRT)."
fi
