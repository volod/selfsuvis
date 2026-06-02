#!/usr/bin/env python3
"""
Hardware detection for nanochat - stdlib only, no torch required.

Queries nvidia-smi for VRAM, maps to a training profile, and emits the
parameters needed to pick uv extras and configure the training run.

Usage:
    python3 scripts/detect_hw.py                        # human-readable info table
    python3 scripts/detect_hw.py profile                # profile name: cpu | 8g | ...
    python3 scripts/detect_hw.py extras                 # uv extras: cpu | gpu
    python3 scripts/detect_hw.py shell                  # bash eval-able KEY=VALUE lines
    python3 scripts/detect_hw.py max_jobs               # safe MAX_JOBS for CUDA builds
    python3 scripts/detect_hw.py nvcc_ver               # best nvcc version e.g. "12.6" or "0.0"
    python3 scripts/detect_hw.py cuda_home              # CUDA_HOME for best nvcc or ""
    python3 scripts/detect_hw.py max_sm VER             # max compilable SM for nvcc version
    python3 scripts/detect_hw.py filtered_archs A V     # filter arch list A by nvcc version V
    python3 scripts/detect_hw.py arch                   # architecture name: ada | hopper | blackwell | ...
"""

import os
import re
import shlex
import subprocess
import sys

# ── Profile table ─────────────────────────────────────────────────────────────
# Each row: (min_vram_mb, profile, depth, seq_len, device_batch, total_batch,
#            shards, description)
#
# Sizing rationale:
#   model_dim  = depth * 64 (aspect_ratio default)
#   params ≈   depth * (4 * model_dim² / head_dim + 8 * model_dim²) / 1e6  M
#   VRAM use ≈ params(bf16) + 2×params(adam fp32) + activations(B×T×D×L)
#
#   total_batch = 524288 tokens for all GPU profiles (same as speedrun quality)
#   grad_accum  = total_batch / (device_batch × seq_len)
#   shards      = ceil(12 × params / 62e6) + 10% buffer   (chinchilla optimal)
#
# Profile  depth  model_dim  ~params  seq   bs  accum  shards
# ───────  ─────  ─────────  ───────  ────  ──  ─────  ──────
# 40g        20     1280      430M   2048   16     16    120
# 24g        18     1152      330M   2048    8     32     90
# 16g        14      896      400M   1024   16     32     50
# 16g/small  12      768      300M   1024   16     32     35
# 12g        12      768      110M   1024    4    128     35
# 8g         10      640       65M    512    2    512     25
# cpu         4      256       10M    256    4    16      8   (total_batch=16384)

PROFILES = [
    # min_vram  name   depth  seq   bs  total_batch  shards  description
    (40_000,  "40g",   20,  2048, 16,   524_288,   120,  ">=40 GB - A100 / A6000 / H100"),
    (24_000,  "24g",   18,  2048,  8,   524_288,    90,  "24-39 GB - RTX 3090 / 4090"),
    (16_000,  "16g",   14,  1024, 16,   524_288,    50,  "16-23 GB - RTX 4080 / 4060 Ti 16G"),
    (12_000,  "12g",   12,  1024,  4,   524_288,    35,  "12-15 GB - RTX 4070 / 3080"),
    (     1,   "8g",   10,   512,  2,   524_288,    25,  " 8-11 GB - RTX 3070 / 4060 8G"),
    (     0,  "cpu",    4,   256,  4,    16_384,     8,  "no GPU   - CPU only (slow, educational)"),
]


# ── CUDA toolkit detection ─────────────────────────────────────────────────────
# Maps minimum nvcc version → maximum SM capability it can compile for.
# Entries sorted descending; first match wins.
# Empirically verified: CUDA 12.6 tops at SM 9.0 (compute_100 is not defined).
# CUDA 12.8 added SM 10.0 (Blackwell B100/B200) and SM 12.0 (RTX 5000 / Blackwell Ultra).
_NVCC_SM_TABLE: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((12, 8), (12, 0)),   # CUDA 12.8+: Blackwell B100/B200/RTX 5000 (SM 10.0 + SM 12.0)
    ((11, 8), ( 9, 0)),   # CUDA 11.8-12.7: Hopper H100 (SM 9.0)
    (( 0,  0), ( 8, 9)),  # fallback: Ada Lovelace and older (SM 8.9)
]


def _nvcc_version(nvcc: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=5)
        m = re.search(r"release (\d+)\.(\d+)", out.stdout)
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        return None


def detect_cuda_nvcc() -> tuple[str, tuple[int, int]] | tuple[None, None]:
    """Return (nvcc_path, (major, minor)) for the best (newest) available nvcc."""
    import shutil
    candidates: list[str] = []
    for major in (12, 11):
        for minor in range(9, -1, -1):
            candidates.append(f"/usr/local/cuda-{major}.{minor}/bin/nvcc")
    candidates += ["/usr/local/cuda-12/bin/nvcc", "/usr/local/cuda/bin/nvcc"]
    p = shutil.which("nvcc")
    if p:
        candidates.append(p)
    seen: set[str] = set()
    best: tuple[str, tuple[int, int]] | None = None
    for nvcc in candidates:
        if nvcc in seen:
            continue
        seen.add(nvcc)
        if not os.path.isfile(nvcc) or not os.access(nvcc, os.X_OK):
            continue
        ver = _nvcc_version(nvcc)
        if ver and (best is None or ver > best[1]):
            best = (nvcc, ver)
    return best if best else (None, None)


def nvcc_max_sm(ver: tuple[int, int]) -> tuple[int, int]:
    """Return the max SM (major, minor) compilable by a given nvcc version."""
    for nvcc_min, sm in _NVCC_SM_TABLE:
        if ver >= nvcc_min:
            return sm
    return (8, 9)


def filter_archs_by_nvcc(arches: str, nvcc_ver: tuple[int, int]) -> str:
    """Filter a semicolon-separated TORCH_CUDA_ARCH_LIST to archs the nvcc can compile."""
    max_sm = nvcc_max_sm(nvcc_ver)
    out: list[str] = []
    for a in arches.split(";"):
        a = a.strip()
        if not a:
            continue
        base = a.rstrip("+PTX").rstrip("+ptx")
        parts = base.split(".")
        try:
            sm = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            out.append(a)
            continue
        if sm <= max_sm:
            out.append(a)
    return ";".join(out)


# ── GPU architecture naming ───────────────────────────────────────────────────
# Maps compute capability (major, minor) to human-readable architecture name.
# Used for architecture-aware SFT parameter tuning and display.
_CAP_ARCH: list[tuple[tuple[int, int], str]] = [
    ((12, 0), "blackwell"),   # RTX 5000 series (Blackwell Ultra / GB20x)
    ((10, 0), "blackwell"),   # B100 / B200 (Blackwell)
    (( 9, 0), "hopper"),      # H100 / H200
    (( 8, 9), "ada"),         # RTX 4000 series (Ada Lovelace)
    (( 8, 7), "ampere"),      # Jetson Orin
    (( 8, 6), "ampere"),      # RTX 3000 series (GA106/GA104/GA102)
    (( 8, 0), "ampere"),      # A100 / A30
    (( 7, 5), "turing"),      # RTX 2000 series / T4
    (( 7, 0), "volta"),       # V100
]

def compute_cap_arch(cap: tuple[int, int]) -> str:
    """Map SM compute capability tuple to GPU architecture name."""
    major, _ = cap
    for threshold, name in _CAP_ARCH:
        if cap >= threshold:
            return name
    return "unknown"


# ── GPU detection via nvidia-smi ───────────────────────────────────────────────
def _nvidia_smi(*query_fields: str) -> list[str] | None:
    """Run nvidia-smi and return the first GPU's values, or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(query_fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0]
            return [v.strip() for v in line.split(",")]
    except Exception:
        pass
    return None


def _sm_count_from_name(gpu_name: str) -> int:
    """Best-effort SM count fallback for common NVIDIA cards."""
    name = gpu_name.lower()
    if "4090" in name:
        return 76 if "laptop" in name or "mobile" in name else 128
    if "4080" in name:
        return 58 if "laptop" in name or "mobile" in name else 76
    if "4070 ti" in name:
        return 60
    if "4070" in name:
        return 36 if "laptop" in name or "mobile" in name else 46
    if "4060 ti" in name:
        return 34
    if "4060" in name:
        return 24
    if "3090" in name:
        return 82
    if "3080 ti" in name:
        return 80
    if "3080" in name:
        return 68
    return 0


def _detect_sm_count(gpu_name: str) -> int:
    """Detect SM count without making core VRAM/name detection depend on it."""
    vals = _nvidia_smi("multiprocessor_count")
    if vals:
        try:
            return int(vals[0])
        except ValueError:
            pass
    return _sm_count_from_name(gpu_name)


def detect_compute_cap() -> tuple[int, int]:
    """Return GPU SM compute capability (major, minor), or (0, 0) if unavailable."""
    vals = _nvidia_smi("compute_cap")
    if vals:
        try:
            parts = vals[0].split(".")
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            pass
    return (0, 0)


def detect_gpu() -> tuple[int, str, int, tuple[int, int]]:
    """Return (vram_mb, gpu_name, sm_count, compute_cap) for the first GPU."""
    vals = _nvidia_smi("memory.total", "name")
    if vals and len(vals) >= 2:
        try:
            vram_mb = int(vals[0])
        except ValueError:
            return 0, "", 0, (0, 0)
        gpu_name = vals[1]
        compute_cap = detect_compute_cap()
        return vram_mb, gpu_name, _detect_sm_count(gpu_name), compute_cap
    return 0, "", 0, (0, 0)


def _sft_params(seq: int, total: int, profile_name: str) -> dict:
    """
    Return hardware-appropriate SFT training parameters for a profile.

    sft_buffer        conversations pre-fetched for best-fit packing; larger = less
                      padding but more RAM.
    sft_eval_every    how often to run validation bpb (steps); reduced on small GPUs
                      to avoid spending most of training time in eval.
    sft_chatcore_every  how often to run ChatCORE eval; -1 = disabled.
    sft_eval_tokens   tokens to evaluate val loss on; reduced on small GPUs.
    sft_save_every    how often to write mid-run checkpoints so Ctrl-C is resumable.
                      Smaller on small GPUs (each step is cheaper, more frequent
                      saves don't hurt). 0 = save only at end.
    """
    sft_buffer = max(500, total // seq)   # ≥500, scales inversely with seq_len
    if profile_name in ("40g", "24g"):
        return {
            "sft_buffer": sft_buffer,
            "sft_eval_every": 200,
            "sft_chatcore_every": 200,
            "sft_eval_tokens": 40 * 524_288,
            "sft_save_every": 1000,
        }
    elif profile_name == "16g":
        return {
            "sft_buffer": sft_buffer,
            "sft_eval_every": 500,
            "sft_chatcore_every": 500,
            "sft_eval_tokens": 20 * 524_288,
            "sft_save_every": 500,
        }
    elif profile_name == "12g":
        return {
            "sft_buffer": sft_buffer,
            "sft_eval_every": 500,
            "sft_chatcore_every": -1,
            "sft_eval_tokens": 10 * 524_288,
            "sft_save_every": 500,
        }
    else:  # 8g, cpu
        return {
            "sft_buffer": sft_buffer,
            "sft_eval_every": 1000,
            "sft_chatcore_every": -1,
            "sft_eval_tokens": 5 * 524_288,
            "sft_save_every": 200,
        }


def _profile_dict(
    min_vram: int,
    name: str,
    depth: int,
    seq: int,
    bs: int,
    total: int,
    shards: int,
    desc: str,
) -> dict:
    grad_accum = total // (bs * seq)
    return {
        "profile": name,
        "min_vram": min_vram,
        "depth": depth,
        "seq_len": seq,
        "device_batch": bs,
        "total_batch": total,
        "shards": shards,
        "grad_accum": grad_accum,
        "description": desc,
        **_sft_params(seq, total, name),
    }


def _adapt_profile(profile: dict, sm_count: int, compute_cap: tuple[int, int] = (0, 0)) -> dict:
    """Tune the selected VRAM profile for compute capacity and GPU architecture."""
    arch = compute_cap_arch(compute_cap)
    p = dict(profile)

    # 16g on low-SM GPU: step down to depth=12 to avoid compute bottleneck
    if profile["profile"] == "16g" and 0 < sm_count <= 40:
        p.update({
            "depth": 12,
            "shards": 35,
            "grad_accum": profile["total_batch"] // (16 * profile["seq_len"]),
            "description": "16 GB low-SM GPU - compact depth=12 profile",
        })

    # Blackwell (RTX 5000 / B100 / B200): significantly higher compute density
    # per SM lets us afford a larger SFT buffer and more frequent eval even on
    # mid-range cards.
    if arch == "blackwell":
        p["sft_buffer"] = min(4000, p["sft_buffer"] * 2)
        if profile["profile"] == "12g":
            p["sft_chatcore_every"] = 500   # fast enough to enable ChatCORE

    # Hopper (H100): high SM count and HBM3 bandwidth — same as large GPU profile
    if arch == "hopper" and profile["profile"] != "40g":
        p["sft_chatcore_every"] = max(200, p["sft_chatcore_every"])

    return p


def select_profile(vram_mb: int, sm_count: int = 0, compute_cap: tuple[int, int] = (0, 0)) -> dict:
    """Choose the best training profile for the given VRAM, SM count and architecture."""
    for min_vram, name, depth, seq, bs, total, shards, desc in PROFILES:
        if vram_mb >= min_vram:
            return _adapt_profile(
                _profile_dict(min_vram, name, depth, seq, bs, total, shards, desc),
                sm_count,
                compute_cap,
            )
    return select_profile(0, 0)  # safety fallback to cpu


def select_profile_name(name: str) -> dict:
    """Return an explicit, non-adaptive profile by name (no architecture tuning)."""
    for min_vram, profile, depth, seq, bs, total, shards, desc in PROFILES:
        if profile == name:
            return _profile_dict(min_vram, profile, depth, seq, bs, total, shards, desc)
    valid = ", ".join(row[1] for row in PROFILES)
    raise SystemExit(f"Unknown profile {name!r}. Valid: {valid}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "info"
    explicit_profile = sys.argv[2] if mode in {"shell", "info"} and len(sys.argv) > 2 else ""
    vram_mb, gpu_name, sm_count, compute_cap = detect_gpu()
    has_cuda = vram_mb > 0
    arch = compute_cap_arch(compute_cap)
    p = select_profile_name(explicit_profile) if explicit_profile else select_profile(vram_mb, sm_count, compute_cap)

    if mode == "profile":
        print(p["profile"])

    elif mode == "extras":
        print("gpu" if has_cuda else "cpu")

    elif mode == "max_jobs":
        # min(max(1,(nproc-2)//2), max(1,total_ram_gb//denom))
        # denom=20 (<64 GB), denom=16 (>=64 GB). Safe default when RAM unreadable.
        nproc = os.cpu_count() or 4
        try:
            with open("/proc/meminfo") as f:
                ram_kb = int(f.readline().split()[1])
            ram_gb = ram_kb // (1024 * 1024)
        except Exception:
            ram_gb = 8
        denom = 20 if ram_gb < 64 else 16
        print(min(max(1, (nproc - 2) // 2), max(1, ram_gb // denom)))

    elif mode == "nvcc_ver":
        _, ver = detect_cuda_nvcc()
        print(f"{ver[0]}.{ver[1]}" if ver else "0.0")

    elif mode == "cuda_home":
        nvcc, _ = detect_cuda_nvcc()
        print(os.path.dirname(os.path.dirname(nvcc)) if nvcc else "")

    elif mode == "max_sm":
        # Usage: max_sm 12.6  → prints "8.9"
        ver_str = sys.argv[2] if len(sys.argv) > 2 else "0.0"
        try:
            parts = ver_str.split(".")
            ver = (int(parts[0]), int(parts[1]))
        except Exception:
            ver = (0, 0)
        sm = nvcc_max_sm(ver)
        print(f"{sm[0]}.{sm[1]}")

    elif mode == "filtered_archs":
        # Usage: filtered_archs "8.9;12.0" "12.6"  → prints "8.9"
        arches_arg = sys.argv[2] if len(sys.argv) > 2 else ""
        ver_str = sys.argv[3] if len(sys.argv) > 3 else "0.0"
        try:
            parts = ver_str.split(".")
            ver = (int(parts[0]), int(parts[1]))
        except Exception:
            ver = (0, 0)
        print(filter_archs_by_nvcc(arches_arg, ver))

    elif mode == "arch":
        print(arch)

    elif mode == "shell":
        # Designed for: eval "$(python3 scripts/detect_hw.py shell)"
        cap_str = f"{compute_cap[0]}.{compute_cap[1]}" if compute_cap != (0, 0) else "unknown"
        print(f"HW_GPU_NAME={shlex.quote(gpu_name)}")
        print(f"HW_VRAM_MB={vram_mb}")
        print(f"HW_SM_COUNT={sm_count}")
        print(f"HW_HAS_CUDA={'1' if has_cuda else '0'}")
        print(f"HW_EXTRAS={'gpu' if has_cuda else 'cpu'}")
        print(f"HW_COMPUTE_CAP={shlex.quote(cap_str)}")
        print(f"HW_SM_ARCH={shlex.quote(arch)}")
        print(f"HW_PROFILE={p['profile']}")
        print(f"HW_DEPTH={p['depth']}")
        print(f"HW_SEQ_LEN={p['seq_len']}")
        print(f"HW_DEVICE_BATCH={p['device_batch']}")
        print(f"HW_TOTAL_BATCH={p['total_batch']}")
        print(f"HW_SHARDS={p['shards']}")
        print(f"HW_GRAD_ACCUM={p['grad_accum']}")
        # SFT-specific parameters (used by runlocal.sh → chat_sft.py)
        print(f"HW_MIN_VRAM={p['min_vram']}")
        print(f"HW_SFT_BUFFER={p['sft_buffer']}")
        print(f"HW_SFT_EVAL_EVERY={p['sft_eval_every']}")
        print(f"HW_SFT_CHATCORE_EVERY={p['sft_chatcore_every']}")
        print(f"HW_SFT_EVAL_TOKENS={p['sft_eval_tokens']}")
        print(f"HW_SFT_SAVE_EVERY={p['sft_save_every']}")

    else:  # info
        tokens_b = p["shards"] * 62 / 1000
        cap_str = f"{compute_cap[0]}.{compute_cap[1]}" if compute_cap != (0, 0) else "n/a"
        print(f"  GPU      : {gpu_name or 'none'}")
        print(f"  VRAM     : {vram_mb:,} MB  ({vram_mb / 1024:.1f} GB)")
        if sm_count:
            print(f"  SMs      : {sm_count}")
        print(f"  Arch     : {arch}  (SM {cap_str})")
        print(f"  Profile  : {p['profile']}  ({p['description']})")
        print(f"  Install  : uv sync --extra {'gpu' if has_cuda else 'cpu'}")
        print(f"  Model    : depth={p['depth']}  seq={p['seq_len']}  "
              f"micro-batch={p['device_batch']}")
        print(f"  Batch    : {p['total_batch']:,} tokens/step  "
              f"({p['grad_accum']} grad-accum steps)")
        print(f"  Dataset  : {p['shards']} shards  (~{tokens_b:.1f}B tokens)")
        print(f"  SFT      : buffer={p['sft_buffer']}  eval-every={p['sft_eval_every']}"
              f"  chatcore={p['sft_chatcore_every']}  save-every={p['sft_save_every']}")
        nvcc, nvcc_ver = detect_cuda_nvcc()
        if nvcc_ver:
            sm = nvcc_max_sm(nvcc_ver)
            ver_str = f"{nvcc_ver[0]}.{nvcc_ver[1]}"
            sm_str = f"{sm[0]}.{sm[1]}"
            fa_tag = "latest" if nvcc_ver >= (12, 8) else "v2.7.4 (nvcc < 12.8)"
            print(f"  nvcc     : {nvcc}  (CUDA {ver_str})")
            print(f"  Max SM   : {sm_str}  flash-attn={fa_tag}")
        else:
            print("  nvcc     : not found - flash-attn build unavailable")


if __name__ == "__main__":
    main()
