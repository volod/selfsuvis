#!/usr/bin/env python3
"""
Hardware detection for nanochat — stdlib only, no torch required.

Queries nvidia-smi for VRAM, maps to a training profile, and emits the
parameters needed to pick uv extras and configure the training run.

Usage:
    python3 scripts/detect_hw.py            # human-readable info table
    python3 scripts/detect_hw.py profile    # profile name: cpu | 8g | 12g | 16g | 24g | 40g
    python3 scripts/detect_hw.py extras     # uv extras:   cpu | gpu
    python3 scripts/detect_hw.py shell      # bash eval-able KEY=VALUE lines
    python3 scripts/detect_hw.py max_jobs   # safe MAX_JOBS for CUDA source builds
"""

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
# 16g        14      896      160M   1024    8     64     50
# 12g        12      768      110M   1024    4    128     35
# 8g         10      640       65M    512    2    512     25
# cpu         4      256       10M    256    4    16      8   (total_batch=16384)

PROFILES = [
    # min_vram  name   depth  seq   bs  total_batch  shards  description
    (40_000,  "40g",   20,  2048, 16,   524_288,   120,  "≥40 GB — A100 / A6000 / H100"),
    (24_000,  "24g",   18,  2048,  8,   524_288,    90,  "24–39 GB — RTX 3090 / 4090"),
    (16_000,  "16g",   14,  1024,  8,   524_288,    50,  "16–23 GB — RTX 4080 / 4060 Ti 16G"),
    (12_000,  "12g",   12,  1024,  4,   524_288,    35,  "12–15 GB — RTX 4070 / 3080"),
    (     1,   "8g",   10,   512,  2,   524_288,    25,  " 8–11 GB — RTX 3070 / 4060 8G"),
    (     0,  "cpu",    4,   256,  4,    16_384,     8,  "no GPU   — CPU only (slow, educational)"),
]


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


def detect_gpu() -> tuple[int, str]:
    """Return (vram_mb, gpu_name) for the first GPU, or (0, '') if no GPU found."""
    vals = _nvidia_smi("memory.total", "name")
    if vals and len(vals) >= 2:
        try:
            return int(vals[0]), vals[1]
        except ValueError:
            pass
    return 0, ""


def select_profile(vram_mb: int) -> dict:
    """Choose the best training profile for the given VRAM."""
    for min_vram, name, depth, seq, bs, total, shards, desc in PROFILES:
        if vram_mb >= min_vram:
            grad_accum = total // (bs * seq)
            return {
                "profile":      name,
                "depth":        depth,
                "seq_len":      seq,
                "device_batch": bs,
                "total_batch":  total,
                "shards":       shards,
                "grad_accum":   grad_accum,
                "description":  desc,
            }
    return select_profile(0)  # safety fallback to cpu


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "info"
    vram_mb, gpu_name = detect_gpu()
    has_cuda = vram_mb > 0
    p = select_profile(vram_mb)

    if mode == "profile":
        print(p["profile"])

    elif mode == "extras":
        print("gpu" if has_cuda else "cpu")

    elif mode == "max_jobs":
        # CLAUDE.md rule: min(max(1,(nproc-2)//2), max(1,available_ram_gb//12))
        # Safe default (4) when RAM can't be determined.
        import os
        nproc = os.cpu_count() or 4
        try:
            with open("/proc/meminfo") as f:
                ram_kb = int(f.readline().split()[1])
            ram_gb = ram_kb // (1024 * 1024)
        except Exception:
            ram_gb = 8
        print(min(max(1, (nproc - 2) // 2), max(1, ram_gb // 12)))

    elif mode == "shell":
        # Designed for: eval "$(python3 scripts/detect_hw.py shell)"
        print(f'HW_GPU_NAME="{gpu_name}"')
        print(f"HW_VRAM_MB={vram_mb}")
        print(f"HW_HAS_CUDA={'1' if has_cuda else '0'}")
        print(f"HW_EXTRAS={'gpu' if has_cuda else 'cpu'}")
        print(f"HW_PROFILE={p['profile']}")
        print(f"HW_DEPTH={p['depth']}")
        print(f"HW_SEQ_LEN={p['seq_len']}")
        print(f"HW_DEVICE_BATCH={p['device_batch']}")
        print(f"HW_TOTAL_BATCH={p['total_batch']}")
        print(f"HW_SHARDS={p['shards']}")

    else:  # info
        tokens_b = p["shards"] * 62 / 1000
        print(f"  GPU      : {gpu_name or 'none'}")
        print(f"  VRAM     : {vram_mb:,} MB  ({vram_mb / 1024:.1f} GB)")
        print(f"  Profile  : {p['profile']}  ({p['description']})")
        print(f"  Install  : uv sync --extra {'gpu' if has_cuda else 'cpu'}")
        print(f"  Model    : depth={p['depth']}  seq={p['seq_len']}  "
              f"micro-batch={p['device_batch']}")
        print(f"  Batch    : {p['total_batch']:,} tokens/step  "
              f"({p['grad_accum']} grad-accum steps)")
        print(f"  Dataset  : {p['shards']} shards  (~{tokens_b:.1f}B tokens)")


if __name__ == "__main__":
    main()
