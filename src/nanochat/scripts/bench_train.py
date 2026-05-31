#!/usr/bin/env python3
"""
Training speed micro-benchmark for nanochat.

Runs forward+backward passes through transformer-shaped matmuls at the
detected profile's exact model dimensions, then estimates full training
wall time for the current hardware.

Called automatically by `make hw-info` when a venv is available.
Requires torch; gracefully skips if torch is not installed yet.
"""
import sys
import time
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------

def _get_hw() -> dict:
    """Call detect_hw.py in shell mode using the same Python interpreter."""
    script = Path(__file__).parent / "detect_hw.py"
    try:
        r = subprocess.run(
            [sys.executable, str(script), "shell"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            return {
                k: v.strip('"')
                for line in r.stdout.strip().splitlines()
                for k, _, v in [line.partition("=")]
            }
    except Exception:
        pass
    return {}


def _fmt_duration(s: float) -> str:
    if s < 3_600:
        return f"{s / 60:.0f} min"
    if s < 86_400:
        return f"{s / 3_600:.1f} h"
    return f"{s / 86_400:.1f} days"


# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import torch
    except ImportError:
        print("  Benchmark : torch not available — run 'make venv' first")
        return

    hw = _get_hw()
    depth        = int(hw.get("HW_DEPTH",        12))
    seq_len      = int(hw.get("HW_SEQ_LEN",    1024))
    device_batch = int(hw.get("HW_DEVICE_BATCH",   4))
    total_batch  = int(hw.get("HW_TOTAL_BATCH", 524_288))
    shards       = int(hw.get("HW_SHARDS",        35))

    D  = depth * 64           # model_dim (aspect_ratio = 64, matches nanochat default)
    BT = device_batch * seq_len   # tokens per micro-step on this device

    # Chinchilla-optimal training horizon (same logic as runlocal.sh)
    total_tokens = shards * 62_000_000
    num_iters    = total_tokens // total_batch

    # ── device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        dev   = torch.device("cuda", 0)
        sync  = torch.cuda.synchronize
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        dev   = torch.device("mps")
        sync  = lambda: None
        dtype = torch.float32
    else:
        dev   = torch.device("cpu")
        sync  = lambda: None
        dtype = torch.float32

    # ── representative weight matrices ────────────────────────────────────────
    # One set of shared weights is reused across all depth iterations.
    # Shapes match the real model exactly — same matmul cost per layer.
    # Sharing keeps VRAM near zero so the benchmark fits any GPU.
    W_qkv = torch.randn(D, 3 * D, device=dev, dtype=dtype).requires_grad_(True)
    W_o   = torch.randn(D,     D, device=dev, dtype=dtype).requires_grad_(True)
    W_up  = torch.randn(D, 4 * D, device=dev, dtype=dtype).requires_grad_(True)
    W_dn  = torch.randn(4 * D, D, device=dev, dtype=dtype).requires_grad_(True)
    x0    = torch.randn(BT, D, device=dev, dtype=dtype)

    # One training step: depth iterations of {attn projections + MLP} fwd+bwd.
    # Skips actual attention scores (O(T²) term); they add ~12% overhead on top
    # and are accounted for by the ATTN_OVERHEAD correction below.
    def step():
        x = x0
        for _ in range(depth):
            x = (x @ W_qkv)[:, :D] @ W_o   # QKV projection + output projection
            x = torch.relu(x @ W_up) @ W_dn  # MLP (relu^2 approx)
        x.sum().backward()
        W_qkv.grad = W_o.grad = W_up.grad = W_dn.grad = None

    # ── warm-up then timed runs ───────────────────────────────────────────────
    WARMUP, TIMED = 5, 20
    for _ in range(WARMUP):
        step()
    sync()

    t0 = time.perf_counter()
    for _ in range(TIMED):
        step()
    sync()
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed / TIMED * 1000
    tok_per_sec = BT / (elapsed / TIMED)

    # Estimate total training time: total tokens / measured throughput.
    # +15% for attention scores and layer norms not in the benchmark
    # (SSSL sliding-window attention ≈ T/(12D) extra FLOP, 11–17% across profiles).
    # Note: real training uses torch.compile which adds ~20% speedup on top,
    # so this estimate is slightly conservative.
    ATTN_OVERHEAD = 1.15
    total_sec = (total_tokens / tok_per_sec) * ATTN_OVERHEAD

    # ── output (matches detect_hw.py alignment style) ─────────────────────────
    print(f"  Benchmark : {tok_per_sec:>10,.0f} tok/sec"
          f"  |  {ms_per_step:.1f} ms/step  (micro-batch, no torch.compile)")
    print(f"  Est. train: ~{_fmt_duration(total_sec):<12}"
          f"  ({total_tokens/1e9:.1f}B tokens, ~20% faster in practice with compile)")


if __name__ == "__main__":
    main()
