#!/usr/bin/env python3
"""GPU memory budget profiling for selfsuvis worker models.

Measures peak VRAM for each model (CLIP, DINOv3, Florence-2) loaded individually,
then in the combinations used during a full mission indexing run, to determine
whether they can coexist on the worker GPU or must be loaded sequentially.

Usage:
    python scripts/profile_gpu_memory.py

Requires: torch with CUDA, open_clip, transformers (pip install transformers timm einops)
Results are printed to stdout and written to docs/gpu_memory_profile.md.
"""
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

DEVICE = "cuda"
DTYPE = torch.float16   # FP16 — matches USE_FP16=true default

# ── helpers ──────────────────────────────────────────────────────────────────

def _free_vram_mib() -> float:
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    return free / (1024 ** 2)


def _used_vram_mib() -> float:
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(0) / (1024 ** 2)


def _peak_vram_mib() -> float:
    return torch.cuda.max_memory_allocated(0) / (1024 ** 2)


def _reset_peak():
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()


def _baseline_mib() -> float:
    """VRAM used before any model is loaded (CUDA runtime + desktop)."""
    torch.cuda.empty_cache()
    # Warm up CUDA context
    _ = torch.zeros(1, device=DEVICE)
    torch.cuda.synchronize()
    return _used_vram_mib()


def _measure(label: str, load_fn, inference_fn=None):
    """Load a model, optionally run inference, measure peak VRAM delta."""
    gc.collect()
    torch.cuda.empty_cache()
    _reset_peak()
    vram_before = _used_vram_mib()

    t0 = time.perf_counter()
    model = load_fn()
    torch.cuda.synchronize()
    vram_after_load = _used_vram_mib()
    load_sec = time.perf_counter() - t0

    peak_during_inference = vram_after_load
    if inference_fn is not None:
        _reset_peak()
        t1 = time.perf_counter()
        inference_fn(model)
        torch.cuda.synchronize()
        peak_during_inference = max(_peak_vram_mib(), vram_after_load)
        infer_sec = time.perf_counter() - t1
    else:
        infer_sec = 0.0

    load_delta = vram_after_load - vram_before
    infer_peak_delta = peak_during_inference - vram_before

    result = {
        "label": label,
        "vram_load_mib": load_delta,
        "vram_infer_peak_mib": infer_peak_delta,
        "load_sec": load_sec,
        "infer_sec": infer_sec,
    }
    print(f"  {label:<35} load={load_delta:>7.0f} MiB  peak={infer_peak_delta:>7.0f} MiB  "
          f"({load_sec:.1f}s load, {infer_sec:.1f}s infer)")
    return result, model


def _unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── model loaders ─────────────────────────────────────────────────────────────

def _load_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=DEVICE
    )
    model = model.to(DTYPE)
    model.eval()
    return (model, preprocess)


def _infer_clip(m):
    model, preprocess = m
    # Simulate batch of 32 frames (typical for embedding pass)
    import torch
    imgs = torch.randn(32, 3, 224, 224, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        _ = model.encode_image(imgs)


def _load_dinov3():
    """Load DINOv2 via torch.hub (ViT-B/14 — matches dino_model.py usage)."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           pretrained=True, verbose=False)
    model = model.to(DEVICE).to(DTYPE)
    model.eval()
    return model


def _infer_dino(model):
    imgs = torch.randn(16, 3, 224, 224, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        _ = model(imgs)


def _load_florence2():
    # Florence-2 remote code checks for flash_attn at import time.
    # Inject a stub so the import succeeds even without flash_attn installed.
    import sys, types
    from importlib.machinery import ModuleSpec
    if "flash_attn" not in sys.modules:
        stub = types.ModuleType("flash_attn")
        stub.__spec__ = ModuleSpec("flash_attn", None)
        stub.__package__ = "flash_attn"
        stub.flash_attn_func = None
        stub.flash_attn_varlen_func = None
        stub.flash_attn_with_kvcache = None
        iface = types.ModuleType("flash_attn.flash_attn_interface")
        iface.__spec__ = ModuleSpec("flash_attn.flash_attn_interface", None)
        iface.flash_attn_func = None
        iface.flash_attn_varlen_func = None
        sys.modules["flash_attn"] = stub
        sys.modules["flash_attn.flash_attn_interface"] = iface

    from transformers import AutoModelForCausalLM, AutoProcessor
    model_id = "microsoft/Florence-2-large"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    return (model, processor)


def _infer_florence2(m):
    model, processor = m
    import torch
    from PIL import Image
    import numpy as np
    imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(4)]
    inputs = processor(
        text=["<CAPTION>"] * 4,
        images=imgs,
        return_tensors="pt",
        padding=True,
    )
    # Cast all float tensors to match model dtype (FP16)
    inputs = {
        k: v.to(DEVICE).to(DTYPE) if v.dtype in (torch.float32, torch.float64) else v.to(DEVICE)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=20)


# ── combination test ──────────────────────────────────────────────────────────

def _measure_combination(label: str, models: list):
    """Measure VRAM with multiple models loaded simultaneously."""
    torch.cuda.empty_cache()
    used = _used_vram_mib()
    free, total = torch.cuda.mem_get_info(0)
    free_mib = free / (1024 ** 2)
    print(f"  {label:<45} used={used:>7.0f} MiB  free={free_mib:>7.0f} MiB")
    return used, free_mib


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Run on a GPU machine.")
        sys.exit(1)

    total_mib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    device_name = torch.cuda.get_device_name(0)
    print(f"\n=== GPU Memory Profile ===")
    print(f"  Device : {device_name}")
    print(f"  Total  : {total_mib:.0f} MiB  ({total_mib/1024:.1f} GiB)")

    baseline = _baseline_mib()
    free0, _ = torch.cuda.mem_get_info(0)
    print(f"  Baseline (CUDA ctx + desktop): {baseline:.0f} MiB used, "
          f"{free0/(1024**2):.0f} MiB free\n")

    results = []
    print("── Individual model VRAM ─────────────────────────────────────────────")

    # CLIP
    r_clip, clip_model = _measure("CLIP ViT-B-16 (FP16)", _load_clip, _infer_clip)
    results.append(r_clip)
    _unload(clip_model)

    # DINOv3 (ViT-B/14)
    try:
        r_dino, dino_model = _measure("DINOv3 ViT-B/14 (FP16)", _load_dinov3, _infer_dino)
        results.append(r_dino)
        _unload(dino_model)
    except Exception as e:
        print(f"  DINOv3 load failed: {e}")
        r_dino = {"label": "DINOv3", "vram_load_mib": 0, "vram_infer_peak_mib": 0}
        results.append(r_dino)

    # Florence-2-large
    try:
        r_f2, f2_model = _measure("Florence-2-large (FP16)", _load_florence2, _infer_florence2)
        results.append(r_f2)
        _unload(f2_model)
    except Exception as e:
        print(f"  Florence-2 load failed: {e}")
        r_f2 = {"label": "Florence-2", "vram_load_mib": 0, "vram_infer_peak_mib": 0}
        results.append(r_f2)

    # ── combination tests ─────────────────────────────────────────────────────
    print("\n── Simultaneous combination VRAM ────────────────────────────────────")

    # CLIP + DINOv3 (both needed during embedding pass)
    gc.collect(); torch.cuda.empty_cache()
    try:
        clip_obj = _load_clip()
        dino_obj = _load_dinov3()
        torch.cuda.synchronize()
        _measure_combination("CLIP + DINOv3 (embedding pass)", [clip_obj, dino_obj])
        _unload(clip_obj); _unload(dino_obj)
    except Exception as e:
        print(f"  CLIP + DINOv3 failed: {e}")

    # Florence-2 + CLIP (captioning + embedding)
    gc.collect(); torch.cuda.empty_cache()
    try:
        f2_obj = _load_florence2()
        clip_obj = _load_clip()
        torch.cuda.synchronize()
        _measure_combination("Florence-2 + CLIP (caption + embed)", [f2_obj, clip_obj])
        _unload(f2_obj); _unload(clip_obj)
    except Exception as e:
        print(f"  Florence-2 + CLIP failed: {e}")

    # Florence-2 + CLIP + DINOv3 (worst case, all at once)
    gc.collect(); torch.cuda.empty_cache()
    try:
        f2_obj = _load_florence2()
        clip_obj = _load_clip()
        dino_obj = _load_dinov3()
        torch.cuda.synchronize()
        _measure_combination("Florence-2 + CLIP + DINOv3 (all)", [f2_obj, clip_obj, dino_obj])
        _unload(f2_obj); _unload(clip_obj); _unload(dino_obj)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  Florence-2 + CLIP + DINOv3: OOM — {e}")
    except Exception as e:
        print(f"  Florence-2 + CLIP + DINOv3 failed: {e}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n── Summary ({device_name}, {total_mib:.0f} MiB total) ──────────────────")
    for r in results:
        fits = "✓" if r["vram_infer_peak_mib"] + baseline < total_mib * 0.9 else "✗"
        print(f"  {fits} {r['label']:<35} "
              f"load={r['vram_load_mib']:>6.0f} MiB  "
              f"peak={r['vram_infer_peak_mib']:>6.0f} MiB")

    clip_mib = results[0]["vram_infer_peak_mib"]
    dino_mib = results[1]["vram_infer_peak_mib"] if len(results) > 1 else 0
    f2_mib   = results[2]["vram_infer_peak_mib"] if len(results) > 2 else 0
    total_all = clip_mib + dino_mib + f2_mib + baseline
    budget_ok = total_all < total_mib * 0.9
    budget_sym = "✓" if budget_ok else "✗"

    print(f"\n  Projected simultaneous total: {total_all:.0f} MiB")
    print(f"  Available (90% of {total_mib:.0f} MiB): {total_mib*0.9:.0f} MiB")
    print(f"  {budget_sym} All three models {'FIT' if budget_ok else 'DO NOT FIT'} simultaneously")

    if not budget_ok:
        print("\n  RECOMMENDATION: Sequential model loading (load → infer → unload → next)")
        print("  Suggested order for pipeline/indexer.py:")
        print("    1. Load Florence-2 → caption all frames → unload Florence-2")
        print("    2. Load CLIP       → embed all frames  → (keep for queries)")
        print("    3. Load DINOv3     → embed all frames  → (keep for AL scoring)")
        print("    nerfstudio runs in separate container — no conflict")
    else:
        print("\n  RECOMMENDATION: Models can coexist — no sequential loading required.")
        print("  Monitor headroom if nerfstudio shares GPU (docker-compose.override.yml).")

    # ── write report ──────────────────────────────────────────────────────────
    _write_report(device_name, total_mib, baseline, results)
    print("\n  Report written to docs/gpu_memory_profile.md")


def _write_report(device_name, total_mib, baseline_mib, results):
    clip_mib = results[0]["vram_infer_peak_mib"] if len(results) > 0 else 0
    dino_mib = results[1]["vram_infer_peak_mib"] if len(results) > 1 else 0
    f2_mib   = results[2]["vram_infer_peak_mib"] if len(results) > 2 else 0
    total_all = clip_mib + dino_mib + f2_mib + baseline_mib
    fits = total_all < total_mib * 0.9
    verdict = "CAN coexist" if fits else "CANNOT coexist — sequential loading required"

    rows = "\n".join(
        f"| {r['label']:<35} | {r['vram_load_mib']:>9.0f} | {r['vram_infer_peak_mib']:>13.0f} |"
        for r in results
    )

    report = f"""# GPU Memory Budget Profile

**Device:** {device_name}
**Total VRAM:** {total_mib:.0f} MiB ({total_mib/1024:.1f} GiB)
**Profiled:** FP16 (USE_FP16=true, matches production default)

## Individual Model VRAM

| Model                               | Load (MiB) | Infer peak (MiB) |
|-------------------------------------|-----------|----------------|
{rows}
| Baseline (CUDA ctx + desktop)       | {baseline_mib:>9.0f} |               — |

## Simultaneous Budget

| Combination                         | Est. total (MiB) | Fits in {total_mib:.0f} MiB? |
|-------------------------------------|-----------------|---------------|
| CLIP + DINOv3 (embedding pass)      | {clip_mib+dino_mib+baseline_mib:>15.0f} | {'✓ Yes' if clip_mib+dino_mib+baseline_mib < total_mib*0.9 else '✗ No'} |
| Florence-2 + CLIP (caption+embed)   | {f2_mib+clip_mib+baseline_mib:>15.0f} | {'✓ Yes' if f2_mib+clip_mib+baseline_mib < total_mib*0.9 else '✗ No'} |
| Florence-2 + CLIP + DINOv3 (all)    | {total_all:>15.0f} | {'✓ Yes' if fits else '✗ No'} |

**Verdict:** {verdict}

## Recommendation for `pipeline/indexer.py`

{'All three models fit simultaneously — no lifecycle management required. Monitor headroom if nerfstudio shares the GPU via docker-compose.override.yml.' if fits else '''Sequential loading order (least VRAM waste):

```
Pass A (SfM) — CPU only, no GPU conflict

Pass B (sparse keyframes):
  Step 1: Load Florence-2  → caption batch → unload Florence-2
  Step 2: Load CLIP        → embed all frames → keep loaded (needed for queries)
  Step 3: Load DINOv3      → embed all frames → keep loaded (needed for AL scoring)

nerfstudio (separate GPU container, docker-compose.override.yml):
  Runs after indexer completes — no simultaneous conflict on single-GPU machines.
  On multi-GPU machines, nerfstudio can run concurrently on GPU 1.
```

**Dockerfile.worker note:** Do NOT load all three models at startup. Use lazy loading
(`MODEL_NAME` controls which models load; Florence-2 is always needed).'''}

## Notes

- RTX 4060 Ti has {total_mib:.0f} MiB. RTX 4090 (24 GB) and A100 (40/80 GB) have more headroom.
- nerfstudio splatfacto peak VRAM is scene-dependent (~8–16 GB for typical outdoor missions).
  On a 16 GB GPU with all worker models loaded, a simultaneous nerfstudio run WILL OOM.
  The v1 architecture already serialises them: indexer completes before mapper starts.
- FP32 doubles all model sizes. Keep USE_FP16=true (default) on the worker.
"""

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "gpu_memory_profile.md").write_text(report)


if __name__ == "__main__":
    main()
