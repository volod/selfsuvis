"""Download and cache all model weights needed by selfsuvis.

Run this once (or in a Docker build step) to pre-populate the local cache so
that the API, worker, and demo can start without network access.

Usage
-----
    python scripts/prepare_models.py            # OpenCLIP + DINOv2 + DINOv3 (default)
    python scripts/prepare_models.py --clip     # OpenCLIP only
    python scripts/prepare_models.py --dino     # DINOv2/v3 hub archive + weights only
    python scripts/prepare_models.py --all      # everything (same as default)

    # Force Hugging Face as the download source (useful when GitHub is unreachable):
    python scripts/prepare_models.py --dino --source hf

    # Force torch.hub / GitHub (skip HF fallback):
    python scripts/prepare_models.py --dino --source hub

If Hugging Face download fails with a license / access error
------------------------------------------------------------
Some models hosted on Hugging Face require you to accept a license agreement
before downloading.  If you see "GatedRepoError" or a 403/401 HTTP error:

    1. Open the model page in your browser and click "Agree and access repository":
           https://huggingface.co/facebook/dinov2-base    (ViT-B/14, ~330 MB)
           https://huggingface.co/facebook/dinov2-large   (ViT-L/14, ~1.1 GB)
           https://huggingface.co/facebook/dinov2-giant   (ViT-g/14, ~4.4 GB)

    2. Log in with your Hugging Face token:
           huggingface-cli login
       (Create a token at https://huggingface.co/settings/tokens if needed.)

    3. Re-run this script:
           python scripts/prepare_models.py --dino --source hf

    Note: the default torch.hub source (GitHub + dl.fbaipublicfiles.com) does NOT
    require login or license acceptance and is tried first automatically.

Environment
-----------
    DINO_MODEL           Comma-separated model names to warm up
                         (default: dinov2_vitb14,dinov3_vitb14)
    OPENCLIP_MODEL       OpenCLIP model name        (default from pipeline.config)
    OPENCLIP_PRETRAINED  OpenCLIP pretrained tag     (default from pipeline.config)
    DEVICE               torch device for loading    (default: cpu)
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy third-party UserWarnings that are irrelevant to warmup.
# xFormers is an optional acceleration library; its absence doesn't affect
# correctness — the model falls back to a standard MLP/attention implementation.
warnings.filterwarnings("ignore", message="xFormers is not available")

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("DEVICE", "auto")
os.environ.setdefault("ALLOWED_INDEX_PATHS", "")
os.environ.setdefault("API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_models")


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _download_openclip(model: str, pretrained: str, device: str) -> None:
    log.info("OpenCLIP — model=%s  pretrained=%s  device=%s", model, pretrained, device)
    import open_clip
    t0 = time.monotonic()
    open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
    log.info("  ✓ OpenCLIP ready  (%.1fs)", time.monotonic() - t0)


# Tracks resolved model names already warmed up in this session so we don't
# re-run the forward pass for aliases that map to the same weights.
_warmed: set = set()


def _label(model_name: str, resolved: str) -> str:
    """Human-readable label that shows alias relationship when relevant."""
    if model_name != resolved:
        return f"{model_name} → {resolved} (alias)"
    return model_name


def _download_dino(model_name: str, device: str, source: str = "auto") -> None:
    """Download (or verify) DINO weights.

    *source* controls where to fetch from:
    - ``"auto"``  — local cache → GitHub → Hugging Face (same as runtime)
    - ``"hub"``   — force GitHub / torch.hub (skip HF fallback)
    - ``"hf"``    — force Hugging Face (skip torch.hub entirely)
    """
    import torch
    import torch.hub as _hub
    from models.dino_model import (
        DINO_HUB_REPO, _DINO_MODEL_ALIAS, _DINO_HF_REPO,
        _resolve_dino_hub, _load_dino_from_hf, hub_load_dino,
    )

    actual_name = _DINO_MODEL_ALIAS.get(model_name, model_name)
    label = _label(model_name, actual_name)

    if source == "hf":
        hf_repo = _DINO_HF_REPO.get(actual_name, "?")
        log.info("DINO (HF) — %s  hf_repo=%s", label, hf_repo)
        t0 = time.monotonic()
        model = _hf_load_with_license_check(model_name, hf_repo)
        model.to(device)
        _run_dummy(model, device)
        _warmed.add(model_name)
        log.info("  ✓ DINO cached from HF  %s  (%.1fs)", label, time.monotonic() - t0)
        return

    hub_source, repo_or_dir, resolved = _resolve_dino_hub(model_name)
    log.info("DINO — %s  source=%s", label, hub_source)

    # Skip only if this exact requested name was already warmed (not by alias).
    if model_name in _warmed:
        log.info("  ✓ DINO already warmed in this session  %s", label)
        return

    if hub_source == "local":
        log.info("  Hub archive cached at %s", repo_or_dir)
    else:
        log.info("  Downloading hub archive from %s …", DINO_HUB_REPO)
        log.info("  Cache dir: %s", _hub.get_dir())

    # Patch downloader to show progress bars.
    _orig = _hub.download_url_to_file

    def _prog(url: str, dst: str, *a, **kw):
        log.info("  ↓ %s", url)
        try:
            from tqdm import tqdm as _tqdm
            import urllib.request as _req
            with _req.urlopen(url) as r:
                total = int(r.headers.get("Content-Length", 0))
            bar = _tqdm(total=total, unit="B", unit_scale=True,
                        desc=f"    {Path(dst).name}", leave=True)

            def _hook(blk, blk_sz, tot):
                if tot > 0:
                    bar.total = tot
                bar.update(blk_sz)

            _req.urlretrieve(url, dst, reporthook=_hook)
            bar.close()
            log.info("  ✓ saved %s", dst)
        except Exception:
            _orig(url, dst, *a, **kw)

    _hub.download_url_to_file = _prog
    try:
        t0 = time.monotonic()
        if source == "hub":
            model = torch.hub.load(repo_or_dir, resolved,
                                   pretrained=True, source=hub_source)
        else:
            model = hub_load_dino(model_name, pretrained=True)
        model.to(device)
        _run_dummy(model, device)
        _warmed.add(model_name)
        elapsed = time.monotonic() - t0
        log.info("  ✓ DINO cached  %s  device=%s  (%.1fs)", label, device, elapsed)
    finally:
        _hub.download_url_to_file = _orig


def _hf_load_with_license_check(model_name: str, hf_repo: str):
    """Load from HF and give clear guidance if the repo requires license acceptance."""
    from models.dino_model import _load_dino_from_hf

    # Import gated-repo exception — location changed between huggingface_hub versions.
    _GatedRepoError = None
    try:
        from huggingface_hub.errors import GatedRepoError as _GatedRepoError  # ≥0.24
    except ImportError:
        try:
            from huggingface_hub.utils import GatedRepoError as _GatedRepoError  # <0.24
        except ImportError:
            pass

    try:
        return _load_dino_from_hf(model_name)
    except Exception as exc:
        if _GatedRepoError and isinstance(exc, _GatedRepoError):
            raise RuntimeError(
                f"\nModel '{hf_repo}' requires license acceptance on Hugging Face.\n"
                f"  1. Open  https://huggingface.co/{hf_repo}\n"
                f"  2. Click 'Agree and access repository'\n"
                f"  3. Log in via CLI:  huggingface-cli login\n"
                f"  4. Re-run:  python scripts/prepare_models.py --dino --source hf"
            ) from exc
        # Re-raise anything else (network error, missing mapping, etc.)
        raise


def _run_dummy(model: "torch.nn.Module", device: str) -> None:
    """Run one dummy forward pass to force lazy weight materialisation."""
    import torch as _torch
    dummy = _torch.zeros(1, 3, 224, 224, device=device)
    with _torch.no_grad():
        model(dummy)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-download all model weights for selfsuvis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clip", action="store_true", help="Download OpenCLIP weights only")
    p.add_argument("--dino", action="store_true", help="Download DINOv2 hub weights only")
    p.add_argument("--all",  action="store_true", help="Download everything (default if no flag given)")
    p.add_argument("--device", default=os.getenv("DEVICE", "auto"),
                   choices=["cpu", "cuda", "auto"],
                   help="Torch device for weight loading (default: auto — uses CUDA when available)")
    _default_dino = os.getenv("DINO_MODEL", "dinov2_vitb14,dinov3_vitb14").split(",")
    p.add_argument("--dino-model", nargs="+", default=_default_dino,
                   metavar="MODEL",
                   help="One or more DINO model names to warm up (default: dinov2_vitb14 dinov3_vitb14)")
    p.add_argument("--source", default="auto", choices=["auto", "hub", "hf"],
                   help=(
                       "DINO weight source: "
                       "'auto' = local cache → GitHub → HF (default); "
                       "'hub' = GitHub/torch.hub only; "
                       "'hf' = Hugging Face only"
                   ))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    do_clip = args.clip or args.all or not (args.clip or args.dino)
    do_dino = args.dino or args.all or not (args.clip or args.dino)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Auto device → %s", device)

    from pipeline.config import settings

    errors: list = []

    if do_clip:
        try:
            _download_openclip(settings.OPENCLIP_MODEL, settings.OPENCLIP_PRETRAINED, device)
        except Exception as exc:
            log.error("OpenCLIP download failed: %s", exc)
            errors.append(("OpenCLIP", exc))

    if do_dino:
        for dino_model in args.dino_model:
            try:
                _download_dino(dino_model, device, source=args.source)
            except Exception as exc:
                log.error("DINO [%s] download failed: %s", dino_model, exc)
                import torch as _t
                hub_dir = _t.hub.get_dir()
                log.error(
                    "  Recovery options:\n"
                    "  1. Hugging Face:  python scripts/prepare_models.py --dino --source hf\n"
                    "  2. Manual clone:  git clone https://github.com/facebookresearch/dinov2 "
                    "%s/facebookresearch_dinov2_main",
                    hub_dir,
                )
                errors.append((f"DINO:{dino_model}", exc))

    if errors:
        log.error("%d download(s) failed — see above.", len(errors))
        sys.exit(1)

    log.info("All models ready.")


if __name__ == "__main__":
    main()
