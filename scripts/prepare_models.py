"""Download and cache all model weights needed by selfsuvis.

Run this once (or in a Docker build step) to pre-populate the local cache so
that the API, worker, and demo can start without network access.

Usage
-----
    # Core models (always needed):
    python scripts/prepare_models.py                # OpenCLIP + DINOv2/v3 (default)
    python scripts/prepare_models.py --clip         # OpenCLIP only
    python scripts/prepare_models.py --dino         # DINOv2/v3 hub archive + weights only

    # Gemma open-weight (downloads weights from HuggingFace — requires HF_TOKEN):
    python scripts/prepare_models.py --gemma        # Step J: google/gemma-3-4b-it (default, multimodal)
    python scripts/prepare_models.py --gemma --gemma-model google/gemma-3-1b-it   # text-only, ~2 GiB
    python scripts/prepare_models.py --gemma --gemma-model google/gemma-3-12b-it  # 12B, ~24 GiB

    # Step-specific optional models:
    python scripts/prepare_models.py --flash-attn   # Install flash-attn (CUDA required)
    python scripts/prepare_models.py --whisper      # Step M: Whisper ASR
    python scripts/prepare_models.py --florence     # Step L: Florence-2 scene captioning
    python scripts/prepare_models.py --ocr          # Step N: OCR (auto-selects by VRAM)
    python scripts/prepare_models.py --depth        # Step O: Depth estimation
    python scripts/prepare_models.py --detection    # Step P: Object detection
    python scripts/prepare_models.py --world-model  # Step Q: World model video embeddings
    python scripts/prepare_models.py --yolo         # Step P2: YOLO11l detection (~48 MB)
    python scripts/prepare_models.py --sam          # Step P2: SAM3/SAM2 segmentation (tries sam3 first)

    # Override auto-selected model for any step:
    python scripts/prepare_models.py --ocr       --ocr-model       microsoft/trocr-base-printed
    python scripts/prepare_models.py --depth     --depth-model     depth-anything/Depth-Anything-V2-Large-hf
    python scripts/prepare_models.py --detection --detection-model IDEA-Research/grounding-dino-base
    python scripts/prepare_models.py --world-model --world-model-id MCG-NJU/videomae-base
    python scripts/prepare_models.py --yolo        --yolo-model yolo11x
    python scripts/prepare_models.py --sam         --sam-model facebook/sam2-hiera-large

    # Download everything (including flash-attn and gated models):
    python scripts/prepare_models.py --all

    # Check what is already cached (no network):
    python scripts/prepare_models.py --verify
    python scripts/prepare_models.py --verify --all    # verify all, not just defaults

    # Force Hugging Face as the download source (useful when GitHub is unreachable):
    python scripts/prepare_models.py --dino --source hf

    # Force torch.hub / GitHub (skip HF fallback):
    python scripts/prepare_models.py --dino --source hub

Gated / access-restricted models
----------------------------------
Some models (e.g. nvidia/Cosmos-1.0-Autoregressive-4B) require accepting a
license on HuggingFace before downloading.  This script detects such errors and
switches to interactive mode: it prints step-by-step instructions and waits for
you to complete authentication before retrying automatically.

If you need to pre-authenticate non-interactively:
    huggingface-cli login           # stores token in ~/.cache/huggingface/token
    export HF_TOKEN=<your_token>    # or set this env var

Environment
-----------
    DINO_MODEL           Comma-separated model names to warm up
                         (default: dinov2_vitb14,dinov3_vitb14)
    OPENCLIP_MODEL       OpenCLIP model name        (default from pipeline.core.config)
    OPENCLIP_PRETRAINED  OpenCLIP pretrained tag     (default from pipeline.core.config)
    GEMMA_MODEL_ID       Gemma model repo ID (default: google/gemma-3-4b-it)
    DEVICE               torch device for loading    (default: cpu)
    HF_TOKEN             HuggingFace token for gated / private model access (required for Gemma)
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy third-party warnings that are irrelevant to warmup.
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="xFormers is available", category=UserWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated",
                        category=FutureWarning)
# timm ResNet50 meta-parameter copy warnings (hundreds of lines, all expected).
warnings.filterwarnings("ignore", message="copying from a non-meta parameter", category=UserWarning)

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env so HF_TOKEN / GEMMA_MODEL_ID etc. are visible to os.getenv() calls below.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

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

# ── Auth error detection & interactive retry ──────────────────────────────────

def _is_auth_error(exc: Exception):
    """Return (is_auth_error: bool, kind: str) where kind is 'gated' or 'unauthorized'."""
    # Try huggingface_hub typed exceptions first (most reliable).
    for module_path in ("huggingface_hub.errors", "huggingface_hub.utils"):
        try:
            m = __import__(module_path, fromlist=["GatedRepoError", "RepositoryNotFoundError"])
            GatedRepoError = getattr(m, "GatedRepoError", None)
            if GatedRepoError and isinstance(exc, GatedRepoError):
                return True, "gated"
            RepositoryNotFoundError = getattr(m, "RepositoryNotFoundError", None)
            if RepositoryNotFoundError and isinstance(exc, RepositoryNotFoundError):
                return True, "unauthorized"
        except (ImportError, AttributeError):
            continue

    # Fall back to string matching for wrapped errors.
    msg = str(exc).lower()
    if "gated" in msg or ("access" in msg and "repo" in msg):
        return True, "gated"
    if "401" in msg or "403" in msg or "unauthorized" in msg or "authentication" in msg:
        return True, "unauthorized"
    return False, ""


def _with_auth_retry(label: str, model_id: str, download_fn) -> None:
    """Run download_fn(); on auth/gated error switch to interactive mode.

    Prints step-by-step HuggingFace authentication instructions and waits for
    the user to complete them, then retries automatically.  Gives up after 3
    failed attempts.  In non-interactive (piped) mode prints instructions and
    raises so the caller can log the error.
    """
    max_retries = 3
    attempts = 0
    while True:
        attempts += 1
        try:
            download_fn()
            return
        except Exception as exc:
            is_auth, kind = _is_auth_error(exc)
            if not is_auth or attempts >= max_retries:
                raise

            hf_url = f"https://huggingface.co/{model_id}"
            bar = "─" * 70
            print(f"\n{bar}", flush=True)
            print(f"  ACCESS REQUIRED — {label}", flush=True)
            print(f"{bar}", flush=True)

            if kind == "gated":
                print(f"""
  This model is gated.  You must accept its license on HuggingFace before
  the weights can be downloaded.

    1. Open the model page in your browser:
         {hf_url}
       Click  "Agree and access repository"

    2. Generate a HuggingFace token (if you don't have one):
         https://huggingface.co/settings/tokens
       Choose "Read" permissions.

    3. Enter your token at the prompt below (it will be set for this session),
       or leave blank if you already ran  huggingface-cli login.

    4. The download will retry automatically.
""", flush=True)
            else:
                print(f"""
  This model requires a HuggingFace account / token.

    1. Create an account (if needed):
         https://huggingface.co/join

    2. Generate a Read token:
         https://huggingface.co/settings/tokens

    3. If the model page shows a license, accept it at:
         {hf_url}

    4. Enter your token at the prompt below (it will be set for this session),
       or leave blank if you already ran  huggingface-cli login.
""", flush=True)

            if not sys.stdin.isatty():
                print(
                    "  Running non-interactively — cannot prompt.\n"
                    "  Set HF_TOKEN env var and re-run this script in an interactive terminal.\n"
                    f"{bar}\n",
                    flush=True,
                )
                raise

            try:
                import getpass
                token_input = getpass.getpass(
                    f"  HF token (leave blank to skip token entry, 's' to skip model): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(flush=True)
                raise exc

            if token_input.lower() == "s":
                log.warning("Skipping %s (user chose to skip)", label)
                raise exc

            if token_input and token_input.lower() != "s":
                os.environ["HF_TOKEN"] = token_input
                # Also propagate to huggingface_hub so it picks up the token immediately.
                try:
                    from huggingface_hub import login as hf_login
                    hf_login(token=token_input, add_to_git_credential=False)
                    log.info("HuggingFace token accepted.")
                except Exception:
                    pass  # login() not critical; env var is the fallback

            log.info("Retrying download: %s …", label)


# ── flash-attn installation ───────────────────────────────────────────────────

def _install_flash_attn() -> None:
    """Install flash-attn using prebuilt PyPI wheel or compile from source.

    Uses ``--no-build-isolation`` so nvcc and torch headers from the active
    environment are used for source builds (avoids version mismatches).
    Prebuilt wheels exist on PyPI for the most common CUDA + Python + torch
    combinations and are used automatically when available.
    """
    log.info("flash-attn — checking installation …")
    try:
        import flash_attn
        log.info("  ✓ flash-attn already installed  (version %s)", flash_attn.__version__)
        return
    except ImportError:
        pass

    import torch
    if not torch.cuda.is_available():
        log.warning(
            "  CUDA not available — skipping flash-attn installation.\n"
            "  flash-attn is a CUDA-only package and cannot run on CPU."
        )
        return

    log.info(
        "  Installing flash-attn (uses prebuilt wheel when available; "
        "otherwise compiles from source — may take several minutes) …"
    )
    import subprocess as _sp
    # Ensure build tools are present (uv-created venvs omit wheel by default).
    _sp.run([sys.executable, "-m", "pip", "install", "wheel", "packaging", "-q"], check=True)
    cmd = [
        sys.executable, "-m", "pip", "install",
        "flash-attn", "--no-build-isolation", "-q",
    ]
    t0 = time.monotonic()
    result = _sp.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "flash-attn installation failed.\n"
            "  Ensure CUDA toolkit is installed (nvcc must be in PATH).\n"
            "  Prebuilt wheels for your CUDA + Python + torch combination:\n"
            "    https://github.com/Dao-AILab/flash-attention/releases\n"
            "  Download the matching .whl and install manually:\n"
            "    pip install <wheel_file.whl>"
        )
    log.info("  ✓ flash-attn installed  (%.1fs)", time.monotonic() - t0)


def _download_openclip(model: str, pretrained: str, device: str) -> None:
    log.info("OpenCLIP — model=%s  pretrained=%s  device=%s", model, pretrained, device)
    if _is_openclip_cached(model, pretrained):
        log.info("  ✓ OpenCLIP already cached — skipping load")
        return
    import open_clip
    t0 = time.monotonic()
    open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
    log.info("  ✓ OpenCLIP ready  (%.1fs)", time.monotonic() - t0)


# Tracks resolved model names already warmed up in this session so we don't
# re-run the forward pass for aliases that map to the same weights.
_warmed: set = set()


def _label(model_name: str, resolved: str) -> str:
    if model_name != resolved:
        return f"{model_name} → {resolved} (alias)"
    return model_name


def _download_whisper(model_id: str) -> None:
    log.info("Whisper ASR — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  ✓ Whisper already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from transformers import pipeline as _hf_pipeline
        _hf_pipeline("automatic-speech-recognition", model=model_id, device="cpu")
        log.info("  ✓ Whisper ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  Whisper download failed: %s", exc)
        raise


def _download_florence(model_id: str = "microsoft/Florence-2-large") -> None:
    log.info("Florence-2 — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  ✓ Florence-2 already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        # trust_remote_code is still required (model repo doesn't include processor_config.json
        # for native loading). transformers>=4.47 correctly handles Florence-2's conditional
        # flash_attn imports without requiring flash_attn to be installed.
        AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto",
        )
        log.info("  ✓ Florence-2 ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  Florence-2 download failed: %s", exc)
        raise


def _download_ocr(model_id: str) -> None:
    """Download (or verify) OCR model weights.

    Dispatches to the correct loader based on model family:
    - TrOCR (microsoft/trocr-*): TrOCRProcessor + VisionEncoderDecoderModel
    - GOT-OCR2 (ucaslcl/GOT-*): AutoTokenizer + AutoModel with trust_remote_code
    - VLM family (Phi-3.5-vision, Qwen2.5-VL, DeepSeek-OCR-2, llava-hf/*):
      AutoProcessor + AutoModelForCausalLM with trust_remote_code
    - Florence-2 (microsoft/Florence-*): AutoProcessor + AutoModelForCausalLM
    """
    log.info("OCR model — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  ✓ OCR model already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        if model_id.startswith("microsoft/trocr-"):
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            TrOCRProcessor.from_pretrained(model_id)
            VisionEncoderDecoderModel.from_pretrained(model_id)
        elif model_id.startswith("ucaslcl/GOT-"):
            from transformers import AutoTokenizer, AutoModel
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            AutoModel.from_pretrained(
                model_id, trust_remote_code=True, use_safetensors=True,
                low_cpu_mem_usage=True,
            )
        else:
            # VLM family (Phi-3.5-vision, Qwen2.5-VL, DeepSeek-OCR-2, llava-hf/*, Florence-2).
            # transformers>=4.47 handles Florence-2's conditional flash_attn imports correctly
            # without requiring flash_attn to be installed.
            from transformers import AutoProcessor, AutoModelForCausalLM
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
        log.info("  ✓ OCR model ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  OCR model download failed: %s", exc)
        raise


def _download_depth(model_id: str) -> None:
    """Download depth-estimation model weights via HF transformers pipeline."""
    log.info("Depth model — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  ✓ Depth model already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from transformers import pipeline as _hf_pipeline
        _hf_pipeline("depth-estimation", model=model_id, device="cpu")
        log.info("  ✓ Depth model ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  Depth model download failed: %s", exc)
        raise


def _download_detection(model_id: str) -> None:
    """Download object-detection model weights via HF transformers pipeline."""
    log.info("Detection model — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  ✓ Detection model already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from transformers import pipeline as _hf_pipeline
        _hf_pipeline("object-detection", model=model_id, device="cpu")
        log.info("  ✓ Detection model ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  Detection model download failed: %s", exc)
        raise


def _download_yolo(model_id: str) -> None:
    """Download YOLO11 weights via ultralytics auto-download.

    ultralytics downloads weights to ``~/.cache/ultralytics/`` on first use.
    Triggering a dummy load here pre-populates that cache so the demo can
    start without network access.
    """
    model_file = model_id if model_id.endswith(".pt") else f"{model_id}.pt"
    log.info("YOLO11 — model=%s", model_file)

    # Check ultralytics cache: ~/.cache/ultralytics/<model_file>
    ult_cache = Path.home() / ".cache" / "ultralytics" / model_file
    if ult_cache.exists():
        log.info("  ✓ YOLO11 already cached at %s — skipping download", ult_cache)
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning("  ultralytics not installed — skipping YOLO download (pip install ultralytics)")
        return

    t0 = time.monotonic()
    try:
        model = YOLO(model_file)  # triggers download if not cached
        # Run a tiny inference to verify weights are valid.
        import numpy as np
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        model(dummy, verbose=False)
        log.info("  ✓ YOLO11 ready  (%.1fs)  file=%s", time.monotonic() - t0, model_file)
    except Exception as exc:
        log.warning("  YOLO11 download/verification failed: %s", exc)
        raise


def _is_yolo_cached(model_id: str) -> bool:
    model_file = model_id if model_id.endswith(".pt") else f"{model_id}.pt"
    return (Path.home() / ".cache" / "ultralytics" / model_file).exists()


def _download_sam(model_id: str) -> None:
    """Download SAM3 (or SAM2 fallback) weights from HuggingFace Hub.

    Tries the requested model_id; if that fails due to SAM3 not being
    available yet, falls back to SAM2.
    """
    log.info("SAM — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  ✓ SAM already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        log.info("  ✓ SAM ready  (%.1fs)  cache=%s", time.monotonic() - t0, local_dir)
    except Exception as exc:
        # SAM3 may not be publicly available yet; fall back to SAM2.
        if "sam3" in model_id.lower():
            fallback = "facebook/sam2-hiera-large"
            log.warning(
                "  SAM3 download failed (%s) — falling back to %s", exc, fallback
            )
            if _is_hf_cached(fallback):
                log.info("  ✓ SAM2 fallback already cached — skipping download")
                return
            try:
                from huggingface_hub import snapshot_download as _sd
                local_dir = _sd(
                    repo_id=fallback,
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
                )
                log.info(
                    "  ✓ SAM2 fallback ready  (%.1fs)  cache=%s",
                    time.monotonic() - t0, local_dir,
                )
                return
            except Exception as exc2:
                log.warning("  SAM2 fallback also failed: %s", exc2)
                raise exc2 from exc
        raise


def _download_world_model(model_id: str) -> None:
    """Download world-model weights.

    Tries AutoFeatureExtractor + AutoModel first (works for VideoMAE, VJEPA2, etc.).
    Falls back to snapshot_download for models that lack preprocessor_config.json
    (e.g. nvidia/Cosmos-1.0-Autoregressive-4B which is a generative autoregressive model).
    """
    log.info("World model — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  ✓ World model already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoFeatureExtractor, AutoModel
        try:
            AutoFeatureExtractor.from_pretrained(model_id)
        except (OSError, EnvironmentError) as feat_exc:
            if "does not appear to have a file named" in str(feat_exc):
                log.info("  No preprocessor_config.json — downloading repo via snapshot_download")
                local_dir = snapshot_download(
                    repo_id=model_id,
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
                )
                log.info("  ✓ World model cached at %s  (%.1fs)", local_dir, time.monotonic() - t0)
                return
            raise
        AutoModel.from_pretrained(model_id, torch_dtype="auto")
        local_dir = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        log.info("  ✓ World model ready  (%.1fs)  cache=%s", time.monotonic() - t0, local_dir)
    except Exception as exc:
        log.warning("  World model download failed: %s", exc)
        raise


def _download_dino(model_name: str, device: str, source: str = "auto") -> None:
    """Download (or verify) DINO weights."""
    import torch
    import torch.hub as _hub
    from models.dino_model import (
        DINO_HUB_REPO, _DINO_MODEL_ALIAS, _DINO_HF_REPO,
        _resolve_dino_hub, hub_load_dino,
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

    if model_name in _warmed:
        log.info("  ✓ DINO already warmed in this session  %s", label)
        return

    if hub_source == "local":
        log.info("  Hub archive cached at %s", repo_or_dir)
        # Check if pretrained weights are already on disk — if so skip the load.
        hub_checkpoints = Path(os.getenv("TORCH_HOME",
                                         str(Path.home() / ".cache" / "torch"))) / "hub" / "checkpoints"
        if any(hub_checkpoints.glob(f"{resolved}*pretrain*.pth")):
            _warmed.add(model_name)
            log.info("  ✓ DINO weights cached  %s  — skipping load", label)
            return
    else:
        log.info("  Downloading hub archive from %s …", DINO_HUB_REPO)
        log.info("  Cache dir: %s", _hub.get_dir())

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

            def _hook(_blk, blk_sz, tot):
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
    from models.dino_model import _load_dino_from_hf

    _GatedRepoError = None
    try:
        from huggingface_hub.errors import GatedRepoError as _GatedRepoError
    except ImportError:
        try:
            from huggingface_hub.utils import GatedRepoError as _GatedRepoError
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
        raise


def _run_dummy(model, device: str) -> None:
    import torch as _torch
    dummy = _torch.zeros(1, 3, 224, 224, device=device)
    with _torch.no_grad():
        model(dummy)


# ── Gemma open-weight model download ─────────────────────────────────────────

def _download_gemma(model_id: str) -> None:
    """Download Gemma weights from HuggingFace and warm up the processor/tokenizer.

    Gemma is a gated model — requires accepting the license on HuggingFace
    and setting HF_TOKEN in .env (or running ``huggingface-cli login``).

    Setup (one-time):
      1. Accept the license at https://huggingface.co/google/gemma-3-4b-it
      2. Add HF_TOKEN=hf_... to your .env file
      3. Run: python scripts/prepare_models.py --gemma

    The function uses :func:`_with_auth_retry` so it prints interactive
    authentication instructions and retries on access errors.
    """
    from pipeline.core.config import mask_secret
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or None
    log.info("Gemma — downloading %s  (token: %s) …", model_id, mask_secret(token or ""))

    def _do_download() -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
        import torch as _torch
        # Try multimodal processor first; fall back to text-only tokenizer.
        log.info("  Downloading processor/tokenizer …")
        try:
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
            log.info("  AutoProcessor loaded (multimodal model)")
        except OSError:
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
            log.info("  AutoTokenizer loaded (text-only model — no vision support)")
        log.info("  Downloading model weights (this may take a while) …")
        AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_torch.bfloat16,
            device_map="cpu",   # download to cache only; no GPU needed for prep
            trust_remote_code=True,
            token=token,
        )
        log.info("  ✓ Gemma %s cached", model_id)

    _with_auth_retry(f"Gemma ({model_id})", model_id, _do_download)


def _is_gemma_cached(model_id: str) -> bool:
    """Return True if Gemma weights are in the local HuggingFace cache."""
    return _is_hf_cached(model_id)


# ── verify ────────────────────────────────────────────────────────────────────

def _is_hf_cached(model_id: str) -> bool:
    """Return True if at least the config for *model_id* is in the local HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        for fname in ("config.json", "model.safetensors", "pytorch_model.bin",
                      "preprocessor_config.json", "tokenizer_config.json"):
            result = try_to_load_from_cache(repo_id=model_id, filename=fname)
            if result is not None:
                return True
    except Exception:
        pass
    return False


def _is_openclip_cached(model: str, pretrained: str) -> bool:
    """Return True if open_clip weights are in the local cache."""
    try:
        # open_clip stores weights under torch hub checkpoints
        cache_dir = Path(os.getenv("TORCH_HOME",
                                   str(Path.home() / ".cache" / "torch"))) / "hub" / "checkpoints"
        tag = f"{model.replace('/', '_')}_{pretrained}".lower()
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                if tag in f.name.lower():
                    return True
    except Exception:
        pass
    return False


def _is_dino_hub_cached(_model_name: str) -> bool:
    """Return True if the DINOv2 torch.hub archive is present."""
    try:
        import torch.hub as _hub
        hub_dir = Path(_hub.get_dir())
        repo_dir = hub_dir / "facebookresearch_dinov2_main"
        return repo_dir.exists()
    except Exception:
        pass
    return False


def _verify_models(model_specs: list) -> tuple:
    """Check cache status for a list of (label, cache_check_fn) tuples.

    Returns (ok_list, missing_list).
    """
    ok, missing = [], []
    for label, check_fn in model_specs:
        cached = False
        try:
            cached = check_fn()
        except Exception:
            pass
        if cached:
            ok.append(label)
        else:
            missing.append(label)
    return ok, missing


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-download all model weights for selfsuvis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clip", action="store_true", help="Download OpenCLIP weights")
    p.add_argument("--dino", action="store_true", help="Download DINOv2/v3 hub weights")
    p.add_argument("--gemma", action="store_true",
                   help="Download Gemma open-weight model (step J; requires HF_TOKEN for gated access)")
    _default_gemma = os.getenv("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
    p.add_argument("--gemma-model", default=_default_gemma, metavar="MODEL_ID",
                   help=(
                       "Gemma model repo ID to cache (requires HF_TOKEN in .env and license accepted). "
                       "Multimodal (vision+text): google/gemma-3-4b-it (~8 GiB, default), "
                       "google/gemma-3-12b-it (~24 GiB). "
                       "Text-only: google/gemma-3-1b-it (~2 GiB, no image encoding). "
                       "Ollama sidecar: gemma4:e4b (set GEMMA_API_URL=http://localhost:11434/v1)."
                   ))
    p.add_argument("--flash-attn", action="store_true",
                   help="Install flash-attn (CUDA required; uses prebuilt wheel or compiles)")
    p.add_argument("--all",  action="store_true",
                   help="Download/verify everything (flash-attn + clip + dino + gemma + florence + whisper + ocr + depth + detection + world-model)")
    p.add_argument("--verify", action="store_true",
                   help="Check cache status for all requested models without downloading")

    p.add_argument("--device", default=os.getenv("DEVICE", "auto"),
                   choices=["cpu", "cuda", "auto"],
                   help="Torch device for weight loading")
    _default_dino = os.getenv("DINO_MODEL", "dinov2_vitb14,dinov3_vitb14").split(",")
    p.add_argument("--dino-model", nargs="+", default=_default_dino, metavar="MODEL",
                   help="DINO model names to warm up")
    p.add_argument("--source", default="auto", choices=["auto", "hub", "hf"],
                   help="DINO weight source: 'auto' = local → GitHub → HF")

    p.add_argument("--whisper", action="store_true",
                   help="Download Whisper ASR model (step M)")
    _default_whisper = os.getenv("ASR_MODEL", "openai/whisper-large-v3-turbo")
    p.add_argument("--whisper-model", default=_default_whisper, metavar="MODEL_ID",
                   help="Whisper model ID to cache")

    p.add_argument("--florence", action="store_true",
                   help="Download Florence-2 captioning model (step L)")
    _default_florence = os.getenv("FLORENCE_MODEL", "microsoft/Florence-2-large")
    p.add_argument("--florence-model", default=_default_florence, metavar="MODEL_ID",
                   help="Florence-2 model ID to cache")

    p.add_argument("--ocr", action="store_true",
                   help="Download OCR model (step N; auto-selects by VRAM)")
    p.add_argument("--ocr-model", default="", metavar="MODEL_ID",
                   help=(
                       "OCR model ID to cache. Empty = auto-select by VRAM. "
                       "Examples: microsoft/trocr-base-printed, ucaslcl/GOT-OCR2_0, "
                       "microsoft/Phi-3.5-vision-instruct"
                   ))

    p.add_argument("--depth", action="store_true",
                   help="Download depth estimation model (step O; auto-selects by VRAM)")
    p.add_argument("--depth-model", default="", metavar="MODEL_ID",
                   help=(
                       "Depth model ID to cache. Empty = auto-select by VRAM. "
                       "Examples: depth-anything/Depth-Anything-V2-Small-hf, "
                       "depth-anything/Depth-Anything-V2-Large-hf"
                   ))

    p.add_argument("--detection", action="store_true",
                   help="Download object detection model (step P; auto-selects by VRAM)")
    p.add_argument("--detection-model", default="", metavar="MODEL_ID",
                   help=(
                       "Detection model ID to cache. Empty = auto-select by VRAM. "
                       "Examples: PekingU/rtdetr_r50vd, IDEA-Research/grounding-dino-base"
                   ))

    p.add_argument("--world-model", action="store_true",
                   help="Download world model for video embeddings (step Q; auto-selects by VRAM)")
    p.add_argument("--world-model-id", default="", metavar="MODEL_ID",
                   help=(
                       "World model ID to cache. Empty = auto-select by VRAM. "
                       "Examples: MCG-NJU/videomae-base, facebook/vjepa2-vitl-fpc64-256"
                   ))

    p.add_argument("--yolo", action="store_true",
                   help="Download YOLO11 detection model (step P2; default model: yolo11l.pt ~48 MB)")
    p.add_argument("--yolo-model", default="yolo11l", metavar="MODEL",
                   help=(
                       "YOLO model filename to cache (without .pt extension). "
                       "Default: yolo11l (~48 MB, 25.3 M params). "
                       "Options: yolo11n (6 MB) | yolo11s (18 MB) | yolo11m (38 MB) "
                       "| yolo11l (48 MB) | yolo11x (109 MB)"
                   ))

    p.add_argument("--sam", action="store_true",
                   help="Download SAM3/SAM2 segmentation model (step P2; tries sam3 then sam2 fallback)")
    p.add_argument("--sam-model", default="facebook/sam3-hiera-large", metavar="MODEL_ID",
                   help=(
                       "SAM model repo ID to cache. "
                       "Default: facebook/sam3-hiera-large (falls back to facebook/sam2-hiera-large). "
                       "Options: facebook/sam3-hiera-large | facebook/sam2-hiera-large | "
                       "facebook/sam2-hiera-base-plus"
                   ))

    return p


def _resolve_hf_model(task: str, override: str) -> str:
    """Return the model ID to use: *override* if set, else auto-select by VRAM."""
    mid = override.strip()
    if mid:
        return mid
    from pipeline.vision.registry import auto_select, detect_resources
    selected = auto_select(task, detect_resources())
    if selected:
        log.info("%s auto-selected model: %s", task, selected)
    return selected or ""


def main() -> None:
    args = _build_parser().parse_args()

    _any_flag = (args.clip or args.dino or args.gemma or args.flash_attn or args.whisper
                 or args.florence or args.ocr or args.depth or args.detection or args.world_model
                 or args.yolo or args.sam)
    # Auto-include Gemma in the default run when HF_TOKEN + GEMMA_MODEL_ID are configured,
    # so `python scripts/prepare_models.py` caches everything needed out of the box.
    _gemma_env_ready = bool(
        (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
        and os.getenv("GEMMA_MODEL_ID")
    )
    do_clip        = args.clip        or args.all or not _any_flag
    do_dino        = args.dino        or args.all or not _any_flag
    do_gemma       = args.gemma       or args.all or (not _any_flag and _gemma_env_ready)
    do_flash_attn  = args.flash_attn  or args.all
    do_whisper     = args.whisper     or args.all
    do_florence    = args.florence    or args.all
    do_ocr         = args.ocr         or args.all
    do_depth       = args.depth       or args.all
    do_detection   = args.detection   or args.all
    do_world_model = args.world_model or args.all
    do_yolo        = args.yolo        or args.all
    do_sam         = args.sam         or args.all

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Auto device → %s", device)

    from pipeline.core.config import settings

    # ── Resolve all HF model IDs up front ────────────────────────────────────
    whisper_id    = args.whisper_model
    florence_id   = args.florence_model
    ocr_id        = _resolve_hf_model("ocr",         args.ocr_model)      if do_ocr         else ""
    depth_id      = _resolve_hf_model("depth",       args.depth_model)    if do_depth       else ""
    detection_id  = _resolve_hf_model("detection",   args.detection_model) if do_detection  else ""
    world_id      = _resolve_hf_model("world_model", args.world_model_id) if do_world_model else ""

    # ── Verify mode ───────────────────────────────────────────────────────────
    if args.verify:
        log.info("Verifying model cache (no downloads) …")
        specs = []
        if do_clip:
            specs.append((
                f"OpenCLIP {settings.OPENCLIP_MODEL}/{settings.OPENCLIP_PRETRAINED}",
                lambda: _is_openclip_cached(settings.OPENCLIP_MODEL, settings.OPENCLIP_PRETRAINED),
            ))
        if do_dino:
            for dm in args.dino_model:
                dm_copy = dm
                specs.append((f"DINOv2/v3 {dm_copy}", lambda d=dm_copy: _is_dino_hub_cached(d)))
        if do_gemma:
            gm = args.gemma_model
            specs.append((f"Gemma {gm}", lambda m=gm: _is_gemma_cached(m)))
        if do_whisper:
            specs.append((f"Whisper {whisper_id}", lambda m=whisper_id: _is_hf_cached(m)))
        if do_florence:
            specs.append((f"Florence-2 {florence_id}", lambda m=florence_id: _is_hf_cached(m)))
        if do_ocr and ocr_id:
            specs.append((f"OCR {ocr_id}", lambda m=ocr_id: _is_hf_cached(m)))
        if do_depth and depth_id:
            specs.append((f"Depth {depth_id}", lambda m=depth_id: _is_hf_cached(m)))
        if do_detection and detection_id:
            specs.append((f"Detection {detection_id}", lambda m=detection_id: _is_hf_cached(m)))
        if do_world_model and world_id:
            specs.append((f"WorldModel {world_id}", lambda m=world_id: _is_hf_cached(m)))
        if do_yolo:
            ym = args.yolo_model
            specs.append((f"YOLO11 {ym}", lambda m=ym: _is_yolo_cached(m)))
        if do_sam:
            sm = args.sam_model
            specs.append((f"SAM {sm}", lambda m=sm: _is_hf_cached(m)))

        ok, missing = _verify_models(specs)
        for label in ok:
            log.info("  ✓ CACHED    %s", label)
        for label in missing:
            log.warning("  ✗ MISSING   %s", label)

        if missing:
            log.error("%d model(s) not cached — run without --verify to download them.", len(missing))
            sys.exit(1)
        log.info("All %d model(s) verified in cache.", len(ok))
        return

    # ── Download mode ─────────────────────────────────────────────────────────
    errors: list = []

    if do_flash_attn:
        try:
            _install_flash_attn()
        except Exception as exc:
            log.error("flash-attn installation failed: %s", exc)
            errors.append(("flash-attn", exc))

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

    if do_gemma:
        try:
            _download_gemma(args.gemma_model)
        except Exception as exc:
            log.error("Gemma download failed: %s", exc)
            errors.append(("Gemma", exc))

    if do_whisper:
        try:
            _with_auth_retry("Whisper", whisper_id, lambda: _download_whisper(whisper_id))
        except Exception as exc:
            log.error("Whisper download failed: %s", exc)
            errors.append(("Whisper", exc))

    if do_florence:
        try:
            _with_auth_retry("Florence-2", florence_id, lambda: _download_florence(florence_id))
        except Exception as exc:
            log.error("Florence-2 download failed: %s", exc)
            errors.append(("Florence-2", exc))

    if do_ocr:
        if not ocr_id:
            log.error("OCR: could not determine a model ID — pass --ocr-model explicitly")
            errors.append(("OCR", ValueError("no model ID")))
        else:
            try:
                _with_auth_retry(f"OCR ({ocr_id})", ocr_id, lambda: _download_ocr(ocr_id))
            except Exception as exc:
                log.error("OCR model [%s] download failed: %s", ocr_id, exc)
                errors.append(("OCR", exc))

    if do_depth:
        if not depth_id:
            log.error("Depth: could not determine a model ID — pass --depth-model explicitly")
            errors.append(("Depth", ValueError("no model ID")))
        else:
            try:
                _with_auth_retry(f"Depth ({depth_id})", depth_id, lambda: _download_depth(depth_id))
            except Exception as exc:
                log.error("Depth model [%s] download failed: %s", depth_id, exc)
                errors.append(("Depth", exc))

    if do_detection:
        if not detection_id:
            log.error("Detection: could not determine a model ID — pass --detection-model explicitly")
            errors.append(("Detection", ValueError("no model ID")))
        else:
            try:
                _with_auth_retry(f"Detection ({detection_id})", detection_id,
                                 lambda: _download_detection(detection_id))
            except Exception as exc:
                log.error("Detection model [%s] download failed: %s", detection_id, exc)
                errors.append(("Detection", exc))

    if do_world_model:
        if not world_id:
            log.error("WorldModel: could not determine a model ID — pass --world-model-id explicitly")
            errors.append(("WorldModel", ValueError("no model ID")))
        else:
            try:
                _with_auth_retry(f"WorldModel ({world_id})", world_id,
                                 lambda: _download_world_model(world_id))
            except Exception as exc:
                log.error("World model [%s] download failed: %s", world_id, exc)
                errors.append(("WorldModel", exc))

    if do_yolo:
        yolo_id = args.yolo_model
        try:
            _download_yolo(yolo_id)
        except Exception as exc:
            log.error("YOLO11 [%s] download failed: %s", yolo_id, exc)
            errors.append(("YOLO11", exc))

    if do_sam:
        sam_id = args.sam_model
        try:
            _with_auth_retry(f"SAM ({sam_id})", sam_id, lambda: _download_sam(sam_id))
        except Exception as exc:
            log.error("SAM [%s] download failed: %s", sam_id, exc)
            errors.append(("SAM", exc))

    if errors:
        log.error("%d download(s) failed — see above.", len(errors))
        sys.exit(1)

    log.info("All models ready.")


if __name__ == "__main__":
    main()
