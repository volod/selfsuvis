"""Download and cache all model weights needed by selfsuvis.

Run this once (or in a Docker build step) to pre-populate the local cache so
that the API, worker, and local full-analysis pipeline can start without network access.

Usage
-----
    # Core models (always needed):
    python scripts/prepare_models.py                # OpenCLIP + DINOv2/v3 (default)
    python scripts/prepare_models.py --clip         # OpenCLIP only
    python scripts/prepare_models.py --dino         # DINOv2/v3 hub archive + weights only

    # Gemma open-weight (downloads weights from HuggingFace — requires HF_TOKEN):
    python scripts/prepare_models.py --gemma        # Step 03: google/gemma-3-4b-it (default, multimodal)
    python scripts/prepare_models.py --gemma --gemma-model google/gemma-3-1b-it   # text-only, ~2 GiB
    python scripts/prepare_models.py --gemma --gemma-model google/gemma-3-12b-it  # 12B, ~24 GiB

    # Step-specific optional models:
    python scripts/prepare_models.py --flash-attn   # Install flash-attn (CUDA required)
    python scripts/prepare_models.py --whisper      # Step 05: Whisper ASR
    python scripts/prepare_models.py --florence     # Step 04: Florence-2 scene captioning
    python scripts/prepare_models.py --ocr          # Step 06: OCR (auto-selects by VRAM)
    python scripts/prepare_models.py --depth        # Step 07: Depth estimation
    python scripts/prepare_models.py --detection    # Step 08: Object detection
    python scripts/prepare_models.py --world-model  # Step 11: World model video embeddings
    python scripts/prepare_models.py --unidrive     # Step 13: UniDriveVLA expert model assets
    python scripts/prepare_models.py --yolo         # Step 09: YOLO11l detection (~48 MB)
    python scripts/prepare_models.py --sam          # Step 09: SAM3/SAM2 segmentation (tries sam3 first)

    # Override auto-selected model for any step:
    python scripts/prepare_models.py --ocr       --ocr-model       microsoft/trocr-base-printed
    python scripts/prepare_models.py --depth     --depth-model     depth-anything/Depth-Anything-V2-Large-hf
    python scripts/prepare_models.py --detection --detection-model IDEA-Research/grounding-dino-base
    python scripts/prepare_models.py --world-model --world-model-id MCG-NJU/videomae-base
    python scripts/prepare_models.py --unidrive --unidrive-model owl10/UniDriveVLA_Nusc_Base_Stage3
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
    UNIDRIVE_MODEL       UniDriveVLA model repo ID (default: owl10/UniDriveVLA_Nusc_Base_Stage3)
    DEVICE               torch device for loading    (default: cpu)
    HF_TOKEN             HuggingFace token for gated / private model access (required for Gemma)
"""

import argparse
import importlib.util
import contextlib
import io
import logging
import os
import shutil
import subprocess
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
# Newer PyTorch prefixes the message with the layer name ("for X.Y: copying…"), so
# the pattern needs .* to match at any position via re.match.
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter", category=UserWarning)
# HF transformers deprecation / slow-processor notices.
# These are emitted as FutureWarning (not UserWarning) in transformers ≥ 4.48,
# so omit the category restriction so the filter matches both.
warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*will be the default.*")
warnings.filterwarnings("ignore", message=".*VideoMAEFeatureExtractor is deprecated.*")
# Force-clear any per-module warning registries that may have cached "show once" state
# for the torch meta-parameter warnings before our filters were installed.
# Guard with isinstance(reg, dict): torch._ops._OpNamespace exposes a
# __warningregistry__ proxy that is not a dict and raises AttributeError on .clear().
for _mod_name in list(sys.modules):
    _mod = sys.modules.get(_mod_name)
    reg = getattr(_mod, "__warningregistry__", None)
    if isinstance(reg, dict):
        reg.clear()

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


@contextlib.contextmanager
def _quiet_hf():
    """Suppress repetitive HF transformers + ultralytics noise during model warmup.

    Redirects stdout/stderr to a sink and sets transformers verbosity to ERROR
    for the duration of the block.  Our own log.info() calls are unaffected
    because they go through the root logging handler, not stdout/stderr directly.
    """
    orig_tf_verbosity = None
    try:
        import transformers
        orig_tf_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    sink = io.StringIO()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter")
            warnings.filterwarnings("ignore", message=".*Using a slow image processor")
            warnings.filterwarnings("ignore", message=".*use_fast.*will be the default")
            warnings.filterwarnings("ignore", message=".*VideoMAEFeatureExtractor is deprecated")
            warnings.filterwarnings("ignore", category=FutureWarning)
            yield
    finally:
        if orig_tf_verbosity is not None:
            try:
                import transformers
                transformers.logging.set_verbosity(orig_tf_verbosity)
            except Exception:
                pass


_UNIDRIVE_DEFAULT_MODEL = "owl10/UniDriveVLA_Nusc_Base_Stage3"
_UNIDRIVE_COLLECTION_URL = "https://huggingface.co/collections/owl10/unidrivevla"
_UNIDRIVE_OLLAMA_FALLBACK_MODEL = "qwen2.5vl:7b"


# ── helpers ───────────────────────────────────────────────────────────────────

def _has_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def _has_vllm_installed() -> bool:
    return importlib.util.find_spec("vllm") is not None


def _is_ollama_model_name(model_id: str) -> bool:
    mid = (model_id or "").strip()
    return bool(mid) and "/" not in mid


def _resolve_unidrive_backend(requested_backend: str, model_id: str) -> str:
    """Choose the backend used to prepare UniDrive assets.

    UniDriveVLA is published on HuggingFace only — it is not available on Ollama.
    vllm is required to serve HF UniDrive repos.  Ollama is only valid for
    Ollama-native model tags (no slash in name).
    """
    have_ollama = _has_ollama_installed()
    have_vllm = _has_vllm_installed()

    backend = (requested_backend or "auto").strip().lower()
    if backend not in {"", "auto", "ollama", "vllm"}:
        raise ValueError(f"Unsupported UniDrive backend: {requested_backend}")
    if backend == "ollama":
        if not have_ollama:
            raise RuntimeError("UniDrive backend 'ollama' requested, but 'ollama' is not installed on this machine.")
        return "ollama"
    if backend == "vllm":
        if not have_vllm:
            raise RuntimeError(
                "UniDrive requires vllm (UniDriveVLA is not available on Ollama). "
                "Install vllm: pip install vllm — or disable UniDrive by setting "
                "UNIDRIVE_ENABLED=false in .env."
            )
        return "vllm"

    # Auto mode: Ollama-tagged models (no slash) can run on Ollama.
    # HuggingFace UniDriveVLA repos require vLLM — they are not published on Ollama.
    if _is_ollama_model_name(model_id):
        if have_ollama:
            return "ollama"
        if have_vllm:
            return "vllm"
    if have_vllm:
        return "vllm"
    raise RuntimeError(
        "UniDriveVLA is not available on Ollama. "
        "Install vllm (pip install vllm) to use the HuggingFace model, "
        "or set UNIDRIVE_ENABLED=false in .env to skip this step."
    )


def _resolve_unidrive_prepare_model(model_id: str, backend: str) -> str:
    """Resolve the model artifact to warm up for the chosen UniDrive backend."""
    requested = (model_id or "").strip()
    if backend == "ollama":
        if requested and _is_ollama_model_name(requested):
            return requested
        if requested and requested.startswith("owl10/UniDriveVLA"):
            log.warning(
                "UniDriveVLA repo '%s' is published on Hugging Face, not in the Ollama library. "
                "Using Ollama fallback model '%s' for UniDrive-style sidecar serving.",
                requested,
                _UNIDRIVE_OLLAMA_FALLBACK_MODEL,
            )
        return _UNIDRIVE_OLLAMA_FALLBACK_MODEL
    if requested:
        return requested
    return _UNIDRIVE_DEFAULT_MODEL


def _is_ollama_model_cached(model: str) -> bool:
    if not _has_ollama_installed():
        return False
    result = subprocess.run(
        ["ollama", "show", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _download_ollama_model(model: str) -> None:
    if not _has_ollama_installed():
        raise RuntimeError("Cannot pull Ollama model because 'ollama' is not installed.")
    log.info("Ollama — model=%s", model)
    if _is_ollama_model_cached(model):
        log.info("  ✓ Ollama model already present — skipping pull")
        return
    t0 = time.monotonic()
    result = subprocess.run(["ollama", "pull", model], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama pull {model!r} failed. Ensure the Ollama daemon is running and the model tag exists."
        )
    log.info("  ✓ Ollama model ready  (%.1fs)", time.monotonic() - t0)

# ── Auth error detection & interactive retry ──────────────────────────────────

def _is_auth_error(exc: Exception):
    """Return (is_auth_error: bool, kind: str) where kind includes repo-not-found."""
    # Try huggingface_hub typed exceptions first (most reliable).
    for module_path in ("huggingface_hub.errors", "huggingface_hub.utils"):
        try:
            m = __import__(module_path, fromlist=["GatedRepoError", "RepositoryNotFoundError"])
            GatedRepoError = getattr(m, "GatedRepoError", None)
            if GatedRepoError and isinstance(exc, GatedRepoError):
                return True, "gated"
            RepositoryNotFoundError = getattr(m, "RepositoryNotFoundError", None)
            if RepositoryNotFoundError and isinstance(exc, RepositoryNotFoundError):
                return False, "repo_not_found"
        except (ImportError, AttributeError):
            continue

    # Fall back to string matching for wrapped errors.
    msg = str(exc).lower()
    if "gated" in msg or ("access" in msg and "repo" in msg):
        return True, "gated"
    if "repository not found" in msg or "404" in msg:
        return False, "repo_not_found"
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
            if kind == "repo_not_found":
                raise RuntimeError(
                    f"Hugging Face repo '{model_id}' was not found. "
                    f"UniDriveVLA weights are currently published under the owl10 collection: "
                    f"{_UNIDRIVE_COLLECTION_URL}"
                ) from exc
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
        with _quiet_hf():
            _hf_pipeline("object-detection", model=model_id, device="cpu")
        log.info("  ✓ Detection model ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  Detection model download failed: %s", exc)
        raise


def _download_yolo(model_id: str) -> None:
    """Download YOLO11 weights via ultralytics auto-download.

    Weights are stored in ``~/.cache/ultralytics/`` — the full path is passed
    to the YOLO constructor so ultralytics downloads there instead of cwd.
    """
    model_file = model_id if model_id.endswith(".pt") else f"{model_id}.pt"
    log.info("YOLO11 — model=%s", model_file)

    # Canonical cache location: ~/.cache/ultralytics/<model_file>
    ult_cache_dir = Path.home() / ".cache" / "ultralytics"
    ult_cache = ult_cache_dir / model_file
    if ult_cache.exists():
        log.info("  ✓ YOLO11 already cached at %s — skipping download", ult_cache)
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning("  ultralytics not installed — skipping YOLO download (pip install ultralytics)")
        return

    ult_cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        import numpy as np
        # Suppress ultralytics' repeated tqdm download lines and settings banner.
        # YOLO_VERBOSE=False turns off their internal VERBOSE flag; redirecting
        # stdout/stderr catches the tqdm progress bar that goes to stdout in non-TTY mode.
        os.environ["YOLO_VERBOSE"] = "False"
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            model = YOLO(str(ult_cache))
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            model(dummy, verbose=False)
        log.info("  ✓ YOLO11 ready  (%.1fs)  file=%s", time.monotonic() - t0, ult_cache)
    except Exception as exc:
        log.warning("  YOLO11 download/verification failed: %s", exc)
        raise


def _is_yolo_cached(model_id: str) -> bool:
    model_file = model_id if model_id.endswith(".pt") else f"{model_id}.pt"
    return (Path.home() / ".cache" / "ultralytics" / model_file).exists()


def _sam3_accessible() -> bool:
    """Return True only if the SAM3 gated repo files are accessible with the current token.

    model_info() is not sufficient — the model card metadata is public even when
    files are gated.  We probe by attempting to fetch a tiny sentinel file which
    hits the same auth gate as the full snapshot.
    """
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="facebook/sam3",
            filename="model_index.json",
            local_files_only=False,
        )
        return True
    except Exception:
        return False


def _sam3_dialog() -> str:
    """Interactive prompt shown when SAM3 access is not granted.

    Prints instructions, then asks the user what to do next.
    Returns one of: 'retry' | 'sam2' | 'skip'.

    When stdin is not a TTY (CI, piped output) the function returns 'sam2'
    immediately so the setup script never blocks.
    """
    import sys

    # Non-interactive path — auto-continue without blocking.
    if not sys.stdin.isatty():
        return "sam2"

    print()
    print("  ┌─ SAM3 — gated model ───────────────────────────────────────────┐")
    print("  │  facebook/sam3 requires HuggingFace access approval.           │")
    print("  │                                                                 │")
    print("  │  To unlock SAM3:                                                │")
    print("  │    1. Visit  https://huggingface.co/facebook/sam3              │")
    print("  │    2. Click 'Access repository' and accept the licence          │")
    print("  │    3. Make sure HF_TOKEN is set in .env (Read scope)            │")
    print("  │    4. Choose [r] Retry below                                    │")
    print("  │                                                                 │")
    print("  │  SAM2 (facebook/sam2-hiera-large) is a fully open fallback.    │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()
    print("  [s]  Use SAM2 fallback  (default)")
    print("  [r]  Retry              (after granting access in another tab)")
    print("  [x]  Skip SAM entirely")
    print()

    while True:
        try:
            raw = input("  Choice [s/r/x]: ")
            # Encode to ASCII, dropping non-ASCII look-alikes (e.g. Cyrillic і vs Latin i)
            raw = raw.encode("ascii", errors="ignore").decode().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "sam2"
        choice = raw or "s"
        if choice in ("s", "sam2"):
            return "sam2"
        if choice in ("r", "retry"):
            return "retry"
        if choice in ("x", "skip"):
            return "skip"
        print("  Please enter s, r, or x.")


def _download_sam(model_id: str) -> None:
    """Download SAM3 (or SAM2 fallback) weights from HuggingFace Hub.

    SAM3 is a gated model.  When access is not granted the function shows an
    interactive dialog (TTY) or auto-continues with SAM2 (non-interactive).
    """
    SAM2_FALLBACK = "facebook/sam2-hiera-large"
    is_sam3 = "sam3" in model_id.lower()

    # For SAM3: probe file access before triggering an expensive snapshot
    # download that would fail with a confusing 403 partway through.
    if is_sam3 and not _is_hf_cached(model_id):
        while not _sam3_accessible():
            action = _sam3_dialog()
            if action == "retry":
                log.info("  Re-checking SAM3 access …")
                continue          # loop back to _sam3_accessible()
            elif action == "skip":
                log.info("SAM — skipped by user choice.")
                return
            else:                 # 'sam2'
                log.info("SAM — using %s (SAM3 access not granted)", SAM2_FALLBACK)
                model_id = SAM2_FALLBACK
                is_sam3 = False
                break

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
        if is_sam3:
            log.info("  SAM3 download failed — falling back to %s", SAM2_FALLBACK)
            if _is_hf_cached(SAM2_FALLBACK):
                log.info("  ✓ SAM2 fallback already cached — skipping download")
                return
            try:
                from huggingface_hub import snapshot_download as _sd
                local_dir = _sd(
                    repo_id=SAM2_FALLBACK,
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


def _download_unidrive(model_id: str) -> None:
    """Download UniDriveVLA model assets for local bridges / sidecars."""
    log.info("UniDriveVLA — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  ✓ UniDriveVLA already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype="auto", low_cpu_mem_usage=True,
            )
        except Exception as exc:
            log.info(
                "  Transformers warmup skipped for %s (%s); repository cache is still ready",
                model_id, exc,
            )
        log.info("  ✓ UniDriveVLA ready  (%.1fs)  cache=%s", time.monotonic() - t0, local_dir)
    except Exception as exc:
        log.warning("  UniDriveVLA download failed: %s", exc)
        raise


def _download_dino(model_name: str, device: str, source: str = "auto") -> None:
    """Download (or verify) DINO weights."""
    import torch
    import torch.hub as _hub
    from selfsuvis.models.dino_model import (
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
    from selfsuvis.models.dino_model import _load_dino_from_hf

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
    from selfsuvis.pipeline.core.config import mask_secret
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
    """Return True if open_clip weights are in the local cache.

    open_clip downloads weights to $CLIP_CACHE (~/.cache/clip by default),
    naming each file after the URL basename (e.g. ViT-B-16.pt).
    HuggingFace-hosted pretrained models land in the HF hub cache instead.
    """
    try:
        # Primary: $CLIP_CACHE / ~/.cache/clip — URL basename (e.g. ViT-B-16.pt)
        clip_cache = Path(os.getenv("CLIP_CACHE", str(Path.home() / ".cache" / "clip")))
        if clip_cache.exists():
            try:
                import open_clip as _oc
                cfg = _oc.get_pretrained_cfg(model, pretrained)
                url = (cfg or {}).get("url", "")
                if url:
                    fname = Path(url).name  # e.g. "ViT-B-16.pt"
                    if (clip_cache / fname).exists():
                        return True
            except Exception:
                pass
            # Fallback: any .pt file whose name starts with the model name
            model_stem = model.replace("/", "-")
            for f in clip_cache.iterdir():
                if f.name.startswith(model_stem) or f.stem == model_stem:
                    return True

        # Secondary: HuggingFace hub cache (HF-hosted pretrained)
        hf_hub_id = None
        try:
            import open_clip as _oc
            cfg = _oc.get_pretrained_cfg(model, pretrained)
            hf_hub_id = (cfg or {}).get("hf_hub", "")
        except Exception:
            pass
        if hf_hub_id:
            return _is_hf_cached(hf_hub_id)
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
                   help="Download Gemma open-weight model (step 03; requires HF_TOKEN for gated access)")
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
                   help="Download/verify everything (default when no other flag is given)")
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
                   help="Download Whisper ASR model (step 05)")
    _default_whisper = os.getenv("ASR_MODEL", "openai/whisper-large-v3-turbo")
    p.add_argument("--whisper-model", default=_default_whisper, metavar="MODEL_ID",
                   help="Whisper model ID to cache")

    p.add_argument("--florence", action="store_true",
                   help="Download Florence-2 captioning model (step 04)")
    _default_florence = os.getenv("FLORENCE_MODEL", "microsoft/Florence-2-large")
    p.add_argument("--florence-model", default=_default_florence, metavar="MODEL_ID",
                   help="Florence-2 model ID to cache")

    p.add_argument("--ocr", action="store_true",
                   help="Download OCR model (step 06; auto-selects by VRAM)")
    p.add_argument("--ocr-model", default="", metavar="MODEL_ID",
                   help=(
                       "OCR model ID to cache. Empty = auto-select by VRAM. "
                       "Examples: microsoft/trocr-base-printed, ucaslcl/GOT-OCR2_0, "
                       "microsoft/Phi-3.5-vision-instruct"
                   ))

    p.add_argument("--depth", action="store_true",
                   help="Download depth estimation model (step 07; auto-selects by VRAM)")
    p.add_argument("--depth-model", default="", metavar="MODEL_ID",
                   help=(
                       "Depth model ID to cache. Empty = auto-select by VRAM. "
                       "Examples: depth-anything/Depth-Anything-V2-Small-hf, "
                       "depth-anything/Depth-Anything-V2-Large-hf"
                   ))

    p.add_argument("--detection", action="store_true",
                   help="Download object detection model (step 08; auto-selects by VRAM)")
    p.add_argument("--detection-model", default="", metavar="MODEL_ID",
                   help=(
                       "Detection model ID to cache. Empty = auto-select by VRAM. "
                       "Examples: PekingU/rtdetr_r50vd, IDEA-Research/grounding-dino-base"
                   ))

    p.add_argument("--world-model", action="store_true",
                   help="Download world model for video embeddings (step 11; auto-selects by VRAM)")
    p.add_argument("--world-model-id", default="", metavar="MODEL_ID",
                   help=(
                       "World model ID to cache. Empty = auto-select by VRAM. "
                       "Examples: MCG-NJU/videomae-base, facebook/vjepa2-vitl-fpc64-256"
                   ))
    _default_unidrive = os.getenv("UNIDRIVE_MODEL", _UNIDRIVE_DEFAULT_MODEL)
    p.add_argument("--unidrive", action="store_true",
                   help="Download UniDriveVLA expert model assets (step 13)")
    p.add_argument("--unidrive-model", default=_default_unidrive, metavar="MODEL_ID",
                   help=(
                       "UniDriveVLA model repo ID to cache for external bridge / sidecar use. "
                       f"Default: {_UNIDRIVE_DEFAULT_MODEL}"
                   ))
    p.add_argument("--unidrive-backend", default=os.getenv("UNIDRIVE_BACKEND", "auto"),
                   choices=["auto", "ollama", "vllm"],
                   help=(
                       "Backend used for UniDrive prep. "
                       "'ollama' pulls an Ollama tag, 'vllm' caches HF weights, "
                       "'auto' prefers vllm for HF UniDrive repos and Ollama for Ollama tags."
                   ))

    p.add_argument("--yolo", action="store_true",
                   help="Download YOLO11 detection model (step 09; default model: yolo11l.pt ~48 MB)")
    p.add_argument("--yolo-model", default="yolo11l", metavar="MODEL",
                   help=(
                       "YOLO model filename to cache (without .pt extension). "
                       "Default: yolo11l (~48 MB, 25.3 M params). "
                       "Options: yolo11n (6 MB) | yolo11s (18 MB) | yolo11m (38 MB) "
                       "| yolo11l (48 MB) | yolo11x (109 MB)"
                   ))

    p.add_argument("--sam", action="store_true",
                   help="Download SAM3/SAM2 segmentation model (step 09; tries sam3 then sam2 fallback)")
    p.add_argument("--sam-model", default="facebook/sam3", metavar="MODEL_ID",
                   help=(
                       "SAM model repo ID to cache. "
                       "Default: facebook/sam3 (falls back to facebook/sam2-hiera-large if access not granted). "
                       "Options: facebook/sam3 | facebook/sam2-hiera-large | "
                       "facebook/sam2-hiera-base-plus"
                   ))

    return p


def _resolve_hf_model(task: str, override: str) -> str:
    """Return the model ID to use: *override* if set, else auto-select by VRAM."""
    mid = override.strip()
    if mid:
        return mid
    from selfsuvis.pipeline.vision.registry import auto_select, detect_resources
    selected = auto_select(task, detect_resources())
    if selected:
        log.info("%s auto-selected model: %s", task, selected)
    return selected or ""


def main() -> None:
    args = _build_parser().parse_args()

    _any_flag = (args.clip or args.dino or args.gemma or args.flash_attn or args.whisper
                 or args.florence or args.ocr or args.depth or args.detection or args.world_model
                 or args.unidrive or args.yolo or args.sam or args.all)
    # Default (no flag given) → behave as --all
    if not _any_flag:
        args.all = True
    do_clip        = args.clip        or args.all
    do_dino        = args.dino        or args.all
    do_gemma       = args.gemma       or args.all
    do_flash_attn  = args.flash_attn  or args.all
    do_whisper     = args.whisper     or args.all
    do_florence    = args.florence    or args.all
    do_ocr         = args.ocr         or args.all
    do_depth       = args.depth       or args.all
    do_detection   = args.detection   or args.all
    do_world_model = args.world_model or args.all
    do_unidrive    = args.unidrive    or args.all
    do_yolo        = args.yolo        or args.all
    do_sam         = args.sam         or args.all

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Auto device → %s", device)

    from selfsuvis.pipeline.core.config import settings

    # ── Resolve all HF model IDs up front ────────────────────────────────────
    errors: list = []
    whisper_id    = args.whisper_model
    florence_id   = args.florence_model
    ocr_id        = _resolve_hf_model("ocr",         args.ocr_model)      if do_ocr         else ""
    depth_id      = _resolve_hf_model("depth",       args.depth_model)    if do_depth       else ""
    detection_id  = _resolve_hf_model("detection",   args.detection_model) if do_detection  else ""
    world_id      = _resolve_hf_model("world_model", args.world_model_id) if do_world_model else ""
    unidrive_backend = ""
    unidrive_id = ""
    if do_unidrive:
        try:
            unidrive_backend = _resolve_unidrive_backend(args.unidrive_backend, args.unidrive_model)
            unidrive_id = _resolve_unidrive_prepare_model(args.unidrive_model, unidrive_backend)
            log.info("UniDrive prepare backend: %s  model=%s", unidrive_backend, unidrive_id)
        except Exception as exc:
            log.warning("UniDrive skipped: %s", exc)
            do_unidrive = False

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
        if do_unidrive and unidrive_id:
            if unidrive_backend == "ollama":
                specs.append((f"UniDriveVLA(Ollama) {unidrive_id}", lambda m=unidrive_id: _is_ollama_model_cached(m)))
            else:
                specs.append((f"UniDriveVLA(vLLM) {unidrive_id}", lambda m=unidrive_id: _is_hf_cached(m)))
        if do_yolo:
            ym = args.yolo_model
            specs.append((f"YOLO11 {ym}", lambda m=ym: _is_yolo_cached(m)))
        if do_sam:
            sm = args.sam_model
            _SAM2_FALLBACK = "facebook/sam2-hiera-large"
            def _sam_cached(m=sm, fb=_SAM2_FALLBACK):
                return _is_hf_cached(m) or _is_hf_cached(fb)
            label = f"SAM {sm} (or {_SAM2_FALLBACK} fallback)"
            specs.append((label, _sam_cached))

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

    if do_unidrive:
        if not unidrive_id:
            log.error("UniDriveVLA: could not determine a model ID — pass --unidrive-model explicitly")
            errors.append(("UniDriveVLA", ValueError("no model ID")))
        else:
            try:
                if unidrive_backend == "ollama":
                    _download_ollama_model(unidrive_id)
                else:
                    _with_auth_retry(f"UniDriveVLA ({unidrive_id})", unidrive_id,
                                     lambda: _download_unidrive(unidrive_id))
            except Exception as exc:
                log.error("UniDriveVLA [%s via %s] download failed: %s", unidrive_id, unidrive_backend or "unknown", exc)
                errors.append(("UniDriveVLA", exc))

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
