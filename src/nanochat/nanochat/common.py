"""
Common utilities for nanochat.
"""

import logging
import os
import re
import urllib.request

import torch
import torch.distributed as dist
from filelock import FileLock

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )
    for logger_name in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # Default: <project_root>/.data/nanochat  (project-wide .data/ convention).
    # __file__ lives at src/nanochat/nanochat/common.py; parents[3] is the project root.
    # Override with NANOCHAT_BASE_DIR env var for any other location (e.g. /mnt/data/nanochat).
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ["NANOCHAT_BASE_DIR"]
    else:
        from pathlib import Path as _Path
        nanochat_dir = str(_Path(__file__).parents[3] / ".data" / "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls, see https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()

class SilentLogger:
    """No-op logger used when --run dummy is set."""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

class LocalLogger:
    """TensorBoard-backed logger with the same .log(dict)/.finish() interface as DummyWandb."""
    def __init__(self, run_name: str, project: str = "nanochat"):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(get_base_dir(), "tb_logs", project, run_name)
        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs -> {log_dir}  (tensorboard --logdir {os.path.join(get_base_dir(), 'tb_logs')})")

    def log(self, data: dict, **kwargs):
        step = data.get("step")
        for key, value in data.items():
            if key == "step":
                continue
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, global_step=step)

    def finish(self):
        self._writer.close()

# BF16 dense tensor-core throughput per SM per clock cycle (FP32 accumulate, no
# sparsity), keyed by CUDA compute capability. Used to ESTIMATE peak FLOPS for GPUs
# not in the exact table below so MFU stays meaningful instead of showing 0%.
# Calibrated against known parts, e.g. RTX 5090 (SM 12.0): 170 SM x 512 x 2.41 GHz
# ~= 209 TFLOPS, matching the tabulated value.
_FLOPS_PER_SM_PER_CYCLE_BF16 = {
    (7, 0): 1024,  # Volta (V100)
    (7, 5): 512,  # Turing (T4 / RTX 20)
    (8, 0): 2048,  # Ampere data center (A100 / A30)
    (8, 6): 512,  # Ampere consumer (RTX 30)
    (8, 7): 512,  # Ampere Jetson Orin
    (8, 9): 512,  # Ada Lovelace (RTX 40 / L4 / L40)
    (9, 0): 2048,  # Hopper (H100 / H200)
    (10, 0): 2048,  # Blackwell data center (B100 / B200)
    (12, 0): 512,  # Blackwell consumer / pro (RTX 50 / RTX PRO)
    (12, 1): 512,  # Blackwell GB10
}


def _query_max_sm_clock_ghz():
    """Best-effort max SM clock (GHz) from nvidia-smi; None if unavailable."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            mhz = int(out.stdout.strip().splitlines()[0])
            if mhz > 0:
                return mhz / 1000.0
    except Exception:
        pass
    return None


def estimate_peak_flops_bf16(compute_cap, sm_count, clock_ghz=None):
    """Estimate BF16 dense peak FLOPS for a GPU not in the exact table.

    peak = sm_count * flops_per_sm_per_cycle(arch) * clock_hz

    Returns None when SM count is unknown. Approximate (clock and per-SM throughput
    vary), but good enough to make MFU a useful signal for untabulated GPUs such as
    laptop / RTX PRO parts.
    """
    if not sm_count:
        return None
    fpc = _FLOPS_PER_SM_PER_CYCLE_BF16.get(tuple(compute_cap))
    if fpc is None:
        # Fall back to the architecture major-version default, else a safe consumer value.
        fpc = _FLOPS_PER_SM_PER_CYCLE_BF16.get((compute_cap[0], 0), 512)
    if clock_ghz is None:
        clock_ghz = _query_max_sm_clock_ghz() or 2.0  # typical boost-clock fallback
    return sm_count * fpc * clock_ghz * 1e9


# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str, device=None) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX — Blackwell
        (["5090"], 209.5e12),
        (["5080"], 137.7e12),
        (["5070 ti"], 107.4e12),
        (["5070"], 86.8e12),
        # Consumer RTX — Ada Lovelace
        (["4090"], 165.2e12),
        (["4080 super"], 105e12),
        (["4080"], 97.5e12),
        (["4070 ti super"], 90.5e12),
        (["4070 ti"], 82.6e12),
        (["4070 super"], 71.5e12),
        (["4070"], 58.5e12),
        (["4060 ti"], 44.3e12),
        (["4060"], 30.0e12),
        # Consumer RTX — Ampere
        (["3090 ti"], 80e12),
        (["3090"], 71e12),
        (["3080 ti"], 68.4e12),
        (["3080"], 51.5e12),
        (["3070 ti"], 43.7e12),
        (["3070"], 40.4e12),
        (["3060 ti"], 32.8e12),
        (["3060"], 25.3e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU: estimate peak FLOPS from SM count, architecture and clock so MFU
    # stays meaningful instead of inf -> 0% (e.g. laptop / RTX PRO Blackwell parts).
    if torch.cuda.is_available():
        try:
            idx = device if isinstance(device, int) else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            est = estimate_peak_flops_bf16((props.major, props.minor), props.multi_processor_count)
            if est:
                logger.info(
                    f"Peak FLOPS not tabulated for {device_name}; estimated {est:.2e} "
                    f"FLOPS (BF16) from {props.multi_processor_count} SMs @ SM "
                    f"{props.major}.{props.minor}. MFU is approximate."
                )
                return est
        except Exception:
            pass

    # Could not estimate - return inf so MFU shows as 0% rather than a wrong guess.
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')
