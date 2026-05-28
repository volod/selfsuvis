import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from sslm.playground.catalog import ModelProfile
from sslm.playground.client import OpenAICompatibleClient


def service_name(model: ModelProfile) -> str:
    return "sslm_" + model.key.replace("-", "_")


def compose_service(model: ModelProfile) -> dict:
    service: dict = {
        "image": model.image,
        "user": "${SSLM_UID:-1000}:${SSLM_GID:-1000}",
        "entrypoint": ["vllm"],
        "command": model.vllm_command(),
        "ports": [f"{model.port}:8000"],
        "gpus": "all",
        # No ipc:host -- single-GPU vLLM does not need host IPC namespace, and
        # ipc:host causes PyTorch SHM to bypass Docker cgroup memory accounting,
        # which can exhaust host RAM and freeze the system.
        "shm_size": "4g",
        # Cap container memory to prevent host RAM exhaustion during long benchmarks.
        # memswap_limit == mem_limit means zero swap allowed for the container.
        # Override via SSLM_CONTAINER_MEM_LIMIT env var (e.g. export SSLM_CONTAINER_MEM_LIMIT=32g).
        "mem_limit": "${SSLM_CONTAINER_MEM_LIMIT:-24g}",
        "memswap_limit": "${SSLM_CONTAINER_MEM_LIMIT:-24g}",
        "mem_swappiness": 0,
        "environment": {
            "HF_HOME": "/data/hf-cache",
            # Point vLLM directly at the hub subdirectory where prefetch writes,
            # so both host-side snapshot_download and the container use the same path.
            "HF_HUB_CACHE": "/data/hf-cache/hub",
            # Accept both token env var names; run-benchmark.sh normalises them.
            "HF_TOKEN": "${HF_TOKEN:-}",
            "HUGGING_FACE_HUB_TOKEN": "${HUGGING_FACE_HUB_TOKEN:-}",
            "NVIDIA_VISIBLE_DEVICES": "${NVIDIA_VISIBLE_DEVICES:-all}",
            # The container runs as SSLM_UID:SSLM_GID (a real host UID with no
            # /etc/passwd entry inside the container).  torch 2.11 calls
            # getpass.getuser() -> pwd.getpwuid(uid) unconditionally at module
            # import time inside cache_dir_utils.py, which raises KeyError and
            # crashes vllm before the server starts.
            #
            # Fix: Python's getpass.getuser() checks LOGNAME/USER/LNAME/USERNAME
            # env vars first and returns early without ever calling getpwuid().
            # TORCHINDUCTOR_CACHE_DIR and HOME cover the remaining fallback
            # paths (expanduser, other inductor helpers) that also avoid getpwuid.
            "LOGNAME": "vllm",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
            "HOME": "/tmp",
            **model.env,
        },
        "volumes": [
            "${SSLM_HF_CACHE:-.data/sslm/hf-cache}:/data/hf-cache",
        ],
        "healthcheck": {
            "test": ["CMD-SHELL", "curl -sf http://localhost:8000/health || exit 1"],
            "interval": "30s",
            "timeout": "10s",
            "retries": 20,
            "start_period": "180s",
        },
        "deploy": {
            "resources": {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "count": 1,
                            "capabilities": ["gpu"],
                        }
                    ]
                }
            }
        },
    }
    if model.build_context and model.dockerfile:
        context = "${SSLM_PROJECT_ROOT:-.}" if model.build_context == "." else model.build_context
        service["build"] = {
            "context": context,
            "dockerfile": model.dockerfile,
            "args": {"MAX_JOBS": str(_cuda_build_jobs())},
        }
    return service


_PEAK_GB_PER_JOB = 12


def _cuda_build_jobs() -> int:
    """Return a safe ninja -j value for CUDA extension compilation.

    Two independent limits, take the minimum:
      cpu_limit  : max(1, (nproc - 2) // 2)  -- same formula as venv-rebuild-xformers
      mem_limit  : available_ram_gb // _PEAK_GB_PER_JOB
    """
    nproc = os.cpu_count() or 4
    cpu_limit = max(1, (nproc - 2) // 2)

    mem_limit = cpu_limit  # fallback if /proc/meminfo is unavailable
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_gb = int(line.split()[1]) // (1024 * 1024)
                mem_limit = max(1, available_gb // _PEAK_GB_PER_JOB)
                break
    except OSError:
        pass

    return min(cpu_limit, mem_limit)


def _remote_commit(url: str, ref: str, *, timeout: int = 30) -> str | None:
    """Return the HEAD commit hash for ref on a remote git URL, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", url, ref],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            first = result.stdout.strip().split("\n")[0]
            if first:
                return first.split()[0]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def render_compose(models: list[ModelProfile]) -> str:
    document = {
        "name": "sslm",
        "services": {service_name(model): compose_service(model) for model in models},
    }
    return yaml.safe_dump(document, sort_keys=False)


class DockerComposeSidecar:
    def __init__(self, compose_file: Path, model: ModelProfile) -> None:
        self.compose_file = compose_file
        self.model = model
        self.service = service_name(model)

    def prefetch(self) -> None:
        """Download model weights to the host HF cache before the container starts.

        Subsequent container runs find the weights already on disk via the bind mount
        and skip the HuggingFace download entirely.
        """
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
        from huggingface_hub.errors import (  # type: ignore[import-untyped]
            GatedRepoError,
            RepositoryNotFoundError,
        )

        env = self._compose_env()
        # snapshot_download's cache_dir == HF_HUB_CACHE == $HF_HOME/hub
        hub_cache = Path(env["SSLM_HF_CACHE"]) / "hub"
        hub_cache.mkdir(parents=True, exist_ok=True)
        # Accept both the new (HF_TOKEN) and legacy (HUGGING_FACE_HUB_TOKEN) names.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

        if token is None:
            print(
                "[prefetch] WARNING: HF_TOKEN is not set -- gated models will fail.\n"
                "           Add HF_TOKEN=<token> to .env or export it before running.\n"
                "           Get a token: https://huggingface.co/settings/tokens",
                flush=True,
            )

        # Fast path: skip network entirely if the snapshot is already in the cache.
        try:
            cached = snapshot_download(
                repo_id=self.model.model_id,
                cache_dir=str(hub_cache),
                local_files_only=True,
            )
            print(f"[prefetch] {self.model.model_id} already cached at {cached}", flush=True)
            return
        except Exception:
            pass  # not cached yet, fall through to download

        print(f"[prefetch] {self.model.model_id} -> {hub_cache}", flush=True)
        try:
            snapshot_download(
                repo_id=self.model.model_id,
                cache_dir=str(hub_cache),
                token=token,
            )
        except (GatedRepoError, RepositoryNotFoundError) as exc:
            model_url = f"https://huggingface.co/{self.model.model_id}"
            raise SystemExit(
                f"\n[prefetch] Cannot access {self.model.model_id} -- "
                f"token missing, expired, or license not accepted.\n"
                f"\n"
                f"  Model page (accept license here if required):\n"
                f"    {model_url}\n"
                f"\n"
                f"  Create or renew a token (Read access is enough):\n"
                f"    https://huggingface.co/settings/tokens\n"
                f"\n"
                f"  Add to project .env and re-run:\n"
                f"    HF_TOKEN=<your-token>\n"
                f"    make sslm-quick\n"
                f"\n"
                f"  Original error: {exc}"
            ) from None

    def _custom_image_exists(self) -> bool:
        if not self.model.build_context:
            return True  # stock image pulled on demand, no local build needed
        result = subprocess.run(
            ["docker", "image", "inspect", self.model.image],
            capture_output=True,
        )
        return result.returncode == 0

    # ── wheel cache helpers ───────────────────────────────────────────────────

    def _wheel_dir(self, env: dict[str, str]) -> Path:
        # Per-model subdir so the original pip-valid wheel filename is preserved.
        return Path(env["SSLM_PROJECT_ROOT"]) / ".data" / "sslm" / "wheels" / self._wheel_slug()

    def _wheel_slug(self) -> str:
        return self.model.image.replace("/", "_").replace(":", "_")

    def _wheel_whl(self, env: dict[str, str]) -> Path | None:
        matches = sorted(self._wheel_dir(env).glob("vllm*.whl"))
        return matches[0] if matches else None

    def _wheel_commit(self, env: dict[str, str]) -> Path:
        return self._wheel_dir(env) / ".commit"

    def _stored_commit(self, env: dict[str, str]) -> str | None:
        try:
            return self._wheel_commit(env).read_text().strip()
        except OSError:
            return None

    def _wheel_is_fresh(self, env: dict[str, str]) -> bool:
        """Return True when the stored wheel commit matches zaya1-pr HEAD (or on network failure)."""
        if not self.model.vllm_source:
            return True
        stored = self._stored_commit(env)
        if stored is None:
            return False
        url, _, branch = self.model.vllm_source.rpartition("@")
        remote = _remote_commit(url, f"refs/heads/{branch}")
        if remote is None:
            print("[wheel-cache] WARNING: could not reach remote, assuming wheel is current", flush=True)
            return True
        return stored == remote

    def _export_wheel(self, env: dict[str, str]) -> None:
        """Extract the compiled vllm wheel from the builder stage to the host wheel cache.

        Runs docker buildx build targeting wheel-export, which reuses BuildKit layers
        already cached from the docker compose up --build step (near-instant).
        """
        if not self.model.vllm_source or not self.model.dockerfile:
            return
        wheel_dir = self._wheel_dir(env)
        wheel_dir.mkdir(parents=True, exist_ok=True)
        # Remove any stale wheel from a previous build before exporting.
        for stale in wheel_dir.glob("vllm*.whl"):
            stale.unlink()
        context = env.get("SSLM_PROJECT_ROOT", ".")
        print(f"[wheel-cache] Extracting wheel to {wheel_dir} ...", flush=True)
        subprocess.check_call(
            [
                "docker",
                "buildx",
                "build",
                "--target",
                "wheel-export",
                "--output",
                f"type=local,dest={wheel_dir}",
                "--build-arg",
                f"MAX_JOBS={_cuda_build_jobs()}",
                "-f",
                self.model.dockerfile,
                context,
            ],
            env=env,
        )
        # Original pip-valid filename is preserved (no rename needed).
        for produced in wheel_dir.glob("vllm*.whl"):
            print(f"[wheel-cache] Wheel saved: {produced}", flush=True)
            break

        # Record the commit so the next run can skip the rebuild.
        url, _, branch = self.model.vllm_source.rpartition("@")
        remote = _remote_commit(url, f"refs/heads/{branch}")
        if remote:
            self._wheel_commit(env).write_text(remote)
            print(f"[wheel-cache] Pinned commit {remote[:12]}", flush=True)

    # ── container lifecycle ───────────────────────────────────────────────────

    def up(self, *, build: bool = False) -> None:
        env = self._compose_env()
        needs_build = build or not self._custom_image_exists()

        if needs_build and self.model.vllm_source:
            fresh = self._wheel_is_fresh(env)
            if fresh and self._custom_image_exists():
                # Explicit --build but source is unchanged: skip entirely.
                print(
                    f"[wheel-cache] {self.model.image}: source unchanged, skipping rebuild",
                    flush=True,
                )
                needs_build = False
            elif fresh and self.model.dockerfile:
                # Image is missing (e.g. after docker system prune) but the
                # wheel is current: rebuild from cache, skip CUDA compilation.
                if self._wheel_whl(env) is not None:
                    self._build_from_wheel(env)
                    needs_build = False

        command = ["docker", "compose", "-f", str(self.compose_file), "up", "-d"]
        if needs_build:
            command.append("--build")
        command.append(self.service)
        subprocess.check_call(command, env=env)

        if needs_build and self.model.vllm_source:
            self._export_wheel(env)

    def _build_from_wheel(self, env: dict[str, str]) -> None:
        """Rebuild the runtime image from the host-cached wheel (no CUDA compilation).

        Uses the from-wheel target in Dockerfile.zaya-vllm.  A minimal temp
        build context (Dockerfile + wheel only) keeps the transfer fast.
        """
        wheel = self._wheel_whl(env)
        dockerfile = self.model.dockerfile
        if wheel is None or not dockerfile:
            return
        print(f"[wheel-cache] Fast rebuild of {self.model.image} from cached wheel...", flush=True)
        with tempfile.TemporaryDirectory(prefix="sslm-wheel-build-") as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy2(wheel, tmp / wheel.name)  # preserve pip-valid filename
            shutil.copy2(dockerfile, tmp / "Dockerfile")
            subprocess.check_call(
                [
                    "docker", "buildx", "build",
                    "--target", "from-wheel",
                    "--tag", self.model.image,
                    "--load",
                    "-f", str(tmp / "Dockerfile"),
                    str(tmp),
                ],
                env=env,
            )
        print(f"[wheel-cache] Fast rebuild complete: {self.model.image}", flush=True)

    def down(self) -> None:
        subprocess.call(
            ["docker", "compose", "-f", str(self.compose_file), "rm", "-sf", self.service],
            env=self._compose_env(),
        )

    def dump_logs(self, tail: int = 100) -> None:
        subprocess.call(
            ["docker", "compose", "-f", str(self.compose_file), "logs", "--tail", str(tail), self.service],
            env=self._compose_env(),
        )

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        OpenAICompatibleClient(self.model.base_url).wait_until_ready(timeout_s=timeout_s)

    def _compose_env(self) -> dict[str, str]:
        env = os.environ.copy()
        project_root = Path.cwd()
        env.setdefault("SSLM_PROJECT_ROOT", str(project_root))
        # Default to the system HF cache so pipeline-cached models are reused.
        # run-benchmark.sh sets this before exec, so this fallback covers direct
        # Python invocations (e.g. tests or manual sslm sequential calls).
        system_hf = Path.home() / ".cache" / "huggingface"
        fallback = system_hf if system_hf.exists() else project_root / ".data" / "sslm" / "hf-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        env.setdefault("SSLM_HF_CACHE", str(fallback))
        env.setdefault("SSLM_UID", str(os.getuid()))
        env.setdefault("SSLM_GID", str(os.getgid()))
        return env
