from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

from sslm.playground.catalog import ModelProfile
from sslm.playground.client import OpenAICompatibleClient


def service_name(model: ModelProfile) -> str:
    return "sslm_" + model.key.replace("-", "_")


def compose_service(model: ModelProfile) -> dict:
    service: dict = {
        "image": model.image,
        "entrypoint": ["vllm"],
        "command": model.vllm_command(),
        "ports": [f"{model.port}:8000"],
        "gpus": "all",
        "ipc": "host",
        "shm_size": "16g",
        "environment": {
            "HF_HOME": "/root/.cache/huggingface",
            "HUGGING_FACE_HUB_TOKEN": "${HUGGING_FACE_HUB_TOKEN:-}",
            "NVIDIA_VISIBLE_DEVICES": "${NVIDIA_VISIBLE_DEVICES:-all}",
            **model.env,
        },
        "volumes": [
            "${SSLM_HF_CACHE:-.data/sslm/hf-cache}:/root/.cache/huggingface",
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
        service["build"] = {"context": context, "dockerfile": model.dockerfile}
    return service


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

    def up(self, *, build: bool = False) -> None:
        env = self._compose_env()
        command = ["docker", "compose", "-f", str(self.compose_file), "up", "-d"]
        if build:
            command.append("--build")
        command.append(self.service)
        subprocess.check_call(command, env=env)

    def down(self) -> None:
        subprocess.call(
            ["docker", "compose", "-f", str(self.compose_file), "rm", "-sf", self.service],
            env=self._compose_env(),
        )

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        OpenAICompatibleClient(self.model.base_url).wait_until_ready(timeout_s=timeout_s)

    def _compose_env(self) -> dict[str, str]:
        env = os.environ.copy()
        project_root = Path.cwd()
        env.setdefault("SSLM_PROJECT_ROOT", str(project_root))
        cache_dir = project_root / ".data" / "sslm" / "hf-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("SSLM_HF_CACHE", str(cache_dir))
        return env
