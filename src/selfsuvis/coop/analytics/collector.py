"""Log collector for Docker containers in the coop-pilot stack."""

from datetime import datetime
from typing import Any


def _docker_client():
    try:
        import docker

        return docker.from_env()
    except ImportError as exc:
        raise ImportError(
            "docker is required for LogCollector. Install with: pip install 'selfsuvis[coop_pilot]'"
        ) from exc


def _get_not_found_exc():
    try:
        from docker.errors import NotFound

        return NotFound
    except ImportError:
        return Exception


_ContainerNotFound = _get_not_found_exc()


class LogCollector:
    """Collects logs and statistics from coop-pilot Docker containers."""

    CONTAINER_MAP: dict[str, str] = {
        "mosquitto": "coop-mosquitto",
        "chirpstack": "coop-chirpstack",
        "chirpstack_gw": "coop-cs-gwbridge",
        "chirpstack_rest": "coop-cs-rest",
        "chirpstack_postgres": "coop-cs-postgres",
        "redis": "coop-cs-redis",
        "frigate": "coop-frigate",
    }

    def __init__(self) -> None:
        self.docker_client = _docker_client()

    # -- Log collection --------------------------------------------------------

    def get_container_logs(
        self,
        service: str,
        since: datetime | None = None,
        until: datetime | None = None,
        tail: int | None = None,
    ) -> list[str]:
        container_name = self.CONTAINER_MAP.get(service, service)
        try:
            container = self.docker_client.containers.get(container_name)
        except _ContainerNotFound:
            return []

        kwargs: dict[str, Any] = {"timestamps": True, "stream": False}
        if since:
            kwargs["since"] = since
        if until:
            kwargs["until"] = until
        if tail:
            kwargs["tail"] = tail

        logs = container.logs(**kwargs)
        if isinstance(logs, bytes):
            logs = logs.decode("utf-8", errors="replace")
        return logs.strip().split("\n") if logs.strip() else []

    def get_all_service_logs(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        tail: int | None = 1000,
    ) -> dict[str, list[str]]:
        return {
            service: lines
            for service in self.CONTAINER_MAP
            if (lines := self.get_container_logs(service, since=since, until=until, tail=tail))
        }

    def get_mosquitto_file_logs(self, tail: int | None = None) -> list[str]:
        try:
            container = self.docker_client.containers.get(self.CONTAINER_MAP["mosquitto"])
            result = container.exec_run(["cat", "/mosquitto/log/mosquitto.log"])
            if result.exit_code == 0:
                lines = result.output.decode("utf-8", errors="replace").split("\n")
                return lines[-tail:] if tail else lines
        except (_ContainerNotFound, OSError, ValueError):
            pass
        return []

    # -- Container statistics --------------------------------------------------

    def get_container_stats(self, service: str) -> dict[str, Any] | None:
        container_name = self.CONTAINER_MAP.get(service, service)
        try:
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)
            cpu_percent = self._calculate_cpu_percent(stats)
            memory_usage = stats["memory_stats"].get("usage", 0)
            memory_limit = stats["memory_stats"].get("limit", 1)
            return {
                "name": container_name,
                "status": container.status,
                "cpu_percent": round(cpu_percent, 2),
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
                "memory_limit_mb": round(memory_limit / (1024 * 1024), 2),
                "memory_percent": round((memory_usage / memory_limit) * 100.0, 2),
            }
        except (_ContainerNotFound, KeyError):
            return None

    def _calculate_cpu_percent(self, stats: dict[str, Any]) -> float:
        cpu_delta = (
            stats["cpu_stats"]["cpu_usage"]["total_usage"]
            - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        )
        system_delta = (
            stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
        )
        return (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0

    def get_all_container_stats(self) -> dict[str, dict[str, Any]]:
        return {
            service: s for service in self.CONTAINER_MAP if (s := self.get_container_stats(service))
        }

    # -- Container health ------------------------------------------------------

    def get_container_health(self) -> dict[str, dict[str, Any]]:
        return {
            service: self._get_single_container_health(container_name)
            for service, container_name in self.CONTAINER_MAP.items()
        }

    def _get_single_container_health(self, container_name: str) -> dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_name)
            container.reload()
            health_state = container.attrs.get("State", {}).get("Health", {})
            return {
                "name": container_name,
                "status": container.status,
                "health": health_state.get("Status", "none"),
                "restart_count": container.attrs.get("RestartCount", 0),
            }
        except _ContainerNotFound:
            return {
                "name": container_name,
                "status": "not_found",
                "health": "unknown",
                "restart_count": 0,
            }
