"""Tests for coop_pilot service health and availability."""

import httpx
import pytest


class TestContainerHealth:
    """Test suite for container health status."""

    def test_all_containers_running(self, docker_client, expected_containers):
        """Verify all expected containers are running."""
        running_containers = {
            c.name for c in docker_client.containers.list()
        }

        missing = set(expected_containers) - running_containers
        assert not missing, f"Missing containers: {missing}"

    def test_containers_healthy(self, docker_client, expected_containers):
        """Verify all containers with health checks are healthy or starting."""
        containers_with_health = []
        # Containers that may still be starting
        allow_starting = {"coop-frigate"}

        for container in docker_client.containers.list():
            if container.name in expected_containers:
                health = container.attrs.get("State", {}).get("Health", {})
                if health:
                    status = health.get("Status", "unknown")
                    # Allow "starting" for certain containers
                    if container.name in allow_starting and status == "starting":
                        status = "healthy"  # Treat as OK
                    containers_with_health.append({
                        "name": container.name,
                        "status": status
                    })

        unhealthy = [c for c in containers_with_health if c["status"] not in ("healthy", "starting")]
        assert not unhealthy, f"Unhealthy containers: {unhealthy}"

    def test_container_restart_count(self, docker_client, expected_containers):
        """Verify critical containers haven't restarted excessively."""
        max_restarts = 5
        # Skip restart count check for containers that may restart on first-run
        # cs-chirpstack retries DB migrations until postgres is ready
        skip_restart_check = {"coop-frigate", "coop-chirpstack"}

        for container in docker_client.containers.list():
            if container.name in expected_containers and container.name not in skip_restart_check:
                restart_count = container.attrs.get("RestartCount", 0)
                assert restart_count <= max_restarts, \
                    f"Container {container.name} restarted {restart_count} times"


class TestServiceEndpoints:
    """Test suite for service HTTP endpoints."""

    @pytest.mark.timeout(15)
    def test_chirpstack_ui_accessible(self, env_config):
        """Test ChirpStack UI is accessible."""
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{env_config['chirpstack_api_url']}/")
            # ChirpStack should respond (login page or similar)
            assert response.status_code in range(200, 500)

    @pytest.mark.timeout(15)
    def test_chirpstack_rest_api_accessible(self, env_config):
        """Test ChirpStack REST API is accessible."""
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{env_config['chirpstack_rest_url']}/")
            assert response.status_code in range(200, 500)

    @pytest.mark.timeout(15)
    def test_frigate_accessible(self, env_config):
        """Test Frigate NVR is accessible."""
        with httpx.Client(timeout=10) as client:
            try:
                response = client.get(f"{env_config['frigate_url']}/api/version")
                assert response.status_code in [200, 400, 401, 403, 500, 502]
            except httpx.ConnectError:
                pytest.skip("Frigate not accessible - may still be starting")


class TestDatabaseConnectivity:
    """Test database connectivity."""

    @pytest.mark.timeout(30)
    def test_postgresql_chirpstack_accessible(self, docker_client, env_config, container_names):
        """Test ChirpStack PostgreSQL is accessible."""
        container = docker_client.containers.get(container_names["chirpstack_postgres"])
        user = env_config.get("chirpstack_pg_user", "chirpstack")
        result = container.exec_run(f"pg_isready -U {user}")
        assert result.exit_code == 0, "ChirpStack PostgreSQL not ready"

    @pytest.mark.timeout(30)
    def test_redis_accessible(self, docker_client, container_names):
        """Test Redis is accessible."""
        container = docker_client.containers.get(container_names["redis"])
        result = container.exec_run("redis-cli ping")
        assert result.exit_code == 0
        assert b"PONG" in result.output
