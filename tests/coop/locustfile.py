"""Load testing for Stack A Pilot services using Locust."""

import os
from pathlib import Path

from dotenv import load_dotenv
from locust import HttpUser, between, events, task

load_dotenv(Path(__file__).resolve().parent.parent / "data" / ".env")


class ChirpStackUser(HttpUser):
    """Simulated user interacting with ChirpStack API."""

    host = f"http://localhost:{os.getenv('CHIRPSTACK_REST_PORT', '8090')}"
    wait_time = between(1, 3)

    def on_start(self):
        """Initialize user session."""
        self.api_key = os.getenv("CHIRPSTACK_API_SECRET", "")

    @task(3)
    def get_device_profiles(self):
        """List device profiles."""
        headers = {"Grpc-Metadata-Authorization": f"Bearer {self.api_key}"}
        with self.client.get(
            "/api/device-profiles",
            headers=headers,
            catch_response=True,
            name="List Device Profiles",
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def get_applications(self):
        """List applications."""
        headers = {"Grpc-Metadata-Authorization": f"Bearer {self.api_key}"}
        with self.client.get(
            "/api/applications", headers=headers, catch_response=True, name="List Applications"
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def get_gateways(self):
        """List gateways."""
        headers = {"Grpc-Metadata-Authorization": f"Bearer {self.api_key}"}
        with self.client.get(
            "/api/gateways", headers=headers, catch_response=True, name="List Gateways"
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class OpenRemoteUser(HttpUser):
    """Simulated user interacting with OpenRemote API."""

    host = "http://localhost:8405"
    wait_time = between(1, 5)

    @task(3)
    def get_master_info(self):
        """Get master realm info."""
        with self.client.get(
            "/api/master/info", catch_response=True, name="Get Master Info"
        ) as response:
            if response.status_code in [200, 401, 403, 404]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def get_realm_info(self):
        """Get realm info."""
        with self.client.get(
            "/api/master/realm/info", catch_response=True, name="Get Realm Info"
        ) as response:
            if response.status_code in [200, 401, 403, 404]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def health_check(self):
        """Simple health check."""
        with self.client.get("/", catch_response=True, name="Health Check") as response:
            if response.status_code in [200, 302, 401, 403, 404]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class FrigateUser(HttpUser):
    """Simulated user interacting with Frigate NVR API."""

    host = f"http://localhost:{os.getenv('FRIGATE_PORT', '8971')}"
    wait_time = between(2, 5)

    @task(3)
    def get_stats(self):
        """Get Frigate statistics."""
        with self.client.get("/api/stats", catch_response=True, name="Get Stats") as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def get_config(self):
        """Get Frigate configuration."""
        with self.client.get("/api/config", catch_response=True, name="Get Config") as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def get_version(self):
        """Get Frigate version."""
        with self.client.get("/api/version", catch_response=True, name="Get Version") as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


# Event hooks for custom reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("=" * 60)
    print("Stack A Pilot Load Test Starting")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("=" * 60)
    print("Stack A Pilot Load Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    import subprocess

    subprocess.run(
        [
            "locust",
            "-f",
            __file__,
            "--headless",
            "-u",
            "10",  # 10 users
            "-r",
            "2",  # spawn rate
            "-t",
            "60s",  # run time
        ]
    )
