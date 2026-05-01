"""Tests for MQTT broker functionality."""

import time

import pytest


class TestMQTTConnectivity:
    """Test MQTT broker connectivity."""

    @pytest.mark.timeout(30)
    def test_health_user_connection(self, env_config, docker_client, container_names):
        """Health user can connect to MQTT broker and read $SYS topics."""
        container = docker_client.containers.get(container_names["mosquitto"])

        result = container.exec_run([
            "mosquitto_sub",
            "-h", "127.0.0.1",
            "-p", "1883",
            "-t", "$SYS/broker/version",
            "-C", "1",
            "-u", env_config["mqtt_health_user"],
            "-P", env_config["mqtt_health_password"],
        ])

        assert result.exit_code == 0, f"MQTT connection failed: {result.output}"

    @pytest.mark.timeout(30)
    def test_chirpstack_user_connection(self, env_config, docker_client, container_names):
        """ChirpStack user can publish to eu868 topics."""
        container = docker_client.containers.get(container_names["mosquitto"])

        result = container.exec_run([
            "mosquitto_pub",
            "-h", "127.0.0.1",
            "-p", "1883",
            "-t", "eu868/test/topic",
            "-m", "test_message",
            "-u", env_config["chirpstack_mqtt_user"],
            "-P", env_config["chirpstack_mqtt_password"],
        ])

        assert result.exit_code == 0, f"MQTT publish failed: {result.output}"


class TestMQTTTopicACL:
    """Test MQTT topic access control."""

    @pytest.mark.timeout(30)
    def test_health_user_can_read_sys_topics(self, env_config, docker_client, container_names):
        """Health user can subscribe to $SYS topics."""
        container = docker_client.containers.get(container_names["mosquitto"])

        result = container.exec_run([
            "mosquitto_sub",
            "-h", "127.0.0.1",
            "-p", "1883",
            "-t", "$SYS/broker/uptime",
            "-C", "1",
            "-u", env_config["mqtt_health_user"],
            "-P", env_config["mqtt_health_password"],
        ])

        assert result.exit_code == 0, f"Could not read $SYS topic: {result.output}"


class TestMQTTPerformance:
    """Test MQTT broker performance."""

    @pytest.mark.timeout(60)
    def test_message_throughput(self, env_config, docker_client, container_names):
        """MQTT broker sustains at least 5 publishes/second under sequential load."""
        container = docker_client.containers.get(container_names["mosquitto"])

        message_count = 100
        start_time = time.time()

        for i in range(message_count):
            result = container.exec_run([
                "mosquitto_pub",
                "-h", "127.0.0.1",
                "-p", "1883",
                "-t", f"eu868/test/perf/{i}",
                "-m", f"message_{i}",
                "-u", env_config["chirpstack_mqtt_user"],
                "-P", env_config["chirpstack_mqtt_password"],
            ])

            if result.exit_code != 0:
                pytest.fail(f"Message {i} failed: {result.output}")

        elapsed = time.time() - start_time
        messages_per_second = message_count / elapsed

        # Accounts for docker exec overhead; validates broker doesn't stall
        assert messages_per_second >= 5, \
            f"Low throughput: {messages_per_second:.2f} msg/s"
