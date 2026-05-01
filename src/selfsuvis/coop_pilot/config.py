"""coop_pilot runtime configuration — reads from environment / .env."""

import os


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


class CoopPilotSettings:
    """Settings for the coop_pilot sensor mesh integration."""

    # MQTT broker (Mosquitto)
    mqtt_host: str = _env("COOP_MQTT_HOST", "localhost")
    mqtt_port: int = int(_env("COOP_MQTT_PORT", "1883"))
    mqtt_user: str = _env("COOP_MQTT_USER", "")
    mqtt_password: str = _env("COOP_MQTT_PASSWORD", "")
    mqtt_tls: bool = _env("COOP_MQTT_TLS", "false").lower() == "true"

    # ChirpStack application topics
    # Pattern: application/{appId}/device/{devEUI}/event/up
    chirpstack_topic: str = _env("COOP_CHIRPSTACK_TOPIC", "application/+/device/+/event/up")

    # Frigate NVR
    frigate_topic_prefix: str = _env("COOP_FRIGATE_TOPIC_PREFIX", "frigate")
    frigate_api_url: str = _env("COOP_FRIGATE_API_URL", "http://localhost:8971")

    # Site state rolling window
    sensor_window_sec: int = int(_env("COOP_SENSOR_WINDOW_SEC", "300"))
    camera_event_window_sec: int = int(_env("COOP_CAMERA_EVENT_WINDOW_SEC", "120"))


settings = CoopPilotSettings()
