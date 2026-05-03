"""coop_pilot runtime configuration — reads from environment / .env."""

from selfsuvis.pipeline.core.env import env_bool, env_int, env_str


class CoopPilotSettings:
    """Settings for the coop_pilot sensor mesh integration."""

    # MQTT broker (Mosquitto)
    mqtt_host: str = env_str("COOP_MQTT_HOST", "localhost")
    mqtt_port: int = env_int("COOP_MQTT_PORT", 1883)
    mqtt_user: str = env_str("COOP_MQTT_USER", "")
    mqtt_password: str = env_str("COOP_MQTT_PASSWORD", "")
    mqtt_tls: bool = env_bool("COOP_MQTT_TLS", False)

    # ChirpStack application topics
    # Pattern: application/{appId}/device/{devEUI}/event/up
    chirpstack_topic: str = env_str("COOP_CHIRPSTACK_TOPIC", "application/+/device/+/event/up")

    # Frigate NVR
    frigate_topic_prefix: str = env_str("COOP_FRIGATE_TOPIC_PREFIX", "frigate")
    frigate_api_url: str = env_str("COOP_FRIGATE_API_URL", "http://localhost:8971")

    # Site state rolling window
    sensor_window_sec: int = env_int("COOP_SENSOR_WINDOW_SEC", 300)
    camera_event_window_sec: int = env_int("COOP_CAMERA_EVENT_WINDOW_SEC", 120)


settings = CoopPilotSettings()
