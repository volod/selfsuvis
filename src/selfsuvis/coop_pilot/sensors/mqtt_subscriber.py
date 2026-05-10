"""Async MQTT subscriber that fans out ChirpStack LoRaWAN and Frigate NVR events.

Usage:
    subscriber = MqttSubscriber(on_sensor=handle_reading, on_camera=handle_event)
    await subscriber.run()   # blocks until cancelled

Requires the `aiomqtt` package (listed in the coop_pilot optional dep group).
"""

import asyncio
from collections.abc import Awaitable, Callable

from selfsuvis.pipeline.core import get_logger

from ..config import settings
from .frigate_events import CameraEvent, FrigateEventConsumer
from .lorawan_decoder import SensorReading, decode_chirpstack_uplink

logger = get_logger(__name__)

OnSensorCallback = Callable[[SensorReading], Awaitable[None]]
OnCameraCallback = Callable[[CameraEvent], Awaitable[None]]


class MqttSubscriber:
    """Subscribe to MQTT broker and dispatch typed sensor + camera events.

    Attributes:
        on_sensor: Async callback invoked for each decoded LoRaWAN uplink.
        on_camera: Async callback invoked for each Frigate camera event.
    """

    def __init__(
        self,
        on_sensor: OnSensorCallback | None = None,
        on_camera: OnCameraCallback | None = None,
    ) -> None:
        self.on_sensor = on_sensor
        self.on_camera = on_camera
        self._frigate_consumer = FrigateEventConsumer()
        self._prefix = settings.frigate_topic_prefix.rstrip("/")

    async def run(self, reconnect_interval: float = 5.0) -> None:
        """Connect to the MQTT broker and process messages indefinitely.

        Reconnects automatically on connection loss.
        """
        try:
            import aiomqtt  # noqa: F401 — verified at runtime
        except ImportError as exc:
            raise ImportError(
                "aiomqtt is required for MqttSubscriber. "
                "Install it with: pip install 'selfsuvis[coop_pilot]'"
            ) from exc

        import aiomqtt

        while True:
            try:
                async with aiomqtt.Client(
                    hostname=settings.mqtt_host,
                    port=settings.mqtt_port,
                    username=settings.mqtt_user or None,
                    password=settings.mqtt_password or None,
                    tls_params=aiomqtt.TLSParameters() if settings.mqtt_tls else None,
                ) as client:
                    logger.info("MQTT connected to %s:%d", settings.mqtt_host, settings.mqtt_port)
                    await client.subscribe(settings.chirpstack_topic)
                    await client.subscribe(f"{self._prefix}/events")
                    await client.subscribe(f"{self._prefix}/+/events")
                    logger.info(
                        "Subscribed to ChirpStack (%s) and Frigate (%s/events)",
                        settings.chirpstack_topic,
                        self._prefix,
                    )

                    async for message in client.messages:
                        await self._dispatch(str(message.topic), message.payload)

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MQTT connection lost (%s), reconnecting in %ss", exc, reconnect_interval
                )
                await asyncio.sleep(reconnect_interval)

    async def _dispatch(self, topic: str, payload: bytes) -> None:
        """Route an incoming message to the appropriate handler."""
        if topic.startswith(f"{self._prefix}/") and topic.endswith("/events"):
            await self._handle_frigate(payload)
        elif "/event/up" in topic:
            await self._handle_lorawan(payload)

    async def _handle_lorawan(self, payload: bytes) -> None:
        reading = decode_chirpstack_uplink(payload)
        if reading and self.on_sensor:
            try:
                await self.on_sensor(reading)
            except Exception:
                logger.exception("Error in on_sensor callback")

    async def _handle_frigate(self, payload: bytes) -> None:
        event = self._frigate_consumer.decode(payload)
        if event and self.on_camera:
            try:
                await self.on_camera(event)
            except Exception:
                logger.exception("Error in on_camera callback")
