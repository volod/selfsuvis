"""Decode ChirpStack MQTT uplink payloads into typed SensorReading objects.

ChirpStack publishes uplinks as JSON on:
  application/{applicationID}/device/{devEUI}/event/up

The payload contains base64-encoded device data in `data` plus decoded
object fields when a codec is configured (in `object`).  We support both
paths: codec-decoded objects take priority; raw `data` bytes are preserved
for caller-side decoding of custom payloads.
"""

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SensorReading:
    """Typed sensor measurement from a LoRaWAN end-device."""

    dev_eui: str
    application_id: str
    received_at: datetime
    f_cnt: int
    rssi: float | None
    snr: float | None

    # Decoded sensor values — populated from codec `object` when present
    temperature_c: float | None = None
    humidity_pct: float | None = None
    co2_ppm: float | None = None
    pressure_hpa: float | None = None
    battery_v: float | None = None
    motion: bool | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_alt_m: float | None = None

    # Raw payload bytes (base64-decoded)
    raw_bytes: bytes | None = None

    # Original decoded object from ChirpStack codec (passthrough)
    decoded_object: dict[str, Any] = field(default_factory=dict)


def decode_chirpstack_uplink(payload: str | bytes | dict) -> SensorReading | None:
    """Parse a ChirpStack uplink MQTT message into a SensorReading.

    Args:
        payload: Raw MQTT message payload — JSON string, bytes, or already-parsed dict.

    Returns:
        SensorReading on success, None if the message cannot be parsed.
    """
    try:
        if isinstance(payload, dict):
            msg = payload
        else:
            msg = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None

    dev_info = msg.get("deviceInfo", {})
    dev_eui = dev_info.get("devEui", msg.get("devEui", "unknown"))
    application_id = dev_info.get("applicationId", msg.get("applicationId", ""))

    received_at = _parse_timestamp(msg.get("time", msg.get("receivedAt")))

    # Radio stats from first RX window
    rx_info = (msg.get("rxInfo") or [{}])[0]
    rssi = _to_float(rx_info.get("rssi"))
    snr = _to_float(rx_info.get("snr"))

    raw_bytes: bytes | None = None
    if raw_b64 := msg.get("data"):
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception:
            pass

    decoded_object: dict[str, Any] = msg.get("object", {}) or {}

    reading = SensorReading(
        dev_eui=dev_eui,
        application_id=application_id,
        received_at=received_at,
        f_cnt=_to_int(msg.get("fCnt"), default=0),
        rssi=rssi,
        snr=snr,
        raw_bytes=raw_bytes,
        decoded_object=decoded_object,
    )

    _populate_known_fields(reading, decoded_object)
    return reading


# ── Codec field mapping ───────────────────────────────────────────────────────

# Each entry is (canonical_field_on_SensorReading, [possible_key_names_in_decoded_object])
_FIELD_ALIASES: list[tuple[str, list[str]]] = [
    ("temperature_c", ["temperature", "temp", "temperature_c", "TempC"]),
    ("humidity_pct", ["humidity", "relativeHumidity", "humidity_pct", "RH"]),
    ("co2_ppm", ["co2", "CO2", "co2_ppm"]),
    ("pressure_hpa", ["pressure", "pressure_hPa", "barometric_pressure"]),
    ("battery_v", ["battery", "batteryVoltage", "vbat", "battery_v"]),
    ("gps_lat", ["latitude", "lat", "gps_lat"]),
    ("gps_lon", ["longitude", "lon", "lng", "gps_lon"]),
    ("gps_alt_m", ["altitude", "alt", "alt_m", "gps_alt_m"]),
]

_MOTION_KEYS = {"motion", "pir", "occupancy", "presence", "motion_detected"}


def _populate_known_fields(reading: SensorReading, obj: dict[str, Any]) -> None:
    for attr, aliases in _FIELD_ALIASES:
        for alias in aliases:
            if alias in obj:
                if (value := _to_float(obj[alias])) is not None:
                    setattr(reading, attr, value)
                break

    for key in _MOTION_KEYS:
        if key in obj:
            reading.motion = _to_bool(obj[key])
            break


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on", "motion", "occupied"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", "none", "clear", "unoccupied"}:
            return False
    return None
