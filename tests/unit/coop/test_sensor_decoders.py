from datetime import timezone

from selfsuvis.coop.sensors.frigate_events import FrigateEventConsumer
from selfsuvis.coop.sensors.lorawan_decoder import decode_chirpstack_uplink


def test_decode_chirpstack_uplink_normalizes_types_and_timestamp() -> None:
    reading = decode_chirpstack_uplink(
        {
            "deviceInfo": {"devEui": "aabbccddeeff0011", "applicationId": "site-a"},
            "time": "2026-05-01T12:34:56",
            "fCnt": "42",
            "rxInfo": [{"rssi": "-71", "snr": "8.5"}],
            "object": {
                "temperature": "21.5",
                "humidity": "65",
                "motion": "false",
                "lat": "50.45",
                "lon": "30.52",
            },
        }
    )

    assert reading is not None
    assert reading.received_at.tzinfo is timezone.utc
    assert reading.f_cnt == 42
    assert reading.rssi == -71.0
    assert reading.snr == 8.5
    assert reading.temperature_c == 21.5
    assert reading.humidity_pct == 65.0
    assert reading.motion is False
    assert reading.gps_lat == 50.45
    assert reading.gps_lon == 30.52


def test_decode_chirpstack_uplink_ignores_unparseable_values() -> None:
    reading = decode_chirpstack_uplink(
        {
            "devEui": "bad-values",
            "fCnt": "not-an-int",
            "rxInfo": [{"rssi": "bad", "snr": None}],
            "object": {"temperature": "bad", "motion": "unknown"},
        }
    )

    assert reading is not None
    assert reading.f_cnt == 0
    assert reading.rssi is None
    assert reading.snr is None
    assert reading.temperature_c is None
    assert reading.motion is None


def test_frigate_event_decode_handles_numeric_strings_and_bad_values() -> None:
    event = FrigateEventConsumer.decode(
        {
            "type": "new",
            "after": {
                "id": "event-1",
                "camera": "entrance",
                "label": "person",
                "score": "0.88",
                "top_score": "bad",
                "start_time": 1777622400.0,
                "end_time": 0,
                "has_snapshot": True,
                "has_clip": False,
                "region": {"x": "0.1", "y": "bad", "width": 0.3},
            },
        }
    )

    assert event is not None
    assert event.score == 0.88
    assert event.top_score == 0.0
    assert event.ended_at is not None
    assert event.region == {"x": 0.1, "width": 0.3}
