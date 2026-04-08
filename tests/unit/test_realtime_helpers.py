"""Unit tests for pure realtime helper methods."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.realtime.ingest import normalize_packets
from pipeline.realtime.occupancy import normalize_map_tile
from pipeline.realtime.pose import build_stub_pose_from_packet, pose_freshness_ms
from pipeline.realtime.semantics import normalize_semantic_observation


def test_normalize_packets_lowercases_and_coerces_types():
    packets = normalize_packets(
        [
            {
                "sensor_type": " GPS ",
                "t_device": "12.5",
                "seq": "7",
                "payload": {"east": 1},
            }
        ]
    )
    assert packets == [
        {
            "sensor_type": "gps",
            "t_device": 12.5,
            "seq": 7,
            "payload": {"east": 1},
        }
    ]


def test_normalize_packets_rejects_unknown_sensor_type():
    with pytest.raises(ValueError, match="unsupported sensor_type"):
        normalize_packets([{"sensor_type": "radar", "t_device": 1.0}])


def test_normalize_packets_requires_t_device():
    with pytest.raises(ValueError, match="packet missing t_device"):
        normalize_packets([{"sensor_type": "imu"}])


def test_normalize_map_tile_defaults_and_coercions():
    tile = normalize_map_tile(
        {
            "tile_key": 123,
            "map_type": " Occupancy ",
            "storage_path": "/tmp/a.bin",
            "resolution_m": "0.5",
            "stats": {"occupied": 10},
            "global_map_id": "4",
        }
    )
    assert tile == {
        "tile_key": "123",
        "map_type": "occupancy",
        "storage_path": "/tmp/a.bin",
        "resolution_m": 0.5,
        "bounds": {},
        "stats": {"occupied": 10},
        "global_map_id": 4,
    }


def test_normalize_semantic_observation_normalizes_and_preserves_fields():
    obs = normalize_semantic_observation(
        {
            "frame_id": "frame_1",
            "class_name": " Tree ",
            "confidence": "0.9",
            "position_enu": {"x": 1},
            "bbox": {"x1": 2},
            "mask_ref": "mask.png",
            "track_id": "trk-1",
            "facts": {"source": "yolo"},
        }
    )
    assert obs == {
        "frame_id": "frame_1",
        "class_name": "tree",
        "confidence": 0.9,
        "position_enu": {"x": 1},
        "bbox": {"x1": 2},
        "mask_ref": "mask.png",
        "track_id": "trk-1",
        "facts": {"source": "yolo"},
    }


def test_normalize_semantic_observation_rejects_missing_class_name():
    with pytest.raises(ValueError, match="class_name is required"):
        normalize_semantic_observation({"confidence": 0.2})


def test_normalize_semantic_observation_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence must be within \\[0, 1\\]"):
        normalize_semantic_observation({"class_name": "tree", "confidence": 1.2})


def test_build_stub_pose_from_packet_returns_pose_for_gps():
    pose = build_stub_pose_from_packet(
        {
            "sensor_type": "gps",
            "t_device": 12.5,
            "payload": {"east": 1.5, "north": -2.0, "up": 7.0, "global_map_id": 3},
        }
    )
    assert pose == {
        "source": "gps_fallback",
        "t_sec": 12.5,
        "position_enu": {"x": 1.5, "y": -2.0, "z": 7.0},
        "orientation_quat": None,
        "velocity_enu": None,
        "covariance": None,
        "tracking_status": "degraded",
        "global_map_id": 3,
    }


def test_build_stub_pose_from_packet_returns_none_without_required_fields():
    pose = build_stub_pose_from_packet(
        {"sensor_type": "gps", "t_device": 1.0, "payload": {"east": 1.0}}
    )
    assert pose is None


def test_pose_freshness_ms_handles_datetime_and_string():
    now = datetime(2026, 4, 8, 12, 0, 1, tzinfo=timezone.utc)
    created = now - timedelta(milliseconds=250)
    assert pose_freshness_ms(created, now=now) == 250
    assert pose_freshness_ms(created.isoformat(), now=now) == 250


def test_pose_freshness_ms_none_returns_none():
    assert pose_freshness_ms(None) is None
