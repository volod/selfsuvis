"""Configuration for realtime_pilot engine integrations."""

import os


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


class RealtimePilotSettings:
    pose_backend: str = _env("REALTIME_POSE_BACKEND", "stub").strip().lower()
    pose_api_url: str = _env("REALTIME_POSE_API_URL", "").strip()
    occupancy_backend: str = _env("REALTIME_OCCUPANCY_BACKEND", "stub").strip().lower()
    occupancy_api_url: str = _env("REALTIME_OCCUPANCY_API_URL", "").strip()

    vins_fusion_api_url: str = _env("REALTIME_VINS_FUSION_API_URL", "http://realtime-vins-fusion:8101").strip()
    orbslam3_api_url: str = _env("REALTIME_ORBSLAM3_API_URL", "http://realtime-orbslam3:8101").strip()
    liosam_api_url: str = _env("REALTIME_LIOSAM_API_URL", "http://realtime-liosam:8101").strip()
    nvblox_api_url: str = _env("REALTIME_NVBLOX_API_URL", "http://realtime-nvblox:8101").strip()
    voxblox_api_url: str = _env("REALTIME_VOXBLOX_API_URL", "http://realtime-voxblox:8101").strip()


settings = RealtimePilotSettings()
