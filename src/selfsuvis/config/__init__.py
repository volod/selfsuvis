"""Unified configuration facade for all three selfsuvis playgrounds.

Provides a single import point for settings from every component:

    from selfsuvis.config import settings          # core pipeline + server
    from selfsuvis.config import coop_settings     # coop sensor mesh
    from selfsuvis.config import realtime_settings # realtime SLAM bridges
    from selfsuvis.config import validate_all      # validate all three

Each component's settings class is also importable directly:

    from selfsuvis.config import Settings, CoopPilotSettings, RealtimePilotSettings

The underlying config lives in each component's own module (not here) so
existing imports like ``from selfsuvis.pipeline.core.config import settings``
continue to work without change.

GRASP note: this package is a Pure Fabrication.  It does not represent a
domain concept -- it exists solely to reduce coupling between callers and the
three config subsystems, and to provide a single place to run cross-component
validation.
"""

from selfsuvis.pipeline.core.config import Settings
from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.config import validate_settings as _validate_core
from selfsuvis.pipeline.core.config import mask_secret

from sencoop.config import CoopPilotSettings
from sencoop.config import settings as coop_settings

from selfsuvis.realtime.config import RealtimePilotSettings
from selfsuvis.realtime.config import settings as realtime_settings


def validate_all() -> None:
    """Validate settings for all three playgrounds.

    Runs the core pipeline/server validation first, then validates
    coop and realtime settings that have no equivalent check today.

    Raises ValueError for hard configuration errors.
    Emits logger.warning for soft problems (missing optional credentials, etc.).
    """
    from selfsuvis.pipeline.core.logging import get_logger

    logger = get_logger(__name__)

    # -- Playground 1: core pipeline + production server ----------------------
    _validate_core()

    # -- Playground 2: local pipeline (no extra cross-checks needed here;
    #    pipeline-specific validation happens inside the pipeline preflight) ---

    # -- Playground 3: coop sensor mesh ---------------------------------------
    if coop_settings.mqtt_tls:
        if not coop_settings.mqtt_user:
            logger.warning(
                "COOP_MQTT_TLS is enabled but COOP_MQTT_USER is not set; "
                "anonymous TLS connections may be rejected by the broker."
            )
        if not coop_settings.mqtt_password:
            logger.warning(
                "COOP_MQTT_TLS is enabled but COOP_MQTT_PASSWORD is not set."
            )
    if coop_settings.sensor_window_sec < 10:
        raise ValueError("COOP_SENSOR_WINDOW_SEC must be >= 10 seconds")
    if coop_settings.camera_event_window_sec < 10:
        raise ValueError("COOP_CAMERA_EVENT_WINDOW_SEC must be >= 10 seconds")

    # -- Realtime SLAM bridges ------------------------------------------------
    pose_backends = {"stub", "vins_fusion", "orbslam3", "liosam"}
    if realtime_settings.pose_backend not in pose_backends:
        raise ValueError(
            f"REALTIME_POSE_BACKEND must be one of {sorted(pose_backends)}"
        )
    occupancy_backends = {"stub", "nvblox", "voxblox"}
    if realtime_settings.occupancy_backend not in occupancy_backends:
        raise ValueError(
            f"REALTIME_OCCUPANCY_BACKEND must be one of {sorted(occupancy_backends)}"
        )

    logger.info("All playground settings validated successfully")


__all__ = [
    "Settings",
    "settings",
    "CoopPilotSettings",
    "coop_settings",
    "RealtimePilotSettings",
    "realtime_settings",
    "mask_secret",
    "validate_all",
]
