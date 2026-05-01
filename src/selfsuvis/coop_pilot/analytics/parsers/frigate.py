"""Log parser for the Frigate NVR."""

import re
from datetime import datetime
from typing import Any

from .base import BaseLogParser, LogEntry


class FrigateLogParser(BaseLogParser):
    """Parser for Frigate Network Video Recorder logs.

    Log format: "2026-03-09 11:28:52.577  [INFO] message"
    """

    LOG_PATTERN = re.compile(
        r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+\[(\w+)\]\s+(.+)$"
    )
    DETECTION_PATTERN = re.compile(r"detected (?P<object>\w+) in (?P<camera>\w+)")
    MOTION_PATTERN = re.compile(r"motion detected")

    def parse_line(self, line: str) -> LogEntry | None:
        line = line.strip()
        if not line:
            return None

        timestamp = None
        level = "INFO"
        message = line

        if match := self.LOG_PATTERN.match(line):
            try:
                timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                pass
            level = match.group(2).upper()
            message = match.group(3)

        metadata = self._extract_event_metadata(message)

        return LogEntry(timestamp=timestamp, level=level, message=message, raw=line, metadata=metadata)

    def _extract_event_metadata(self, message: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        if det_match := self.DETECTION_PATTERN.search(message):
            metadata["event"] = "detection"
            metadata["object"] = det_match.group("object")
            metadata["camera"] = det_match.group("camera")
        elif self.MOTION_PATTERN.search(message):
            metadata["event"] = "motion"

        return metadata

    def get_detection_stats(self, entries: list[LogEntry]) -> dict[str, Any]:
        """Calculate detection statistics from parsed log entries."""
        objects_detected: dict[str, int] = {}
        cameras_active: set[str] = set()
        total_detections = 0
        motion_events = 0

        for entry in entries:
            event = entry.metadata.get("event")

            if event == "detection":
                total_detections += 1
                obj = entry.metadata.get("object", "unknown")
                objects_detected[obj] = objects_detected.get(obj, 0) + 1
                if camera := entry.metadata.get("camera"):
                    cameras_active.add(camera)
            elif event == "motion":
                motion_events += 1

        return {
            "total_detections": total_detections,
            "objects_detected": objects_detected,
            "cameras_active": list(cameras_active),
            "motion_events": motion_events,
        }
