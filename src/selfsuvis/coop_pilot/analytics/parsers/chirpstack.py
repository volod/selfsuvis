"""Log parser for the ChirpStack LoRaWAN Network Server."""

import re
from datetime import datetime
from typing import Any

from .base import BaseLogParser, LogEntry


class ChirpStackLogParser(BaseLogParser):
    """Parser for ChirpStack LoRaWAN Network Server logs.

    Parses device activity, gateway communications, and uplink/downlink
    events from ChirpStack Docker log output.
    """

    DOCKER_TS_PATTERN = re.compile(
        r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+(.+)$"
    )
    LEVEL_PATTERN = re.compile(r"\b(DEBUG|INFO|WARN|ERROR|FATAL)\b", re.IGNORECASE)
    DEVICE_PATTERN = re.compile(r"dev_eui=([a-fA-F0-9]{16})")
    GATEWAY_PATTERN = re.compile(r"gateway_id=([a-fA-F0-9]{16})")
    UPLINK_PATTERN = re.compile(r"uplink|rx|received", re.IGNORECASE)
    DOWNLINK_PATTERN = re.compile(r"downlink|tx|transmitted", re.IGNORECASE)

    def parse_line(self, line: str) -> LogEntry | None:
        line = line.strip()
        if not line:
            return None

        # Single match reused for both timestamp and message extraction
        match = self.DOCKER_TS_PATTERN.match(line)
        timestamp = self._parse_timestamp(match)
        message = match.group(2) if match else line
        level = self._extract_level(message)
        metadata = self._extract_lorawan_metadata(message)

        return LogEntry(timestamp=timestamp, level=level, message=message, raw=line, metadata=metadata)

    def _parse_timestamp(self, match: re.Match | None) -> datetime | None:
        if not match:
            return None
        try:
            ts_str = match.group(1)
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _extract_level(self, message: str) -> str:
        if level_match := self.LEVEL_PATTERN.search(message):
            return level_match.group(1).upper()
        return "INFO"

    def _extract_lorawan_metadata(self, message: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        if dev_match := self.DEVICE_PATTERN.search(message):
            metadata["dev_eui"] = dev_match.group(1)

        if gw_match := self.GATEWAY_PATTERN.search(message):
            metadata["gateway_id"] = gw_match.group(1)

        if self.UPLINK_PATTERN.search(message):
            metadata["direction"] = "uplink"
        elif self.DOWNLINK_PATTERN.search(message):
            metadata["direction"] = "downlink"

        return metadata

    def get_lorawan_stats(self, entries: list[LogEntry]) -> dict[str, Any]:
        """Calculate LoRaWAN activity statistics from parsed log entries.

        Returns counts of uplinks/downlinks, unique devices and gateways, and errors.
        """
        unique_devices: set[str] = set()
        unique_gateways: set[str] = set()
        uplinks = 0
        downlinks = 0
        errors = 0

        for entry in entries:
            direction = entry.metadata.get("direction")
            if direction == "uplink":
                uplinks += 1
            elif direction == "downlink":
                downlinks += 1

            if dev_eui := entry.metadata.get("dev_eui"):
                unique_devices.add(dev_eui)

            if gw_id := entry.metadata.get("gateway_id"):
                unique_gateways.add(gw_id)

            if entry.level in ("ERROR", "FATAL"):
                errors += 1

        return {
            "total_messages": len(entries),
            "uplinks": uplinks,
            "downlinks": downlinks,
            "unique_devices": len(unique_devices),
            "unique_gateways": len(unique_gateways),
            "errors": errors,
        }
