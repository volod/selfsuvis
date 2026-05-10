"""Log parser for the Mosquitto MQTT broker."""

import re
from datetime import datetime
from typing import Any

from .base import BaseLogParser, LogEntry


class MosquittoLogParser(BaseLogParser):
    """Parser for Mosquitto MQTT broker logs.

    Log format: Unix timestamp followed by message.
    Example: "1773052744: mosquitto version 2.1.2 starting"
    """

    TIMESTAMP_PATTERN = re.compile(r"^(\d+):\s+(.+)$")
    CLIENT_CONNECT_PATTERN = re.compile(
        r"New client connected from (?P<ip>[\d.]+):(?P<port>\d+) as (?P<client_id>\S+)"
    )
    CLIENT_DISCONNECT_PATTERN = re.compile(r"Client (?P<client_id>\S+) disconnected")
    AUTH_FAILURE_PATTERN = re.compile(
        r"Client (?P<client_id>\S+) (?:not authorized|authentication failure)"
    )

    def parse_line(self, line: str) -> LogEntry | None:
        line = line.strip()
        if not line:
            return None

        timestamp = None
        message = line
        metadata: dict[str, Any] = {}

        if match := self.TIMESTAMP_PATTERN.match(line):
            try:
                timestamp = datetime.fromtimestamp(int(match.group(1)))
            except (ValueError, OSError):
                pass
            message = match.group(2)

        level = self._infer_level(message)
        self._extract_connection_metadata(message, metadata)

        if metadata.get("event") == "auth_failure":
            level = "WARNING"

        return LogEntry(
            timestamp=timestamp, level=level, message=message, raw=line, metadata=metadata
        )

    def _infer_level(self, message: str) -> str:
        msg_lower = message.lower()
        if "error" in msg_lower or "fail" in msg_lower:
            return "ERROR"
        if "warning" in msg_lower or "warn" in msg_lower:
            return "WARNING"
        return "INFO"

    def _extract_connection_metadata(self, message: str, metadata: dict[str, Any]) -> None:
        if conn_match := self.CLIENT_CONNECT_PATTERN.search(message):
            metadata["event"] = "client_connect"
            metadata["client_id"] = conn_match.group("client_id")
            metadata["ip"] = conn_match.group("ip")
            metadata["port"] = conn_match.group("port")
        elif disconn_match := self.CLIENT_DISCONNECT_PATTERN.search(message):
            metadata["event"] = "client_disconnect"
            metadata["client_id"] = disconn_match.group("client_id")
        elif auth_match := self.AUTH_FAILURE_PATTERN.search(message):
            metadata["event"] = "auth_failure"
            metadata["client_id"] = auth_match.group("client_id")

    def get_connection_stats(self, entries: list[LogEntry]) -> dict[str, Any]:
        """Calculate connection statistics from parsed log entries."""
        unique_clients: set[str] = set()
        unique_ips: set[str] = set()
        total_connects = 0
        total_disconnects = 0
        auth_failures = 0

        for entry in entries:
            event = entry.metadata.get("event")
            if event == "client_connect":
                total_connects += 1
                if client_id := entry.metadata.get("client_id"):
                    unique_clients.add(client_id)
                if ip := entry.metadata.get("ip"):
                    unique_ips.add(ip)
            elif event == "client_disconnect":
                total_disconnects += 1
            elif event == "auth_failure":
                auth_failures += 1

        return {
            "total_connects": total_connects,
            "total_disconnects": total_disconnects,
            "auth_failures": auth_failures,
            "unique_clients": len(unique_clients),
            "unique_ips": len(unique_ips),
        }
