"""Log parser for the OpenRemote Manager."""

import re
from datetime import datetime
from typing import Any

from .base import BaseLogParser, LogEntry


class OpenRemoteLogParser(BaseLogParser):
    """Parser for OpenRemote Manager logs.

    Log format: "2026-03-09T11:28:52.577Z INFO [logger.name] message"
    """

    LOG_PATTERN = re.compile(
        r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)\s+"
        r"(\w+)\s+\[([^\]]+)\]\s+(.+)$"
    )

    def parse_line(self, line: str) -> LogEntry | None:
        line = line.strip()
        if not line:
            return None

        timestamp = None
        level = "INFO"
        message = line
        metadata: dict[str, Any] = {}

        if match := self.LOG_PATTERN.match(line):
            timestamp = self._parse_timestamp(match.group(1))
            level = match.group(2).upper()
            metadata["logger"] = match.group(3)
            message = match.group(4)

        return LogEntry(timestamp=timestamp, level=level, message=message, raw=line, metadata=metadata)

    def _parse_timestamp(self, ts_str: str) -> datetime | None:
        try:
            ts_str = ts_str.replace(" ", "T")
            if not (ts_str.endswith("Z") or "+" in ts_str or "-" in ts_str[-6:]):
                ts_str += "+00:00"
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            return None
