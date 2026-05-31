"""Base types shared by all service log parsers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LogEntry:
    """Parsed log entry with structured metadata.

    Attributes:
        timestamp: Parsed timestamp, if available.
        level: Log level (INFO, WARNING, ERROR, etc.).
        message: The log message content.
        raw: Original unparsed log line.
        metadata: Additional structured data extracted from the log.
    """

    timestamp: datetime | None
    level: str
    message: str
    raw: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLogParser(ABC):
    """Abstract base class for service-specific log parsers."""

    @abstractmethod
    def parse_line(self, line: str) -> LogEntry | None:
        """Parse a single log line into a structured LogEntry.

        Returns None if the line cannot be parsed.
        """

    def parse_lines(self, lines: list[str]) -> Iterator[LogEntry]:
        """Parse multiple log lines, yielding valid entries."""
        for line in lines:
            if entry := self.parse_line(line):
                yield entry

    def count_by_level(self, entries: list[LogEntry]) -> dict[str, int]:
        """Count log entries grouped by log level."""
        counts: dict[str, int] = {}
        for entry in entries:
            level = entry.level.upper()
            counts[level] = counts.get(level, 0) + 1
        return counts
