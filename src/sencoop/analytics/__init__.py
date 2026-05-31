"""Stack analytics — log collection, parsing and reporting for coop-pilot services.

LogCollector and LogAnalyzer require the `docker` package (selfsuvis[coop]).
Parsers and ReportRenderer have no heavy dependencies and can be imported freely.
"""

from .parsers import (
    BaseLogParser,
    ChirpStackLogParser,
    FrigateLogParser,
    LogEntry,
    MosquittoLogParser,
    OpenRemoteLogParser,
)

__all__ = [
    "BaseLogParser",
    "LogEntry",
    "ChirpStackLogParser",
    "FrigateLogParser",
    "MosquittoLogParser",
    "OpenRemoteLogParser",
    # Heavy (docker dep) — import directly from submodules:
    # from sencoop.analytics.collector import LogCollector
    # from sencoop.analytics.analyzer import LogAnalyzer
    # from sencoop.analytics.reporter import ReportRenderer
]
