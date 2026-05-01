"""Stack analytics — log collection, parsing and reporting for coop-pilot services.

LogCollector and LogAnalyzer require the `docker` package (selfsuvis[coop_pilot]).
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
    # from selfsuvis.coop_pilot.analytics.collector import LogCollector
    # from selfsuvis.coop_pilot.analytics.analyzer import LogAnalyzer
    # from selfsuvis.coop_pilot.analytics.reporter import ReportRenderer
]
