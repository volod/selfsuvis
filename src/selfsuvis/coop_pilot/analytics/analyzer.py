"""Log analysis and summary generation for coop-pilot stack services."""

from datetime import datetime
from typing import Any

import pandas as pd

from .collector import LogCollector
from .parsers import (
    ChirpStackLogParser,
    FrigateLogParser,
    LogEntry,
    MosquittoLogParser,
    OpenRemoteLogParser,
)


class LogAnalyzer:
    """Collects, parses, and analyzes logs from all coop-pilot stack services."""

    def __init__(self) -> None:
        self.collector = LogCollector()
        self.parsers = {
            "mosquitto": MosquittoLogParser(),
            "chirpstack": ChirpStackLogParser(),
            "chirpstack_gw": ChirpStackLogParser(),
            "frigate": FrigateLogParser(),
            "manager": OpenRemoteLogParser(),
            "proxy": OpenRemoteLogParser(),
        }

    # ── Collection ────────────────────────────────────────────────────────────

    def collect_and_parse(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        tail: int = 1000,
    ) -> dict[str, list[LogEntry]]:
        raw_logs = self.collector.get_all_service_logs(since=since, until=until, tail=tail)
        return {
            service: self._parse_service_logs(service, lines) for service, lines in raw_logs.items()
        }

    def _parse_service_logs(self, service: str, lines: list[str]) -> list[LogEntry]:
        parser = self.parsers.get(service)
        if parser:
            return list(parser.parse_lines(lines))
        return [
            LogEntry(timestamp=None, level="INFO", message=line, raw=line)
            for line in lines
            if line.strip()
        ]

    # ── Summaries ─────────────────────────────────────────────────────────────

    def get_error_summary(self, parsed_logs: dict[str, list[LogEntry]]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "total_errors": 0,
            "total_warnings": 0,
            "by_service": {},
            "recent_errors": [],
        }
        for service, entries in parsed_logs.items():
            errors = [e for e in entries if e.level in ("ERROR", "FATAL")]
            warnings = [e for e in entries if e.level == "WARNING"]
            summary["by_service"][service] = {"errors": len(errors), "warnings": len(warnings)}
            summary["total_errors"] += len(errors)
            summary["total_warnings"] += len(warnings)
            for entry in errors[-5:]:
                summary["recent_errors"].append(
                    {
                        "service": service,
                        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                        "message": entry.message[:200],
                    }
                )
        summary["recent_errors"].sort(key=lambda x: x["timestamp"] or "", reverse=True)
        summary["recent_errors"] = summary["recent_errors"][:20]
        return summary

    def get_mqtt_summary(self, parsed_logs: dict[str, list[LogEntry]]) -> dict[str, Any]:
        if "mosquitto" not in parsed_logs:
            return {"available": False}
        parser = MosquittoLogParser()
        entries = parsed_logs["mosquitto"]
        return {
            "available": True,
            "connections": parser.get_connection_stats(entries),
            "log_levels": parser.count_by_level(entries),
            "total_log_entries": len(entries),
        }

    def get_lorawan_summary(self, parsed_logs: dict[str, list[LogEntry]]) -> dict[str, Any]:
        chirpstack_logs = parsed_logs.get("chirpstack", [])
        gw_logs = parsed_logs.get("chirpstack_gw", [])
        if not chirpstack_logs and not gw_logs:
            return {"available": False}
        parser = ChirpStackLogParser()
        return {
            "available": True,
            "chirpstack": parser.get_lorawan_stats(chirpstack_logs) if chirpstack_logs else {},
            "gateway_bridge": parser.get_lorawan_stats(gw_logs) if gw_logs else {},
            "total_log_entries": len(chirpstack_logs) + len(gw_logs),
        }

    def get_nvr_summary(self, parsed_logs: dict[str, list[LogEntry]]) -> dict[str, Any]:
        if "frigate" not in parsed_logs:
            return {"available": False}
        parser = FrigateLogParser()
        entries = parsed_logs["frigate"]
        return {
            "available": True,
            "detections": parser.get_detection_stats(entries),
            "log_levels": parser.count_by_level(entries),
            "total_log_entries": len(entries),
        }

    def get_time_series(
        self,
        parsed_logs: dict[str, list[LogEntry]],
        interval: str = "1h",
    ) -> "pd.DataFrame | None":
        records = [
            {"timestamp": entry.timestamp, "service": service, "level": entry.level}
            for service, entries in parsed_logs.items()
            for entry in entries
            if entry.timestamp
        ]
        if not records:
            return None
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return (
            df.groupby([pd.Grouper(freq=interval), "service", "level"]).size().unstack(fill_value=0)
        )

    def get_full_report(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        tail: int = 2000,
    ) -> dict[str, Any]:
        parsed_logs = self.collect_and_parse(since=since, until=until, tail=tail)
        return {
            "generated_at": datetime.now().isoformat(),
            "period": {
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
            },
            "infrastructure": {
                "health": self.collector.get_container_health(),
                "resources": self.collector.get_all_container_stats(),
            },
            "logs": {
                "errors": self.get_error_summary(parsed_logs),
                "mqtt": self.get_mqtt_summary(parsed_logs),
                "lorawan": self.get_lorawan_summary(parsed_logs),
                "nvr": self.get_nvr_summary(parsed_logs),
            },
            "services_analyzed": list(parsed_logs.keys()),
        }
