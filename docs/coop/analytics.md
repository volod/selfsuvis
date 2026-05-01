# Analytics Guide

This guide covers the log analytics module for monitoring and reporting on Stack A Pilot.

## Overview

The `coop_stack_analytics` module provides:

- Log collection from Docker containers
- Service-specific log parsing
- Statistics aggregation
- Report generation (console, JSON, HTML, Markdown)

## Quick Start

```bash
source .venv/bin/activate

# View console report
python -m coop_stack_analytics.cli

# View last hour's activity
python -m coop_stack_analytics.cli --since 1h

# Export JSON report
python -m coop_stack_analytics.cli --format json --output reports/report.json
```

## CLI Reference

```
usage: python -m coop_stack_analytics.cli [options]

Options:
  --since TEXT      Start time (e.g., '2024-01-01 00:00' or '1h', '30m', '7d')
  --until TEXT      End time
  --tail INT        Log lines per service (default: 2000)
  --format FORMAT   Output: console, json, html, markdown
  --output PATH     Output file path
  --health-only     Only show container health
  --stats-only      Only show resource statistics
```

### Time Shortcuts

| Format | Meaning |
|--------|---------|
| `1h` | 1 hour ago |
| `30m` | 30 minutes ago |
| `7d` | 7 days ago |
| `2024-01-01` | Specific date |
| `2024-01-01 14:30` | Specific datetime |

## Report Contents

### Infrastructure Health

Shows status of all containers:

```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Service     ┃ Container     ┃ Status  ┃ Health  ┃ Restarts ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ mosquitto   │ coop-mosquitto│ running │ healthy │ 0        │
│ chirpstack  │ coop-chirpstack │ running │ none    │ 0        │
│ frigate     │ coop-frigate   │ running │ healthy │ 0        │
└─────────────┴───────────────┴─────────┴─────────┴──────────┘
```

### Resource Usage

CPU and memory for each service:

```
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Service     ┃ CPU % ┃ Memory MB ┃ Memory % ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ manager     │  0.5% │     686.3 │    16.8% │
│ frigate     │  0.1% │     551.8 │    13.5% │
│ keycloak    │  0.0% │     312.9 │    20.4% │
└─────────────┴───────┴───────────┴──────────┘
```

### Error Summary

Aggregated errors and warnings:

```
Total Errors: 3
Total Warnings: 12

┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Service   ┃ Errors ┃ Warnings ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ frigate   │ 2      │ 5        │
│ chirpstack│ 1      │ 7        │
└───────────┴────────┴──────────┘
```

### MQTT Summary

Connection and authentication statistics:

- Total connections/disconnections
- Unique clients
- Authentication failures

### LoRaWAN Summary

ChirpStack activity:

- Uplink/downlink message counts
- Unique devices and gateways
- Error counts

### NVR Summary

Frigate detection statistics:

- Total detections
- Objects detected (person, car, etc.)
- Motion events
- Active cameras

## Programmatic Usage

### Basic Usage

```python
from coop_stack_analytics import LogAnalyzer, ReportRenderer

# Generate report
aggregator = LogAnalyzer()
report = aggregator.get_full_report(tail=1000)

# Print to console
reporter = ReportRenderer(aggregator)
reporter.print_console_report(report)
```

### Collect Logs

```python
from coop_stack_analytics import LogCollector

collector = LogCollector()

# Get logs from specific service
logs = collector.get_container_logs("mosquitto", tail=500)

# Get all service logs
all_logs = collector.get_all_service_logs(tail=1000)

# Get container stats
stats = collector.get_container_stats("manager")
```

### Parse Logs

```python
from coop_stack_analytics import MosquittoLogParser, ChirpStackLogParser

# Parse Mosquitto logs
parser = MosquittoLogParser()
entries = list(parser.parse_lines(logs))

# Get connection statistics
conn_stats = parser.get_connection_stats(entries)
print(f"Unique clients: {conn_stats['unique_clients']}")
```

### Export Reports

```python
from pathlib import Path
from coop_stack_analytics import ReportRenderer

reporter = ReportRenderer()
report = reporter.generate_report(tail=2000)

# Export to different formats
reporter.export_json(report, Path("report.json"))
reporter.export_html(report, Path("report.html"))
reporter.export_markdown(report, Path("report.md"))
```

## Custom Parsers

Create parsers for additional log formats:

```python
from coop_stack_analytics.parsers import BaseLogParser, LogEntry

class MyServiceParser(BaseLogParser):
    def parse_line(self, line: str) -> LogEntry:
        # Parse your log format
        return LogEntry(
            timestamp=parsed_timestamp,
            level="INFO",
            message=parsed_message,
            raw=line,
            metadata={"custom": "data"}
        )
```

## Scheduling Reports

### Cron Job

```bash
# Generate daily report at midnight
0 0 * * * cd /path/to/stack && .venv/bin/python -m coop_stack_analytics.cli \
  --since 24h --format html --output /var/www/reports/daily-$(date +\%Y\%m\%d).html
```

### Systemd Timer

Create `/etc/systemd/system/stack-report.service`:
```ini
[Unit]
Description=Stack Analytics Report

[Service]
Type=oneshot
WorkingDirectory=/path/to/stack
ExecStart=/path/to/stack/.venv/bin/python -m coop_stack_analytics.cli --since 1h --format json --output /var/log/stack/hourly.json
```

Create `/etc/systemd/system/stack-report.timer`:
```ini
[Unit]
Description=Run Stack Report Hourly

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

Enable: `systemctl enable --now stack-report.timer`

## Integration with Monitoring

### Prometheus Metrics

Export stats as Prometheus metrics:

```python
from prometheus_client import Gauge, start_http_server
from coop_stack_analytics import LogCollector

collector = LogCollector()

# Define metrics
container_restarts = Gauge('stack_container_restarts', 'Container restart count', ['service'])
container_memory = Gauge('stack_container_memory_bytes', 'Container memory usage', ['service'])

def update_metrics():
    for service, stats in collector.get_all_container_stats().items():
        container_memory.labels(service=service).set(stats['memory_usage_mb'] * 1024 * 1024)

    for service, health in collector.get_container_health().items():
        container_restarts.labels(service=service).set(health['restart_count'])

# Start metrics server on port 8000
start_http_server(8000)
```
