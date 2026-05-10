"""Report rendering in console, JSON, HTML, and Markdown formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analyzer import LogAnalyzer

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class ReportRenderer:
    """Renders coop-pilot analytics reports in multiple formats."""

    def __init__(self, analyzer: LogAnalyzer | None = None) -> None:
        self.analyzer = analyzer or LogAnalyzer()
        self.console = Console()
        self.jinja_env = Environment(
            loader=FileSystemLoader(_TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # -- Report generation -----------------------------------------------------

    def generate_report(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        tail: int = 2000,
    ) -> dict[str, Any]:
        return self.analyzer.get_full_report(since=since, until=until, tail=tail)

    def get_health(self) -> dict[str, Any]:
        return self.analyzer.collector.get_container_health()

    def get_stats(self) -> dict[str, Any]:
        return self.analyzer.collector.get_all_container_stats()

    # -- Console output --------------------------------------------------------

    def print_health_only(self) -> None:
        self._print_health_table(self.get_health())

    def print_stats_only(self) -> None:
        self._print_resource_table(self.get_stats())

    def print_console_report(self, report: dict[str, Any]) -> None:
        self._print_header(report)
        self._print_health_table(report.get("infrastructure", {}).get("health", {}))
        self._print_resource_table(report.get("infrastructure", {}).get("resources", {}))
        self._print_error_summary(report.get("logs", {}).get("errors", {}))
        self._print_mqtt_summary(report.get("logs", {}).get("mqtt", {}))
        self._print_lorawan_summary(report.get("logs", {}).get("lorawan", {}))
        self._print_nvr_summary(report.get("logs", {}).get("nvr", {}))

    def _print_header(self, report: dict[str, Any]) -> None:
        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold blue]coop-pilot Stack Analytics Report[/bold blue]", border_style="blue"
            )
        )
        self.console.print(f"Generated: {report['generated_at']}")
        self.console.print()

    def _print_health_table(self, health: dict[str, Any]) -> None:
        if not health:
            return
        table = Table(title="Container Health", show_header=True)
        table.add_column("Service", style="cyan")
        table.add_column("Container", style="white")
        table.add_column("Status", style="green")
        table.add_column("Health", style="yellow")
        table.add_column("Restarts", style="red")
        for service, info in sorted(health.items()):
            table.add_row(
                service,
                info["name"],
                Text(info["status"], style="green" if info["status"] == "running" else "red"),
                Text(info["health"], style="green" if info["health"] == "healthy" else "yellow"),
                str(info["restart_count"]),
            )
        self.console.print(table)
        self.console.print()

    def _print_resource_table(self, resources: dict[str, Any]) -> None:
        if not resources:
            return
        table = Table(title="Resource Usage", show_header=True)
        table.add_column("Service", style="cyan")
        table.add_column("CPU %", justify="right")
        table.add_column("Memory MB", justify="right")
        table.add_column("Memory %", justify="right")
        for service, stats in sorted(resources.items()):
            table.add_row(
                service,
                Text(
                    f"{stats['cpu_percent']:.1f}%",
                    style="red" if stats["cpu_percent"] > 80 else "green",
                ),
                f"{stats['memory_usage_mb']:.1f}",
                Text(
                    f"{stats['memory_percent']:.1f}%",
                    style="red" if stats["memory_percent"] > 80 else "green",
                ),
            )
        self.console.print(table)
        self.console.print()

    def _print_error_summary(self, errors: dict[str, Any]) -> None:
        self.console.print(Panel.fit("[bold red]Error Summary[/bold red]", border_style="red"))
        self.console.print(f"Total Errors: [red]{errors.get('total_errors', 0)}[/red]")
        self.console.print(f"Total Warnings: [yellow]{errors.get('total_warnings', 0)}[/yellow]")
        self.console.print()
        if errors.get("by_service"):
            table = Table(show_header=True)
            table.add_column("Service", style="cyan")
            table.add_column("Errors", style="red", justify="right")
            table.add_column("Warnings", style="yellow", justify="right")
            for service, counts in sorted(errors["by_service"].items()):
                if counts["errors"] > 0 or counts["warnings"] > 0:
                    table.add_row(service, str(counts["errors"]), str(counts["warnings"]))
            self.console.print(table)
        if errors.get("recent_errors"):
            self.console.print("\n[bold]Recent Errors:[/bold]")
            for err in errors["recent_errors"][:5]:
                self.console.print(
                    f"  [{err['service']}] {err.get('timestamp', 'unknown')}: {err['message'][:80]}..."
                )
        self.console.print()

    def _print_mqtt_summary(self, mqtt: dict[str, Any]) -> None:
        if not mqtt.get("available"):
            return
        self.console.print(Panel.fit("[bold green]MQTT Summary[/bold green]", border_style="green"))
        conn = mqtt.get("connections", {})
        self.console.print(f"Total Connections: {conn.get('total_connects', 0)}")
        self.console.print(f"Total Disconnections: {conn.get('total_disconnects', 0)}")
        self.console.print(f"Unique Clients: {conn.get('unique_clients', 0)}")
        self.console.print(f"Auth Failures: [red]{conn.get('auth_failures', 0)}[/red]")
        self.console.print()

    def _print_lorawan_summary(self, lorawan: dict[str, Any]) -> None:
        if not lorawan.get("available"):
            return
        self.console.print(
            Panel.fit("[bold magenta]LoRaWAN Summary[/bold magenta]", border_style="magenta")
        )
        cs = lorawan.get("chirpstack", {})
        self.console.print(f"  Uplinks: {cs.get('uplinks', 0)}")
        self.console.print(f"  Downlinks: {cs.get('downlinks', 0)}")
        self.console.print(f"  Unique Devices: {cs.get('unique_devices', 0)}")
        self.console.print(f"  Unique Gateways: {cs.get('unique_gateways', 0)}")
        self.console.print()

    def _print_nvr_summary(self, nvr: dict[str, Any]) -> None:
        if not nvr.get("available"):
            return
        self.console.print(Panel.fit("[bold cyan]NVR Summary[/bold cyan]", border_style="cyan"))
        det = nvr.get("detections", {})
        self.console.print(f"Total Detections: {det.get('total_detections', 0)}")
        self.console.print(f"Motion Events: {det.get('motion_events', 0)}")
        if objects := det.get("objects_detected", {}):
            self.console.print("\n[bold]Objects Detected:[/bold]")
            for obj, count in sorted(objects.items(), key=lambda x: -x[1]):
                self.console.print(f"  {obj}: {count}")
        if cameras := det.get("cameras_active", []):
            self.console.print(f"\nActive Cameras: {', '.join(cameras)}")
        self.console.print()

    # -- File export -----------------------------------------------------------

    def export_json(self, report: dict[str, Any], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )

    def export_html(self, report: dict[str, Any], output_path: Path) -> None:
        self._render_and_write("report.html.j2", report, output_path)

    def export_markdown(self, report: dict[str, Any], output_path: Path) -> None:
        self._render_and_write("report.md.j2", report, output_path)

    def _render_and_write(
        self, template_name: str, report: dict[str, Any], output_path: Path
    ) -> None:
        content = self.jinja_env.get_template(template_name).render(report=report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
