"""One-process realtime aggregator boundary for sector/global threat updates."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .degraded_mode import apply_degraded_mode_to_threat, evaluate_degraded_mode
from .freshness import downweight_score, expire_event


class RealtimeThreatAggregator:
    """Consume replayed or live event envelopes and emit operator snapshots."""

    def __init__(self) -> None:
        self._sensor_events: List[Dict[str, Any]] = []
        self._threat_events: List[Dict[str, Any]] = []
        self._node_health_events: List[Dict[str, Any]] = []

    def consume(self, event: Dict[str, Any]) -> None:
        kind = str(event.get("event_kind", "")).strip().lower()
        if kind == "sensor":
            self._sensor_events.append(dict(event))
        elif kind == "threat":
            self._threat_events.append(dict(event))
        elif kind == "node_health":
            self._node_health_events.append(dict(event))
        else:
            raise ValueError(f"unsupported event_kind: {kind or '<empty>'}")

    def consume_all(self, events: Iterable[Dict[str, Any]]) -> None:
        for event in events:
            self.consume(event)

    def snapshot(self) -> Dict[str, Any]:
        sector_rows = self._sector_rows()
        health = evaluate_degraded_mode(
            self._sensor_events,
            self._node_health_events,
            base_automation_confidence=self._base_automation_confidence(),
        )
        route_rows = self._route_rows(sector_rows, automation_confidence=float(health["automation_confidence"]))
        return {
            "global_threat_map": [
                {
                    "sector_id": row["sector_id"],
                    "threat_score": row["threat_score"],
                    "risk_level": row["risk_level"],
                }
                for row in sector_rows
            ],
            "sector_risk_levels": sector_rows,
            "persistent_anomalies": self._persistent_anomalies(),
            "route_advisories": route_rows,
            "threat_corridor_graph": self._corridor_graph(),
            "automation_confidence": health["automation_confidence"],
            "degraded": health["degraded"],
            "health_warnings": health["health_warnings"],
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

    def write_snapshot(self, output_path: Path) -> Dict[str, Any]:
        snapshot = self.snapshot()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return snapshot

    def _sector_rows(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in self._threat_events:
            if expire_event(event, hard_expiry_sec=120.0):
                continue
            grouped[str(event.get("sector_id", "unknown"))].append(event)

        rows: List[Dict[str, Any]] = []
        for sector_id, events in sorted(grouped.items()):
            score_terms: List[float] = []
            nodes: List[str] = []
            route_ids: List[str] = []
            primitive_types: List[str] = []
            for event in events:
                payload = dict(event.get("payload") or {})
                if str(event.get("sensor_type", "")) == "local_threat":
                    score = float(payload.get("local_threat_score", 0.0) or 0.0)
                else:
                    score = float(payload.get("score", 0.0) or 0.0)
                score_terms.append(downweight_score(score, float(event.get("freshness_sec", 0.0) or 0.0)))
                if event.get("node_id") not in nodes:
                    nodes.append(str(event.get("node_id")))
                route_id = str(payload.get("route_id", "") or "")
                if route_id and route_id not in route_ids:
                    route_ids.append(route_id)
                threat_type = str(payload.get("threat_type", event.get("sensor_type", "")) or "")
                if threat_type and threat_type not in primitive_types:
                    primitive_types.append(threat_type)

            remaining = 1.0
            for score in score_terms:
                remaining *= (1.0 - max(0.0, min(1.0, score)))
            aggregated = 1.0 - remaining if score_terms else 0.0
            support_bonus = min(0.15, 0.05 * max(0, len(nodes) - 1))
            threat_score = min(1.0, aggregated + support_bonus)
            rows.append(
                {
                    "sector_id": sector_id,
                    "threat_score": round(threat_score, 4),
                    "risk_level": _risk_level(threat_score),
                    "supporting_nodes": nodes,
                    "route_ids": route_ids,
                    "primitive_types": primitive_types,
                    "observation_count": len(events),
                }
            )
        return rows

    def _route_rows(self, sector_rows: Sequence[Dict[str, Any]], *, automation_confidence: float) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in sector_rows:
            for route_id in row.get("route_ids") or []:
                grouped[str(route_id)].append(row)
        out: List[Dict[str, Any]] = []
        for route_id, rows in sorted(grouped.items()):
            max_score = max(float(row.get("threat_score", 0.0) or 0.0) for row in rows)
            health = apply_degraded_mode_to_threat(
                max_score,
                automation_confidence,
            )
            if max_score >= 0.75:
                action = "abort"
            elif health["automation_confidence"] < 0.50:
                action = "inspect_sensor"
            elif max_score >= 0.60:
                action = "reroute"
            elif max_score >= 0.35:
                action = "reduce_speed"
            else:
                action = "continue"
            out.append(
                {
                    "route_id": route_id,
                    "advisory_score": round(max_score, 4),
                    "recommended_action": action,
                    "sector_ids": [str(row.get("sector_id", "unknown")) for row in rows],
                    "automation_confidence": health["automation_confidence"],
                }
            )
        return out

    def _persistent_anomalies(self) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for event in self._threat_events:
            if str(event.get("sensor_type", "")) == "local_threat":
                continue
            payload = dict(event.get("payload") or {})
            threat_type = str(payload.get("threat_type", event.get("sensor_type", "")) or "")
            grouped[(threat_type, str(event.get("sector_id", "unknown")))].append(event)
        rows: List[Dict[str, Any]] = []
        for (threat_type, sector_id), events in sorted(grouped.items()):
            if len(events) < 2:
                continue
            rows.append(
                {
                    "anomaly_id": f"{threat_type}:{sector_id}",
                    "anomaly_type": threat_type,
                    "sector_ids": [sector_id],
                    "persistence_count": len(events),
                    "supporting_nodes": sorted({str(event.get('node_id', 'unknown')) for event in events}),
                }
            )
        return rows

    def _corridor_graph(self) -> List[Dict[str, Any]]:
        edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        route_sequences: Dict[str, List[str]] = defaultdict(list)
        for row in self._sector_rows():
            for route_id in row.get("route_ids") or []:
                route_sequences[route_id].append(str(row.get("sector_id", "unknown")))
        for sectors in route_sequences.values():
            for left, right in zip(sectors, sectors[1:]):
                if not left or not right or left == right:
                    continue
                edge_weights[(left, right)] += 1.0
        return [
            {"from_sector": left, "to_sector": right, "weight": weight}
            for (left, right), weight in sorted(edge_weights.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        ]

    def _base_automation_confidence(self) -> float:
        if not self._node_health_events:
            return 1.0
        return max(
            0.0,
            min(
                1.0,
                sum(float((event.get("payload") or {}).get("automation_confidence", 1.0) or 1.0) for event in self._node_health_events)
                / len(self._node_health_events),
            ),
        )


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.25:
        return "low"
    return "none"
