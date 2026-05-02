"""Shared helpers for telemetry-to-packet media bridges."""

from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional


def build_packet(
    *,
    sensor_type: str,
    t_device: float,
    payload: Dict[str, Any],
    seq: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "sensor_type": str(sensor_type).strip().lower(),
        "t_device": float(t_device),
        "seq": int(seq) if seq is not None else None,
        "payload": dict(payload or {}),
    }


def flatten_packet_batches(
    messages: Iterable[Dict[str, Any]],
    converter: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    for message in messages:
        packets.extend(converter(message))
    return packets


class PacketBridge:
    """Shared async bridge that converts external messages into realtime packets."""

    def __init__(
        self,
        converter: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        *,
        on_packet: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        self._converter = converter
        self._on_packet = on_packet

    async def ingest_message(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        packets = self._converter(message)
        if self._on_packet is not None:
            for packet in packets:
                await self._on_packet(packet)
        return packets

    async def ingest_messages(self, messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        for message in messages:
            collected.extend(await self.ingest_message(message))
        return collected
