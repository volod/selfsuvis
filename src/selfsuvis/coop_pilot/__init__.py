"""coop_pilot — edge IoT sensor mesh integration for selfsuvis.

Sub-packages:
  analytics/   Log analytics for coop-pilot stack services (Mosquitto, ChirpStack, Frigate, OpenRemote).
  sensors/     Async ingestors for live MQTT sensor streams (LoRaWAN + Frigate NVR events).
  mesh/        Real-time multi-modal site state aggregation and fusion.

Optional extras (pip install 'selfsuvis[coop_pilot]') are required for:
  - MqttSubscriber (aiomqtt)
  - LogCollector / LogAnalyzer / ReportRenderer (docker, pandas, jinja2, rich)
"""
