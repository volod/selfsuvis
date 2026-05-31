"""sencoop — standalone edge IoT sensor mesh package.

Sub-packages:
  analytics/   Log analytics for sencoop stack services (Mosquitto, ChirpStack, Frigate, OpenRemote).
  sensors/     Async ingestors for live MQTT sensor streams (LoRaWAN + Frigate NVR events).
  mesh/        Real-time multi-modal site state aggregation and fusion.

Optional extras (pip install 'sencoop[sensors]') are required for:
  - MqttSubscriber (aiomqtt)
  - LogCollector / LogAnalyzer / ReportRenderer (docker, pandas, jinja2, rich)
"""
