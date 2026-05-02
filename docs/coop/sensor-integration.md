# Sensor Integration Guide

This guide explains how to integrate sensors and cameras with Stack A Pilot services:
OpenRemote, Mosquitto (MQTT), ChirpStack (LoRaWAN), and Frigate (NVR).

## Overview

| Sensor Type | Primary Service | Data Path |
|-------------|-----------------|-----------|
| LoRaWAN devices | ChirpStack | Device -> Gateway -> MQTT(eu868/#) -> ChirpStack -> MQTT(application/#) |
| IP/RTSP cameras | Frigate | RTSP -> Frigate -> MQTT(frigate/#) |
| USB cameras | Frigate | /dev/video* -> Frigate -> MQTT(frigate/#) |
| MQTT sensors | OpenRemote | Direct MQTT publish -> OpenRemote Manager |
| Generic MQTT | Mosquitto | Publish to any topic (requires ACL) |

---

## 1. LoRaWAN Sensors (ChirpStack)

### Prerequisites

- LoRaWAN gateway connected to the network
- Gateway configured to forward packets to host UDP port 1700

### Integration Steps

1. **Access ChirpStack UI**: http://localhost:8080

2. **Create an Application**:
   - Applications -> Add
   - Name: e.g. "Pilot Sensors"
   - Description: optional

3. **Create a Device Profile**:
   - Device Profiles -> Add
   - Name: e.g. "Class A Sensor"
   - Region: EU868
   - LoRaWAN MAC version: 1.0.3 or 1.1
   - LoRaWAN Regional Parameters: EU868
   - Payload codec: None (or custom for decoding)

4. **Register the Device**:
   - Devices -> Add
   - Application: select your application
   - Device profile: select your profile
   - DevEUI: 16 hex chars (e.g. from device label)
   - Device name: descriptive name

5. **Activate the Device** (OTAA):
   - Click device -> Activation
   - Add device (ABP) or enable OTAA
   - For OTAA: AppEUI, AppKey from your LoRaWAN provider/join server

6. **Data Flow**:
   - Uplinks appear in ChirpStack UI
   - Payloads published to MQTT: `application/<app_id>/device/<dev_eui>/event/up`
   - OpenRemote can subscribe to these topics for dashboards

### ChirpStack REST API

For automation, use the REST API at http://localhost:8090:

```bash
# List applications (requires CHIRPSTACK_API_SECRET)
curl -H "Grpc-Metadata-Authorization: Bearer $CHIRPSTACK_API_SECRET" \
  http://localhost:8090/api/applications

# List devices
curl -H "Grpc-Metadata-Authorization: Bearer $CHIRPSTACK_API_SECRET" \
  http://localhost:8090/api/devices
```

---

## 2. IP/RTSP Cameras (Frigate)

### Prerequisites

- Camera on same network or reachable from host
- RTSP URL (e.g. `rtsp://user:pass@192.168.1.100:554/stream1`)

### Integration Steps

1. **Edit Frigate config**: `config/coop/frigate/config.yml`

2. **Add camera**:

```yaml
cameras:
  front_door:
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://USER:PASS@192.168.1.100:554/stream1
          roles:
            - detect
            - record
    detect:
      width: 1280
      height: 720
      fps: 5
```

3. **Restart Frigate**:
   ```bash
   docker compose restart frigate
   ```

4. **Verify**: Open http://localhost:8971

5. **MQTT events**: Frigate publishes to `frigate/#`:
   - `frigate/events` - detection events
   - `frigate/<camera>/person` - person detections per camera

---

## 3. USB Cameras (Frigate)

### Prerequisites

- USB camera connected to the host
- Run `./scripts/coop-test-usb-cameras.sh` to verify detection

### Step 1: Test USB Camera

```bash
./scripts/coop-test-usb-cameras.sh
```

This lists V4L2 devices and tests basic capture. Note the device path (e.g. `/dev/video0`).

### Step 2: Pass Device to Frigate

Edit `docker/docker-compose.yml`, add USB camera to Frigate service:

```yaml
  frigate:
    image: ghcr.io/blakeblackshear/frigate:stable
    # ... existing config ...
    devices:
      - /dev/dri:/dev/dri
      # Add USB camera (use by-id for stable device naming)
      - /dev/v4l/by-id/usb-<VENDOR>_<PRODUCT>-video-index0:/dev/video0
    # Or use numeric device if by-id not available:
    # - /dev/video0:/dev/video0
```

To find the stable by-id path:
```bash
ls -la /dev/v4l/by-id/
```

### Step 3: Configure Frigate for USB

Edit `config/coop/frigate/config.yml`:

```yaml
cameras:
  usb_camera:
    enabled: true
    ffmpeg:
      inputs:
        - path: /dev/video0
          input_args: -f v4l2
          roles:
            - detect
            - record
    detect:
      width: 640
      height: 480
      fps: 5
```

For rotated image, add per-input `output_args`:
```yaml
        - path: /dev/video0
          input_args: -f v4l2
          roles:
            - detect
            - record
          output_args:
            detect: -vf transpose=1 -f rawvideo -pix_fmt yuv420p
            record: -vf transpose=1 -c:v libx264 -preset ultrafast -tune zerolatency -f segment -segment_time 10 -segment_format mp4 -reset_timestamps 1 -strftime 1 -an
```

### Step 4: Restart Stack

```bash
docker compose --compatibility up -d
```

---

## 4. MQTT Sensors (OpenRemote)

### Prerequisites

- MQTT client credentials (create user in Mosquitto if needed)
- TLS certs for port 8883, or use internal port 1883 from host

### Integration Steps

1. **Add MQTT user** (if new sensor needs its own user):
   - Edit `config/coop/mosquitto/pwfile` or use `mosquitto_passwd`
   - Add ACL rules in `config/coop/mosquitto/aclfile` for allowed topics

2. **Publish from sensor**:
   - Connect to `localhost:8883` (TLS) or `localhost:1883` (internal)
   - Publish to topic e.g. `sensors/temperature/room1`
   - Payload: JSON recommended, e.g. `{"value": 22.5, "unit": "C"}`

3. **Create asset in OpenRemote**:
   - Log in to https://localhost
   - Create asset type "Temperature Sensor"
   - Add attribute linked to MQTT topic
   - Create dashboard to visualize

4. **Example ACL** (add to `config/coop/mosquitto/aclfile`):
   ```
   user sensor_client
   topic write sensors/#
   topic read sensors/#
   ```

---

## 5. Frigate MQTT Integration

Frigate publishes detection events to MQTT. To consume in OpenRemote or other apps:

| Topic | Description |
|-------|-------------|
| `frigate/events` | All detection events (JSON) |
| `frigate/<camera>/<object>/snapshot` | Snapshot image on detection |
| `frigate/available` | Frigate online/offline |

Example event payload:
```json
{
  "type": "end",
  "before": {...},
  "after": {
    "camera": "front_door",
    "label": "person",
    "score": 0.92,
    "start_time": 1234567890.123
  }
}
```

---

## 6. Dynamic Camera Attachment

Use the `add_camera` script to add cameras without manual config editing:

```bash
# Add RTSP camera
./scripts/coop-camera.sh --name front_door --rtsp rtsp://user:pass@ip:554/stream1 --restart

# Add USB camera (pass /dev/video0 to Frigate in docker-compose first)
./scripts/coop-camera.sh --name usb_cam --usb /dev/video0 --restart

# List cameras
./scripts/coop-camera.sh --list
```

Note: Frigate requires a restart to pick up new cameras. Use `--restart` to restart automatically.

---

## 7. Quick Reference

### Ports

| Port | Service | Use |
|------|---------|-----|
| 8883 | Mosquitto | MQTT over TLS (external sensors) |
| 1883 | Mosquitto | MQTT internal (containers only) |
| 1700/UDP | Gateway Bridge | LoRaWAN gateway packets |
| 554 | RTSP | Camera streams (if camera exposes) |

### MQTT Topics

| Topic | Publisher | Subscriber |
|-------|-----------|------------|
| `eu868/#` | Gateway Bridge | ChirpStack |
| `application/#` | ChirpStack | Applications |
| `frigate/#` | Frigate | OpenRemote, dashboards |
| `$SYS/#` | Mosquitto | health user (read-only) |
| `sensors/#` | Custom | OpenRemote (add ACL) |

### File Locations

| File | Purpose |
|------|---------|
| `config/coop/frigate/config.yml` | Camera config |
| `config/coop/mosquitto/aclfile` | MQTT ACL |
| `config/coop/mosquitto/pwfile` | MQTT passwords |
| `config/coop/chirpstack/` | ChirpStack config |
| `.env` | Passwords, ports |
