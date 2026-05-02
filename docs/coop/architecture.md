# Architecture Overview

Stack A Pilot is a containerized IoT infrastructure platform designed to run on a single amd64 nettop.

## System Architecture

```
                                    ┌─────────────────────────────────────────────────────────┐
                                    │                    Internet/LAN                         │
                                    └───────────────────────────┬─────────────────────────────┘
                                                                │
                    ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
                    │                                           │                                           │
              ┌─────┴─────┐                              ┌──────┴──────┐                            ┌───────┴───────┐
              │ Port 443  │                              │ Port 8883   │                            │ Port 1700/UDP │
              │ HTTPS     │                              │ MQTT/TLS    │                            │ LoRaWAN       │
              └─────┬─────┘                              └──────┬──────┘                            └───────┬───────┘
                    │                                           │                                           │
                    ▼                                           ▼                                           ▼
        ┌───────────────────┐                        ┌───────────────────┐                    ┌─────────────────────────┐
        │   OpenRemote      │                        │    Mosquitto      │                    │  ChirpStack Gateway     │
        │   Proxy           │                        │    MQTT Broker    │                    │  Bridge (UDP)           │
        │   (Nginx + SSL)   │                        │                   │                    │                         │
        └─────────┬─────────┘                        └─────────┬─────────┘                    └───────────┬─────────────┘
                  │                                            │                                          │
                  ▼                                            │                                          │
        ┌───────────────────┐                                  │                                          │
        │   OpenRemote      │◄─────────────────────────────────┼──────────────────────────────────────────┤
        │   Manager         │                                  │                                          │
        │   (IoT Platform)  │                                  │                                          │
        └─────────┬─────────┘                                  ▼                                          ▼
                  │                              ┌─────────────────────────┐              ┌─────────────────────────┐
                  ▼                              │                         │              │      ChirpStack         │
        ┌───────────────────┐                    │    MQTT Message Bus     │◄────────────►│   (LoRaWAN Network      │
        │   Keycloak        │                    │    (Topics)             │              │    Server)              │
        │   (Auth/IAM)      │                    │                         │              │                         │
        └─────────┬─────────┘                    │  • eu868/#              │              └───────────┬─────────────┘
                  │                              │  • application/#        │                          │
                  ▼                              │  • frigate/#            │                          ▼
        ┌───────────────────┐                    │  • $SYS/#               │              ┌─────────────────────────┐
        │   PostgreSQL      │                    │                         │              │  ChirpStack REST API    │
        │   (OpenRemote DB) │                    └─────────────────────────┘              │                         │
        └───────────────────┘                                  ▲                          └─────────────────────────┘
                                                               │                                      │
                                                               │                                      ▼
                                               ┌───────────────┴───────────┐              ┌─────────────────────────┐
                                               │                           │              │  ChirpStack PostgreSQL  │
                                               │      Frigate NVR          │              │                         │
                                               │   (Video Surveillance)    │              └─────────────────────────┘
                                               │                           │                          │
                                               └───────────────────────────┘                          ▼
                                                               │                          ┌─────────────────────────┐
                                                               ▼                          │  ChirpStack Redis       │
                                                       RTSP Cameras                       │  (Cache)                │
                                                                                          └─────────────────────────┘
```

## Component Overview

### MQTT Infrastructure

| Component | Container | Description |
|-----------|-----------|-------------|
| Mosquitto | `coop-mosquitto` | Central MQTT message bus (Eclipse Mosquitto v2); bridges all subsystems — ChirpStack, Frigate, and OpenRemote Manager — with TLS on external port 8883 and ACL-based per-user topic access control. |

### LoRaWAN Stack (ChirpStack)

| Component | Container | Description |
|-----------|-----------|-------------|
| ChirpStack | `coop-chirpstack` | LoRaWAN Network Server v4; handles device activation (OTAA/ABP), uplink/downlink routing, and publishes decoded payloads to MQTT topic `application/#`. Configured for EU868 band. |
| Gateway Bridge | `coop-cs-gwbridge` | Translates Semtech UDP packets arriving on port 1700 from physical LoRa gateways into MQTT messages on `eu868/gateway/{id}/event/#` topics. |
| REST API | `coop-cs-rest` | HTTP/JSON façade over ChirpStack's internal gRPC API; exposes device and gateway management on localhost port 8090. |
| PostgreSQL | `coop-cs-postgres` | Dedicated Postgres 14 instance for ChirpStack; stores tenants, applications, devices, gateways, and frame logs. Initialised with the `pg_trgm` extension via `config/postgresql/initdb/`. |
| Redis | `coop-cs-redis` | In-memory cache for ChirpStack; holds session state, downlink queues, and deduplication windows. Configured with periodic RDB snapshots, no AOF. |

### Video Surveillance

| Component | Container | Description |
|-----------|-----------|-------------|
| Frigate | `coop-frigate` | Network Video Recorder with real-time object detection; ingests RTSP and USB/V4L2 camera streams, publishes detection events to MQTT topic `frigate/#`, stores recordings and clips to bind-mounted media volume. Uses host DRI device for optional hardware-accelerated decoding. CPU detector enabled by default. |

### Monitoring (Optional)

| Component | Container | Description |
|-----------|-----------|-------------|
| Prometheus | `mon-prometheus` | Time-series metrics collector; active only under the `monitoring` Docker Compose profile. Scrapes Manager metrics and stores data in a bind-mounted volume. |

### Supporting Tools

| Tool | Location | Description |
|------|----------|-------------|
| bootstrap.sh | `scripts/coop-bootstrap.sh` | One-shot startup script; creates data directories, generates Mosquitto TLS certs if missing, initialises MQTT user credentials, then launches the stack via `compose.sh`. |
| compose.sh | `scripts/coop-compose.sh` | Thin wrapper around `docker compose` that injects `PUID`/`PGID` from the current user at runtime, overriding any values in `.env`. |
| ensure_data_dirs.sh | `scripts/coop-data-dirs.sh` | Creates all bind-mount directories under `$DATA_DIR` with correct ownership before first stack start. |
| gen_mosquitto_selfsigned_tls.sh | `scripts/coop-mosquitto-tls.sh` | Generates a self-signed CA and server TLS certificate for Mosquitto. |
| init_mosquitto_users.sh | `scripts/coop-mqtt-users.sh` | Creates and hashes MQTT user credentials in Mosquitto's password file from `.env` values. |
| add_camera.sh | `scripts/coop-camera.sh` | Shell entrypoint that invokes the packaged camera CLI to add RTSP or USB/V4L2 cameras to `config/coop/frigate/config.yml` and optionally restart the Frigate container. |
| clean_data.sh | `scripts/coop-clean-data.sh` | Removes all data directories under `$DATA_DIR`; destructive — intended for development resets only. |
| coop_stack_analytics | `coop_stack_analytics/` | Python package for log collection, service-specific parsing (Mosquitto, ChirpStack, Frigate, OpenRemote), statistics aggregation, and report generation in console, JSON, HTML, or Markdown formats. Invoked via `python -m coop_stack_analytics.cli`. |
| tests | `tests/` | Pytest integration tests (container health checks, MQTT connectivity) and a Locust load-test file for stress-testing the stack. |

## Network Architecture

All containers run on a single Docker bridge network: `coop-stack-a-net`

### External Ports

| Port | Protocol | Service | Description |
|------|----------|---------|-------------|
| 80 | TCP | Proxy | HTTP (redirects to HTTPS) |
| 443 | TCP | Proxy | HTTPS |
| 8883 | TCP | Mosquitto | MQTT over TLS |
| 1700 | UDP | Gateway Bridge | LoRaWAN gateway packets |

### Internal Ports (localhost only)

| Port | Service | Description |
|------|---------|-------------|
| 8080 | ChirpStack | Web UI |
| 8090 | ChirpStack REST | REST API |
| 8405 | Manager | Management API |
| 8971 | Frigate | NVR interface |
| 9090 | Prometheus | Monitoring |

## Data Flow

### LoRaWAN Data Path

```
LoRa Device → Gateway → UDP:1700 → Gateway Bridge → MQTT(eu868/#) → ChirpStack → MQTT(application/#) → Application
```

### Video Surveillance Path

```
RTSP Camera → Frigate → Object Detection → MQTT(frigate/#) → Consumers
```

### IoT Management Path

```
Web Browser → HTTPS:443 → Proxy → Manager → Keycloak (auth) → PostgreSQL
```

## Security Model

### Authentication Layers

1. **Web Access**: Keycloak OAuth2/OIDC
2. **MQTT Access**: Username/password with ACLs
3. **LoRaWAN**: Network key encryption
4. **Databases**: Internal network only

### MQTT Access Control

| User | Allowed Topics | Permissions |
|------|---------------|-------------|
| health | `$SYS/#` | Read |
| chirpstack | `eu868/#`, `application/#`, `device/#` | Read/Write |
| chirpstack_gw | `eu868/gateway/#` | Read/Write |
| frigate | `frigate/#` | Read/Write |

### TLS Configuration

- MQTT: Self-signed or custom CA on port 8883
- HTTPS: Let's Encrypt or self-signed on port 443

## Resource Allocation

### Memory Budget (8GB system)

| Component | Allocation |
|-----------|------------|
| Manager | 4 GB |
| Frigate | 4 GB |
| PostgreSQL | 2 GB |
| Keycloak | 1.5 GB |
| ChirpStack | 768 MB |
| ChirpStack PostgreSQL | 1 GB |
| Others | ~500 MB each |

### CPU Budget (4 cores)

| Component | Allocation |
|-----------|------------|
| Frigate | 3.0 cores |
| Manager | 2.0 cores |
| Others | 0.25-1.0 cores |

## Persistence

### Data Directories (Bind Mounts)

All persistent data uses bind mounts under `$DATA_DIR` (default: `./data`), so the user running `docker compose` owns the location. Run `scripts/coop-data-dirs.sh` before first start.

| Path | Purpose | Data |
|------|---------|------|
| `$DATA_DIR/postgresql` | OpenRemote DB | User data, assets |
| `$DATA_DIR/manager` | Manager storage | Files, configs |
| `$DATA_DIR/chirpstack-postgres` | ChirpStack DB | Devices, gateways |
| `$DATA_DIR/mosquitto/data` | MQTT persistence | Retained messages |
| `$DATA_DIR/mosquitto/log` | MQTT logs | Broker logs |
| `$DATA_DIR/chirpstack-redis` | Redis | ChirpStack cache |
| `$DATA_DIR/frigate-media` | Video storage | Recordings, clips |
| `$DATA_DIR/prometheus` | Metrics | Time series data |
| `$DATA_DIR/proxy` | Proxy deployment | TLS, config |

### Backup Strategy

Critical data to backup:
1. PostgreSQL databases (both)
2. Manager data volume
3. Configuration files
4. TLS certificates

## Scalability Considerations

This pilot deployment is designed for:
- 3-5 installation sites (strongholds)
- 5-10 LoRaWAN devices
- 2-6 RTSP cameras

For larger deployments:
- Separate database servers
- Multiple ChirpStack instances
- Distributed Frigate with dedicated GPU
- Kubernetes orchestration
