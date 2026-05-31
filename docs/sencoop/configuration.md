# Configuration Guide

This guide covers detailed configuration options for all Stack A Pilot services.

## Environment Variables

All configuration is managed through the `.env` file. Here's a complete reference:

### OpenRemote Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OR_HOSTNAME` | `localhost` | Primary hostname for the hub |
| `OR_SSL_PORT` | `443` | HTTPS port |
| `OR_ADDITIONAL_HOSTNAMES` | - | Additional hostnames (comma-separated) |
| `OR_EMAIL_ADMIN` | - | Admin email (used for Let's Encrypt) |
| `OR_ADMIN_PASSWORD` | `secret` | Admin account password |
| `OR_SETUP_TYPE` | - | Setup type (leave empty for default) |
| `OR_SETUP_RUN_ON_RESTART` | `false` | Re-run setup on restart |
| `OR_DEV_MODE` | `false` | Enable development mode |
| `OR_METRICS_ENABLED` | `true` | Enable Prometheus metrics |
| `OR_POSTGRES_MAX_CONNECTIONS` | `200` | PostgreSQL connection pool |

### MQTT Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MOSQUITTO_MQTTS_PORT` | `8883` | External MQTT TLS port |
| `MOSQUITTO_HEALTH_USER` | `health` | Health check username |
| `MOSQUITTO_HEALTH_PASSWORD` | - | Health check password |

### ChirpStack Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHIRPSTACK_UI_PORT` | `8080` | Web UI port |
| `CHIRPSTACK_REST_PORT` | `8090` | REST API port |
| `CHIRPSTACK_PG_USER` | `chirpstack` | PostgreSQL username |
| `CHIRPSTACK_PG_PASSWORD` | - | PostgreSQL password |
| `CHIRPSTACK_PG_DB` | `chirpstack` | Database name |
| `CHIRPSTACK_API_SECRET` | - | API secret (base64, 32 bytes) |
| `CHIRPSTACK_MQTT_USERNAME` | `chirpstack` | MQTT username |
| `CHIRPSTACK_MQTT_PASSWORD` | - | MQTT password |
| `CHIRPSTACK_GWBRIDGE_MQTT_USERNAME` | `chirpstack_gw` | Gateway bridge MQTT user |
| `CHIRPSTACK_GWBRIDGE_MQTT_PASSWORD` | - | Gateway bridge MQTT password |

### Frigate Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FRIGATE_PORT` | `8971` | Web UI port |
| `FRIGATE_MQTT_PASSWORD` | - | MQTT password |
| `FRIGATE_LIBVA_DRIVER_NAME` | - | Hardware acceleration driver |

### System Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Base path for bind-mounted volumes (user-accessible) |
| `PUID` / `PGID` | (dynamic) | Set by `compose.sh` and `bootstrap.sh` from `id -u` / `id -g`; do not add to .env |
| `TZ` | `Europe/Kyiv` | Timezone |
| `PROMETHEUS_PORT` | `9090` | Prometheus port |

## Service Configuration Files

### Mosquitto (`config/sencoop/mosquitto/`)

Static configuration (version-controlled): `mosquitto.conf`, `aclfile`.
Generated runtime secrets (written to `data/coop/mosquitto/` by `coop-bootstrap.sh`): `pwfile`, `certs/`.

**mosquitto.conf** - Main broker configuration:
```
persistence true
persistence_location /mosquitto/data/

log_dest file /mosquitto/log/mosquitto.log
log_timestamp true

listener 1883
allow_anonymous false
password_file /mosquitto/secrets/pwfile
acl_file /mosquitto/config/aclfile

listener 8883
cafile /mosquitto/secrets/certs/ca.crt
certfile /mosquitto/secrets/certs/server.crt
keyfile /mosquitto/secrets/certs/server.key
```

**aclfile** - Access control:
```
user health
topic read $SYS/#

user chirpstack
topic readwrite eu868/#
topic readwrite application/#

user frigate
topic readwrite frigate/#
```

### ChirpStack (`config/sencoop/chirpstack/`)

**chirpstack.toml** - Network server configuration:
```toml
[logging]
level = "info"

[postgresql]
dsn = "postgres://user:pass@host/db?sslmode=disable"

[redis]
servers = ["redis://chirpstack-redis/"]

[network]
net_id = "000000"
enabled_regions = ["eu868"]

[api]
bind = "0.0.0.0:8080"

[integration.mqtt]
server = "tcp://mosquitto:1883/"
username = "$CHIRPSTACK_MQTT_USERNAME"
password = "$CHIRPSTACK_MQTT_PASSWORD"
```

**region_eu868.toml** - EU868 region settings with extra channels.

### Frigate

Template (version-controlled): `config/sencoop/frigate/config.yml`, `config/sencoop/frigate/go2rtc_homekit.yml`.
Live working directory (written by Frigate at runtime): `data/coop/frigate/` — contains the active config and all runtime state (`frigate.db`, `.jwt_secret`, etc.).
`coop-bootstrap.sh` copies the templates to `data/coop/frigate/` on first run.

**config.yml** - NVR configuration:
```yaml
mqtt:
  enabled: true
  host: mosquitto
  port: 1883
  user: frigate
  password: "your-password"
  topic_prefix: frigate

auth:
  enabled: true

cameras:
  camera_name:
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://user:pass@ip:554/stream
          roles: [detect, record]
    detect:
      width: 1280
      height: 720
      fps: 5

detectors:
  cpu1:
    type: cpu
```

## Resource Limits

Default resource limits in `docker/core/docker-compose.yml`:

| Service | CPU | Memory |
|---------|-----|--------|
| Proxy | 0.5 | 256MB |
| Manager | 2.0 | 4GB |
| Keycloak | 1.0 | 1.5GB |
| PostgreSQL | 1.0 | 2GB |
| Mosquitto | 0.25 | 256MB |
| ChirpStack | 0.75 | 768MB |
| ChirpStack PostgreSQL | 0.75 | 1GB |
| Redis | 0.25 | 256MB |
| Frigate | 3.0 | 4GB |

Adjust in `docker/core/docker-compose.yml` under `deploy.resources.limits`.

## Enabling Prometheus Monitoring

Start with the monitoring profile:

```bash
docker compose --profile monitoring up -d
```

Access Prometheus at http://localhost:9090.

## TLS/SSL Configuration

### Self-Signed Certificates

Generated by `scripts/sencoop/sencoop-mosquitto-tls.sh`:
- Valid for 730 days (server cert)
- CA valid for 10 years
- 4096-bit RSA keys

### Production Certificates

For production, replace files in `data/coop/mosquitto/certs/` (created by `coop-bootstrap.sh`):
- `ca.crt` - CA certificate
- `server.crt` - Server certificate
- `server.key` - Server private key

OpenRemote proxy handles Let's Encrypt automatically when `OR_EMAIL_ADMIN` is set.
