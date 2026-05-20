# Scripts

Project scripts are organized under `scripts/` subdirectories. The `scripts/` root is intentionally kept minimal.

## Canonical commands

- `install/install_system_deps.sh` — install host system packages for local development
- `install/install_requirements.sh` — install Python dependencies into an existing virtualenv
- `install/install_nvidia_docker.sh` — install NVIDIA Container Toolkit for Docker
- `ssv/ssv-setup.sh` — end-to-end bootstrap for the main selfsuvis local environment
- `ssv/ssv-prepare-sensor-data.sh` — fetch or scaffold public sensor sample data
- `ssv/ssv-reset-qdrant.sh` — delete the configured Qdrant collection
- `ssv/ssv-realtime-bridge.sh` — run the packaged MAVSDK or ROS realtime telemetry bridge runtime
- `ssv/ssv-utilyze.sh` — run Utilyze with project defaults (if installed)

## Coop stack commands

- `coop/coop-ctl.sh` — control the coop IoT edge stack (start, stop, restart, status, logs); installed as `/usr/local/bin/coop-ctl`
- `coop/coop-install.sh` — install the coop stack on a target system from a release bundle; also the entry point (`install.sh`) inside a built bundle
- `coop/coop-release.sh` — build a self-contained offline bundle (Docker images + configs + scripts) for deployment to air-gapped targets
- `coop/coop-bootstrap.sh` — one-shot coop stack bootstrap and startup
- `coop/coop-compose.sh` — canonical coop `docker compose` wrapper with runtime `PUID` and `PGID`
- `coop/coop-env.sh` — generate `data/.env` from the coop env templates
- `coop/coop-credentials.sh` — print coop service URLs and generated credentials
- `coop/coop-data-dirs.sh` — create coop bind-mount directories under `$DATA_DIR`
- `coop/coop-mqtt-users.sh` — build the Mosquitto password file from `.env`
- `coop/coop-mosquitto-tls.sh` — generate self-signed Mosquitto TLS certs
- `coop/coop-clean-data.sh` — destructive reset of coop bind-mounted runtime data
- `coop/coop-test-usb-cameras.sh` — inspect V4L2 devices and optionally test capture
- `coop/coop-camera.sh` — update the Frigate camera config and optionally restart Frigate
- `coop/add_sensor_key.sh` — provision a sensor API key for ingestion

## Audio and edge inference

- `audio/drone_audio_edge_test.sh` — run DroneAudioCNN ONNX inference on a WAV file; supports `--scan` for full detection-range table and `--distance M` for drau physics simulation

## Project utility

- `project/project-package.sh` — create a tarball of the current repo while excluding generated secrets and runtime data
- `project/seed_test_events.sh` — seed test zones and events (development helper)

## Shared shell helpers

Shell entrypoints reuse `common.sh` for:

- project-root resolution
- `data/.env` loading
- `$DATA_DIR` resolution
- consistent logging and fatal errors
- runtime `PUID` and `PGID`
- canonical coop `docker compose` invocation
- package-backed Python module execution via the project venv or `python3`

New shell scripts should source `scripts/shared/common.sh` instead of duplicating path or environment bootstrapping logic.
