# Scripts

Project scripts now live directly under `scripts/`.

## Canonical commands

- `install_system_deps.sh` — install host system packages for local development
- `install_requirements.sh` — install Python dependencies into an existing virtualenv
- `install_nvidia_docker.sh` — install NVIDIA Container Toolkit for Docker
- `selfsuvis-setup.sh` — end-to-end bootstrap for the main selfsuvis local environment
- `selfsuvis-prepare-sensor-data.sh` — fetch or scaffold public sensor sample data
- `selfsuvis-reset-qdrant.sh` — delete the configured Qdrant collection
- `selfsuvis-realtime-bridge.sh` — run the packaged MAVSDK or ROS realtime telemetry bridge runtime

## Coop stack commands

- `coop-bootstrap.sh` — one-shot coop stack bootstrap and startup
- `coop-compose.sh` — canonical coop `docker compose` wrapper with runtime `PUID` and `PGID`
- `coop-env.sh` — generate `data/.env` from the coop env templates
- `coop-credentials.sh` — print coop service URLs and generated credentials
- `coop-data-dirs.sh` — create coop bind-mount directories under `$DATA_DIR`
- `coop-mqtt-users.sh` — build the Mosquitto password file from `.env`
- `coop-mosquitto-tls.sh` — generate self-signed Mosquitto TLS certs
- `coop-clean-data.sh` — destructive reset of coop bind-mounted runtime data
- `coop-test-usb-cameras.sh` — inspect V4L2 devices and optionally test capture
- `coop-camera.sh` — update the Frigate camera config and optionally restart Frigate

## Project utility

- `project-package.sh` — create a tarball of the current repo while excluding generated secrets and runtime data

## Compatibility aliases

Legacy names are kept as thin wrappers for compatibility:

- `setup_local_full.sh` → `selfsuvis-setup.sh` (inherits the default Utilyze install; pass `--no-utilyze` to skip)
- `prepare_sensor_data.sh` → `selfsuvis-prepare-sensor-data.sh`
- `reset_qdrant.sh` → `selfsuvis-reset-qdrant.sh`
- `bootstrap.sh` → `coop-bootstrap.sh`
- `compose.sh` → `coop-compose.sh`
- `gen-env.sh` → `coop-env.sh`
- `first_run_setup.sh` → `coop-credentials.sh`
- `ensure_data_dirs.sh` → `coop-data-dirs.sh`
- `init_mosquitto_users.sh` → `coop-mqtt-users.sh`
- `gen_mosquitto_selfsigned_tls.sh` → `coop-mosquitto-tls.sh`
- `clean_data.sh` → `coop-clean-data.sh`
- `test_usb_cameras.sh` → `coop-test-usb-cameras.sh`
- `add_camera.sh` → `coop-camera.sh`
- `package.sh` → `project-package.sh`

## Shared shell helpers

Shell entrypoints reuse `common.sh` for:

- project-root resolution
- `data/.env` loading
- `$DATA_DIR` resolution
- consistent logging and fatal errors
- runtime `PUID` and `PGID`
- canonical coop `docker compose` invocation
- package-backed Python module execution via the project venv or `python3`

New shell scripts should source `scripts/common.sh` instead of duplicating path or environment bootstrapping logic.
