# Implementation Plan (forward work)

Forward-only: every task in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md)
and the topic files under [`current/`](current/).

## Scope Update (2026-07): The Field Device Layer

The three playgrounds are shipped and documented in [`current.md`](current.md): the
production server answers queries, the local pipeline builds world-model understanding,
and the sencoop mesh collects ground truth. The mesh, however, is **uplink-only and
consume-only** (see [current/sencoop-mesh.md](current/sencoop-mesh.md), "Deliberate gaps"):
it ingests whatever third-party LoRaWAN devices and cameras emit, but it cannot provision,
command, update, or even inventory the devices it depends on, and it fields no first-party
sensing hardware. This scope update adds that missing layer.

### Why this scope, and why now

- **Persistent site awareness is the product.** Buyers of outdoor-autonomy systems
  (critical infrastructure, energy sites, agriculture, perimeter security) pay for
  continuous ground truth between missions, not for one-off video analysis. Camera-only
  systems have blind spots at night, in fog, and outside frustums; the differentiator is
  a cheap multi-modal mesh (environmental + RF + acoustic + motion) that keeps the world
  model honest. Our fusion and threat layers already exist; they are starved for sensors.
- **Off-grid resilience is now a hard requirement, not a feature.** The pattern proven in
  recent conflict and disaster response is infrastructure-less comms (LoRa mesh,
  Meshtastic-class networks), air-gapped deployable stacks (we already ship
  `sencoop-release` offline bundles), and stratospheric platforms (HAB) as low-cost
  persistent relay and wide-area observation. A HAB with a LoRa payload turns a
  single-site mesh into a region-scale collector for the cost of a hobby launch.
- **Edge intelligence on constrained silicon is maturing fast.** ESP32-class MCUs run
  useful TinyML; STM32 covers multi-year battery deployments; the XC7Z020 (Zynq-7000)
  gives a deterministic FPGA front-end for RF spectral work that a Linux SBC cannot do
  with stable latency. Sending semantics instead of raw samples cuts bandwidth, power,
  and RF signature -- all operationally decisive outdoors.
- **Agent-operated infrastructure is the near-term ops trend.** Fleets will be run by
  AI agents. That requires machine-readable device state (a registry), declarative
  provisioning (manifests, not UI clicks), and closed-loop verification (health ledger,
  acceptance gates). This repo is agent-built; the device layer below must be
  agent-operable by construction: every operation exposed as CLI or API, no UI-only path.
- **Open platforms win the integration battle.** ChirpStack, Frigate, OpenRemote,
  Node-RED, and Meshtastic are the de-facto open stacks; customers pick them to avoid
  vendor lock-in. Our value is the fusion, world-model, and threat analytics on top --
  we integrate platforms, we do not rebuild them.

### Stakeholder demand map

| Stakeholder | What they need from this scope |
| --- | --- |
| Operators | One pane of glass (OpenRemote), editable automation without code (Node-RED), incident ack/dismiss they already have in the v1 API |
| Planners | Coverage and mesh-health views, site survey from HAB tracks, RF baseline maps |
| Engineers | Reproducible firmware builds, one-command flash, OTA, device registry with firmware ledger, CI that gates every stack |
| Owners | Offline installable bundles, low cost per node (COTS boards), privacy and RF-compliance evidence |
| Users / robots | Better threat advisories: more independent modalities feeding the existing two-source evidence gate |

### Technology bets and simplicity rules

The main approach stays: **as simple as possible while delivering unique features.**

- **Python remains the integration plane.** All ingestion, APIs, fusion, and analytics
  stay in `src/sencoop` / `src/selfsuvis`. No new brokers, no Kubernetes.
- **One firmware build system.** C/C++ on the Arduino framework under **PlatformIO**
  for both ESP32 and STM32 targets: one `platformio.ini`, one native unit-test runner,
  one CI job. No vendor IDEs in the loop.
- **Go for exactly one thing:** the field gateway agent (`sencoop-agent`) -- a single
  static cross-compiled binary so gateways need no Python runtime. Standard library
  plus MQTT and serial deps only.
- **Rust is allowed but not scheduled.** Adopt only when a concrete SDR/DSP path needs
  memory-safe performance, and record the decision as an ADR first.
- **FPGA work is isolated and sim-first.** The XC7Z020 lives under `firmware/fpga/`,
  builds in a pinned Vivado container, and every software-side consumer must run
  against a numpy simulation so nothing else depends on hardware or licenses.
- **Cross-language contracts are golden-fixture tested.** The wire format between
  firmware (C), ChirpStack codec (JS), and ingestion (Python) is one committed set of
  golden frames decoded identically by all three.

### Target hardware (commodity-first)

| Chip / board | Role | Toolchain |
| --- | --- | --- |
| ESP32-S3 (Heltec WiFi LoRa 32 V3, SX1262) | LoRaWAN sensor node, presence scanner, HAB tracker/gateway | PlatformIO + Arduino + RadioLib |
| STM32WL55 (Nucleo-WL55JC) | Multi-year battery sensor node (integrated LoRa radio) | PlatformIO + Arduino (STM32duino + STM32LoRaWAN) |
| XC7Z020 (PYNQ-Z2) | Deterministic RF spectral front-end | Vivado 2022.1 (pinned container) + PYNQ 3.x |
| amd64 nettop / arm64 SBC | Gateway running sencoop stack + `sencoop-agent` | Docker + Go static binary |

## Forward Tasks

The forward work is split into two sections by **who must act to complete it**:

- **[Agent Implementation Tasks](#agent-implementation-tasks)** land with the unit gate
  green -- `make lint` + `make test-unit`, plus the per-stack gates each task states
  (`pio run` / `pio test -e native` for firmware, `go vet && go test ./...` for the
  agent). Once `ci-cross-stack` lands these unify under `make ci`. Committed fixtures,
  injected fakes, and deterministic harnesses; no human judgment required.
- **[Human-Assisted Tasks](#human-assisted-tasks)** cannot reach their stated acceptance
  without a human: physical hardware assembly and field placement, regulatory approvals
  (RF duty cycle, airspace), privacy sign-off, or measured operator feedback. An agent
  still builds the supporting code, docs, and tests; the marked **human step** gates
  completion.

Task ids are stable kebab-case slugs and never change; every task carries an explicit
`Dependencies` line. Three dependencies cross the section boundary and are called out
inline because they are **blocked by human work**: presence-scanner default-on ingest
(privacy review), HAB real-flight evidence (flight campaign), and XC7Z020 hardware
verification (bench bring-up).

## Agent Implementation Tasks

Recommended sequence: `device-registry-core` and `firmware-workspace` first (independent
roots); then `chirpstack-provisioning`, `downlink-command-bus`, `esp32-sensor-node`;
then `sencoop-agent-go` and `ci-cross-stack` (so every later task lands under CI); then
the automation pair (`node-red-automation`, `openremote-asset-sync`) plus
`esp32-ota-updates` and `meshtastic-mesh-bridge`; then field expansion
(`stm32wl-sensor-node`, `esp32-sigint-scanner`, `hab-telemetry-stack`); finally
`hab-mission-pipeline`, `xc7z020-rf-frontend`, `rf-threat-analytics`.

### `device-registry-core` -- device inventory, health ledger, firmware ledger

- **Dependencies:** none (root task).
- **User-visible outcome:** every field device (LoRaWAN node, Meshtastic node, camera,
  HAB payload, gateway) has a registry row with identity, transport, location, firmware
  version (reported vs desired), last-seen, and battery; operators and agents query and
  mutate it over the v1 API.
- **Scope boundary:** in scope -- PostgreSQL tables `mesh_devices`,
  `mesh_device_events` (append-only ledger: seen/provisioned/flashed/commanded/fault),
  `mesh_firmware` (app, board, version, sha256, url); Pydantic models and store in
  `src/sencoop/registry/{models,store}.py` (asyncpg, mirroring existing storage
  patterns); router `src/selfsuvis/app/routers/v1/devices.py` with
  `GET/POST /api/v1/devices`, `GET /api/v1/devices/{dev_eui}`,
  `POST /api/v1/devices/{dev_eui}/heartbeat`, `GET /api/v1/devices/{dev_eui}/events`;
  `MqttSubscriber` hook so every decoded uplink upserts last-seen/battery/fw-version.
  Out of scope -- any downlink, provisioning, or UI work. Reuse: migration runner
  `selfsuvis.scripts.migrate_postgres`, v1 router/schema conventions, fake DB pools in
  `tests/support/`.
- **Data and artifact paths:** migrations add tables; no filesystem artifacts.
- **Execution path:** `ssv-migrate` applies schema; run API locally (`make up` or
  uvicorn) and exercise endpoints with httpx; unit tests with fake pool.
- **Acceptance gates:** unit gate green; new tests cover upsert-from-uplink, heartbeat,
  event ledger append, and desired-vs-reported firmware diff; OpenAPI diff gate
  regenerated (`make export-openapi`).
- **Documentation target:** new `docs/impl/current/device-management.md` (registry
  section); row added to `current.md` Topic Map; `docs/reference/api.md`.

### `chirpstack-provisioning` -- declarative device provisioning CLI

- **Dependencies:** `device-registry-core`.
- **User-visible outcome:** `sencoop-provision apply site.yaml` idempotently creates
  ChirpStack applications, device profiles, devices, and OTAA keys from a committed
  manifest, and records each device in the registry -- no UI clicking.
- **Scope boundary:** in scope -- `src/sencoop/provision/{manifest,chirpstack_client,cli}.py`;
  manifest schema (site, applications, device_profiles, devices with dev_eui/join_eui/
  app_key/name/tags/gps); ChirpStack REST (`http://localhost:8090`, bearer
  `CHIRPSTACK_API_SECRET`) client with list/create/update; `apply` (idempotent diff),
  `plan` (dry-run diff), `export` (live -> manifest) subcommands; console script in root
  `pyproject.toml`. Out of scope -- gateway provisioning, FUOTA, key generation policy
  (keys come from the manifest or `--gen-keys` writing back to a secrets file under
  `$DATA_DIR/sencoop/provision/`). Reuse: registry store; httpx.
- **Data and artifact paths:** example manifest committed at
  `config/sencoop/provision/site-example.yaml`; recorded REST fixtures at
  `tests/assets/chirpstack/*.json`; generated secrets under `$DATA_DIR/sencoop/provision/`.
- **Execution path:** against live stack -- `make sencoop-up-min` then
  `sencoop-provision plan|apply config/sencoop/provision/site-example.yaml`; in CI --
  client tested against recorded fixtures with a fake transport.
- **Acceptance gates:** unit gate green; `apply` twice produces zero-change second run
  (idempotence test); `plan` output snapshot test; registry rows created.
- **Documentation target:** `docs/impl/current/device-management.md` (provisioning);
  `docs/sencoop/sensor-integration.md` gains the manifest path as the primary flow.

### `downlink-command-bus` -- commands to devices with audit trail

- **Dependencies:** `device-registry-core`.
- **User-visible outcome:** `POST /api/v1/devices/{dev_eui}/command` enqueues a LoRaWAN
  downlink (ChirpStack MQTT `application/{app_id}/device/{dev_eui}/command/down`,
  payload `{"devEui","confirmed","fPort","data":base64}`), rate-limited and recorded in
  the device event ledger; Node-RED and agents get a safe command path.
- **Scope boundary:** in scope -- `src/sencoop/commands/downlink.py`
  (`DownlinkPublisher` on aiomqtt, per-device token bucket honoring EU868 duty-cycle
  conservatism, default max 1 downlink / 60 s / device, env-tunable); command schema
  (typed commands: `set_interval`, `check_update` fPort 0xF0, `raw`); API route +
  ledger append; deny-by-default `COOP_DOWNLINK_ENABLED=false`. Out of scope --
  multicast, FUOTA, Meshtastic admin messages. Reuse: `MqttSubscriber` connection
  config, v1 router conventions.
- **Data and artifact paths:** none beyond ledger rows.
- **Execution path:** live -- enable flag, publish against `make sencoop-up-min`
  Mosquitto and observe in ChirpStack UI queue; tests -- fake MQTT client asserting
  topic/payload/rate-limit behavior.
- **Acceptance gates:** unit gate green; rate-limit test (second command within window
  is 429/deferred); base64/fPort encoding golden test; flag-off returns 403.
- **Documentation target:** `docs/impl/current/device-management.md` (command bus);
  `docs/reference/api.md`; `docs/reference/configuration.md` for new env vars.

### `node-red-automation` -- operator-editable automation layer

- **Dependencies:** none for the stack itself; example flows use
  `downlink-command-bus` when present.
- **User-visible outcome:** `COMPOSE_PROFILES=...,automation make sencoop-up` starts a
  pinned Node-RED with seeded flows: (1) `application/#` uplinks normalized and POSTed
  to `POST /api/v1/events/lorawan`; (2) Frigate person-detection -> example downlink
  command; (3) a broker/stack health dashboard. Operators edit automation without code.
- **Scope boundary:** in scope -- service `coop-nodered` (image `nodered/node-red`
  pinned by digest, localhost port 1880, volume `$DATA_DIR/sencoop/nodered/`) added to
  `docker/sencoop/docker-compose.sencoop.yml` under new profile `automation`;
  flows-as-code `config/sencoop/nodered/flows.json` + `settings.js` (adminAuth and
  `credentialSecret` from env); Mosquitto ACL user `nodered` (read `application/#`,
  `frigate/#`; write `application/+/device/+/command/down`); Make targets
  `sencoop-up-automation`. Out of scope -- custom Node-RED nodes, dashboards beyond the
  seeded three, exposing 1880 beyond localhost. Reuse: ACL/user tooling
  `scripts/sencoop/sencoop-mqtt-users.sh`, bootstrap script.
- **Data and artifact paths:** `config/sencoop/nodered/` (committed flows/settings);
  `$DATA_DIR/sencoop/nodered/` runtime.
- **Execution path:** `make sencoop-up-automation`; verify flow 1 by publishing a
  fixture uplink with `mosquitto_pub` and asserting the v1 event lands
  (`GET /api/v1/site/state`).
- **Acceptance gates:** unit gate green (compose/config lint via existing tests
  pattern); an integration-marked test drives fixture-publish -> API-event assertion;
  flows.json committed and loads without missing nodes.
- **Documentation target:** new `docs/impl/current/automation-platforms.md`;
  `docs/sencoop/getting-started.md` profile table.

### `openremote-asset-sync` -- ChirpStack devices as OpenRemote assets

- **Dependencies:** `device-registry-core`, `chirpstack-provisioning`.
- **User-visible outcome:** `sencoop-or-sync --watch` keeps OpenRemote assets in step
  with the registry/ChirpStack: one asset per device with attributes (battery,
  temperature, last-seen, position) updating live; operators get dashboards and alarms
  in the platform they already run.
- **Scope boundary:** in scope -- `src/sencoop/openremote/{client,asset_sync}.py`
  (Keycloak service-user token via client-credentials grant, OpenRemote Manager REST
  asset CRUD + attribute writes), mapping table registry-device -> asset type/attrs,
  `--once` and `--watch` (default 300 s) modes, console script `sencoop-or-sync`.
  Out of scope -- OpenRemote rules/dashboards content (human task
  `operator-dashboard-acceptance`), MQTT agent-link configuration inside OR. Reuse:
  registry store, httpx, recorded-fixture test pattern from provisioning.
- **Data and artifact paths:** fixtures `tests/assets/openremote/*.json`.
- **Execution path:** live -- full `make sencoop-up` (OR profile) then
  `sencoop-or-sync --once`; tests against recorded fixtures with fake transport.
- **Acceptance gates:** unit gate green; sync is idempotent (second `--once` is
  zero-change); attribute update path golden-tested; token refresh on 401 covered.
- **Documentation target:** `docs/impl/current/automation-platforms.md` (OpenRemote
  section); `docs/sencoop/integration.md`.

### `firmware-workspace` -- PlatformIO monorepo and the ssvnode wire contract

- **Dependencies:** none (root task).
- **User-visible outcome:** `firmware/` builds first-party node firmware for ESP32 and
  STM32 from one `platformio.ini`, and the `ssvnode` payload format is a tested
  three-way contract (C encoder, Python decoder, ChirpStack JS codec) so every future
  node speaks a format the mesh already understands.
- **Scope boundary:** in scope -- layout `firmware/{platformio.ini, lib/ssvnode/,
  lib/ssvcfg/, apps/, test/test_codec/, scripts/}` with PlatformIO envs `native`
  (host, Unity tests), `heltec_v3` (`board = heltec_wifi_lora_32_V3`, framework
  arduino, RadioLib pinned), `nucleo_wl55jc` (platform ststm32, framework arduino,
  STM32duino + STM32LoRaWAN pinned); `lib/ssvnode/ssvnode_codec.{h,c}` implementing the
  frame below; Python twin `src/sencoop/sensors/ssvnode_codec.py` wired into
  `lorawan_decoder.decode_chirpstack_uplink` (fPort 10/11 raw-bytes fallback when no
  codec object); JS codec `config/sencoop/chirpstack/codecs/ssvnode.js`; golden vectors
  `tests/assets/ssvnode/golden_frames.json` (hex frame + expected decoded JSON) consumed
  by both `pio test -e native` and pytest. Out of scope -- any app logic (later tasks),
  OTA, radios beyond library pinning.
  **ssvnode v1 frame (little-endian):** `version:u8=0x01 | flags:u8 | fields in bit
  order` -- bit0 `temperature_c:i16` centi-C; bit1 `humidity_pct:u16` centi-%; bit2
  `co2_ppm:u16`; bit3 `pressure:u16` = (hPa-300)*10; bit4 `battery_v:u8` = V*20; bit5
  `motion:u8` bit0 state, bits1-7 event count; bit6 gps `lat:i32` 1e-7 deg,
  `lon:i32` 1e-7 deg, `alt:u16` = (m+1000)/2; bit7 ext block `type:u8` + payload
  (type 0x01 presence: `wifi_count:u16, ble_count:u16, rssi_avg:i8 dBm,
  window_s:u16`). fPort 10 = telemetry; fPort 11 = status
  `fw_major:u8, fw_minor:u8, fw_patch:u8, reset_reason:u8, uptime_s:u32`.
- **Data and artifact paths:** committed golden fixtures as above; build output stays
  under `firmware/.pio/` (gitignored).
- **Execution path:** `pio test -e native` (codec Unity tests); `pio run -e heltec_v3
  -e nucleo_wl55jc` (build-only; no app yet, a minimal blink main per env);
  `make test-unit` runs the Python decoder against the same goldens.
- **Acceptance gates:** unit gate green; `pio test -e native` green; both device envs
  compile; every golden frame decodes byte-identically in C, Python, and (via committed
  node test script executed with `node`, if available, else documented manual check) JS.
- **Documentation target:** new `docs/impl/current/firmware.md` (workspace + wire
  contract); AGENTS.md "Current layout" gains the `firmware/` entry.

### `esp32-sensor-node` -- LoRaWAN class A environmental node

- **Dependencies:** `firmware-workspace`; provisioning manifest from
  `chirpstack-provisioning` reused for keys.
- **User-visible outcome:** flash a Heltec V3 and it joins via OTAA, uplinks ssvnode
  telemetry (BME280 temperature/humidity/pressure, PIR motion, battery ADC) every 300 s
  with deep sleep between, and reports a status frame (fPort 11) on boot -- appearing
  automatically in `/site/state` and the registry.
- **Scope boundary:** in scope -- `firmware/apps/sensor_node/` (main.cpp, RadioLib
  LoRaWAN EU868 OTAA with session persistence across deep sleep, uplink scheduler,
  sensor drivers behind `ssvcfg` compile flags, battery ADC calibration constant);
  `firmware/scripts/gen_secrets.py` generating `include/secrets.h` from the
  provisioning manifest (single source of keys); downlink handler for `set_interval`
  and `check_update` command codes. Out of scope -- OTA transfer itself
  (`esp32-ota-updates`), non-EU bands (config placeholder only).
- **Data and artifact paths:** golden uplink fixtures appended to
  `tests/assets/ssvnode/golden_frames.json`; built artifact
  `firmware/.pio/build/heltec_v3/firmware.bin`.
- **Execution path:** `pio run -e heltec_v3` builds; `pio test -e native` covers
  scheduler and encoding logic factored into `lib/`; on-hardware join/uplink is
  deferred to `field-pilot-mesh-site` (human).
- **Acceptance gates:** unit gate + native tests green; build succeeds with all sensor
  flags on and off; encoded fixture from app-level code decodes via Python decoder in
  pytest; deep-sleep session persistence logic unit-tested on native env with fake NVS.
- **Documentation target:** `docs/impl/current/firmware.md` (sensor node app);
  `docs/sencoop/sensor-integration.md` gains "first-party node" section.

### `stm32wl-sensor-node` -- long-endurance STM32WL node

- **Dependencies:** `firmware-workspace`, `esp32-sensor-node` (reuses app structure).
- **User-visible outcome:** the same ssvnode telemetry from a Nucleo-WL55JC using the
  integrated SX126x radio and STOP2 low-power mode -- the multi-year-battery variant of
  the sensor node for permanent installs.
- **Scope boundary:** in scope -- `firmware/apps/sensor_node_wl/` on the
  `nucleo_wl55jc` env (STM32LoRaWAN library, RTC-driven wakeup, STOP2 between uplinks,
  same ssvnode codec and secrets generation); shared app logic extracted to
  `firmware/lib/ssvapp/` so ESP32 and STM32 mains stay thin. Out of scope -- MCUboot /
  secure boot (future task if demanded), custom PCBs.
- **Data and artifact paths:** as `esp32-sensor-node`.
- **Execution path:** `pio run -e nucleo_wl55jc`; `pio test -e native` for shared
  `ssvapp` logic.
- **Acceptance gates:** unit gate + native tests green; both node apps build from the
  shared lib with no duplicated scheduler/codec code (checked by review, enforced by
  lib layout); documented measured-vs-budgeted sleep current table template (numbers
  filled by human field task).
- **Documentation target:** `docs/impl/current/firmware.md` (WL variant + power budget).

### `esp32-ota-updates` -- firmware publish and WiFi OTA loop

- **Dependencies:** `device-registry-core`, `esp32-sensor-node`.
- **User-visible outcome:** `sencoop-firmware publish firmware/.pio/build/heltec_v3/
  firmware.bin --app sensor_node --board heltec_v3 --version 1.2.0` stages a signed
  (sha256) artifact; WiFi-capable nodes told `check_update` fetch the manifest, update,
  and the registry shows reported version converging to desired.
- **Scope boundary:** in scope -- publisher CLI `src/sencoop/firmware/publish.py`
  writing `$DATA_DIR/sencoop/firmware/<app>/<board>/{firmware.bin,manifest.json}`
  (manifest: app, board, version, sha256, size, url) and upserting `mesh_firmware`;
  nginx proxy static location `/firmware/` serving that directory (config under
  `config/sencoop/`); device side -- HTTPUpdate flow in `ssvapp` triggered by boot
  check or `check_update` downlink, guarded by sha256 verify and version compare;
  registry endpoint `GET /api/v1/devices/updates-pending`. Out of scope -- LoRaWAN
  FUOTA (explicitly deferred; revisit as its own task when a WiFi-less fleet demands
  it), STM32 DFU (agent task `sencoop-agent-go` covers wired flashing).
- **Data and artifact paths:** `$DATA_DIR/sencoop/firmware/` tree; manifest golden
  fixtures under `tests/assets/firmware/`.
- **Execution path:** publish CLI against a temp `$DATA_DIR` in tests; end-to-end on
  hardware deferred to `field-pilot-mesh-site`.
- **Acceptance gates:** unit gate green; publish is idempotent and refuses
  version-downgrade without `--force`; manifest schema round-trips; registry
  desired/reported diff endpoint tested.
- **Documentation target:** `docs/impl/current/device-management.md` (OTA);
  `docs/sencoop/distribution.md`.

### `esp32-sigint-scanner` -- privacy-preserving presence sensing

- **Dependencies:** `firmware-workspace`, `esp32-sensor-node`. **Cross-section block:**
  enabling ingestion by default is gated by human task `sigint-privacy-review`; until
  then `COOP_PRESENCE_INGEST=false`.
- **User-visible outcome:** a scanner node counts nearby WiFi probe-request emitters
  and BLE advertisers per window (counts and coarse RSSI only -- no identifiers ever
  leave the device) and uplinks an ssvnode ext-presence block; with the flag on, counts
  appear in `/site/state` and feed the threat layer as `rf_presence` events.
- **Scope boundary:** in scope -- `firmware/apps/presence_scanner/` (esp_wifi
  promiscuous mgmt-frame filter + NimBLE scan; per-window dedupe via salted truncated
  SHA-256 where the salt derives from device key + window counter and is discarded --
  nothing persistent, `SSV_PRIVACY_COUNTS_ONLY=1` is compile-default and the only
  shipped mode); ingestion -- decoder aliases `wifi_count`, `ble_count`, `rssi_avg` and
  `coop_ingest.sensor_reading_to_event` emitting
  `SensorEvent(sensor_type="rf_presence")` behind `COOP_PRESENCE_INGEST`. Out of scope
  -- MAC vendor analysis, per-device tracking, deauth or any active transmission
  (passive receive only), channel hopping tuning beyond a fixed default list.
- **Data and artifact paths:** golden ext-block fixtures in
  `tests/assets/ssvnode/golden_frames.json`.
- **Execution path:** `pio run -e heltec_v3` (scanner env variant); native tests for
  window/dedupe/encode logic with injected fake scan results; pytest for ingestion
  flag-on/flag-off behavior.
- **Acceptance gates:** unit gate + native tests green; a code-level assertion/test
  proves no field wider than counts/RSSI is encodable in the presence block; flag-off
  drops events with a single startup log line.
- **Documentation target:** `docs/impl/current/firmware.md` (scanner);
  `docs/operations/` privacy note stub referencing the pending review.

### `meshtastic-mesh-bridge` -- LoRa mesh transport ingestion

- **Dependencies:** `device-registry-core`.
- **User-visible outcome:** stock-firmware Meshtastic nodes (with an MQTT-uplink
  gateway node pointed at coop-mosquitto, topic root `msh`) appear as registry devices
  and mesh members: positions and telemetry land in `/site/mesh` and `/site/state`,
  text messages land in the device event ledger -- infrastructure-less coverage beyond
  LoRaWAN star range.
- **Scope boundary:** in scope -- `src/sencoop/sensors/meshtastic_bridge.py`
  subscribing `msh/#`, decoding `ServiceEnvelope` protobufs via the `meshtastic` pip
  package (added to the `sencoop` extra), handling POSITION_APP, TELEMETRY_APP,
  NODEINFO_APP, TEXT_MESSAGE_APP; mapping to registry (`transport="meshtastic"`),
  `SensorMeshFusion` nodes, and `SensorEvent(sensor_type="mesh")`; Mosquitto ACL user
  `meshtastic`; provisioning helper `sencoop-mesh-provision` that shells to the
  `meshtastic` CLI over serial to set region/channel/MQTT (documented; hardware run
  deferred to field pilot). Out of scope -- acting as a mesh router ourselves, custom
  Meshtastic firmware, encrypted-channel key management beyond documenting defaults.
- **Data and artifact paths:** captured envelope fixtures (base64) under
  `tests/assets/meshtastic/`.
- **Execution path:** pytest against fixtures with fake MQTT; live smoke via
  `mosquitto_pub` replay against `make sencoop-up-min`.
- **Acceptance gates:** unit gate green; each handled port number has a
  fixture-decode test; unknown ports are counted and skipped without error; registry
  upsert covered.
- **Documentation target:** `docs/impl/current/sencoop-mesh.md` (transport matrix
  gains meshtastic); `docs/sencoop/sensor-integration.md`.

### `hab-telemetry-stack` -- HAB payload, ground station, live track

- **Dependencies:** `firmware-workspace`, `esp32-sensor-node`. **Cross-section note:**
  real-flight evidence comes from human task `hab-flight-campaign`; all acceptance here
  is simulation-based.
- **User-visible outcome:** a high-altitude balloon payload streams position/altitude/
  environment over long-range LoRa P2P to a ground station that publishes
  `hab/telemetry/{callsign}` into the mesh; the flight appears live in `/site/state`
  and `/site/mesh` with an ascent/burst/descent/landed state machine -- region-scale
  collection from a hobby-launch budget.
- **Scope boundary:** in scope -- firmware `firmware/apps/hab_tracker/` (ESP32 +
  u-blox GPS with UBX airborne <1g dynamic model set at boot and re-verified, BME280,
  LoRa P2P TX duty-cycle-aware default 1 frame / 30 s) and `firmware/apps/hab_gateway/`
  (RX-only, frames -> JSON lines over USB serial); ground side
  `src/sencoop/sensors/hab_ground_station.py` (pyserial reader, CRC check, MQTT
  publish, registry heartbeat) and flight state machine (vertical-rate thresholds
  +2 / -3 m/s with hysteresis); simulator CLI `sencoop-hab-sim` replaying a committed
  ascent CSV to a pty or MQTT. **HAB v1 frame (32 bytes, little-endian):**
  `sync:u16="SV" | ver:u8 | callsign:6xASCII | counter:u16 | gps_tow_s:u32 | lat:i32
  1e-7 | lon:i32 1e-7 | alt_m:u16 | speed_mps:u8 | sats:u8 | temp_c:i8 | batt:u8=V*20
  | state:u8 | crc16_ccitt:u16`. Out of scope -- cutdown control, landing prediction
  (note tawhiri integration as possible future task), APRS/amateur-radio modes,
  redundant trackers.
- **Data and artifact paths:** sim flight CSV `tests/assets/hab/ascent_sim.csv`;
  flight logs `$DATA_DIR/sencoop/hab/<flight_id>/track.jsonl`.
- **Execution path:** `pio run -e heltec_v3` for both apps; `sencoop-hab-sim --mqtt`
  end-to-end against `make sencoop-up-min`; pytest on frame codec, CRC, state machine.
- **Acceptance gates:** unit gate + native tests green; sim replay produces a
  monotonic track in `/site/state`, correct burst detection at the CSV apex, and a
  complete `track.jsonl`; frame encode/decode golden-tested C-vs-Python.
- **Documentation target:** new `docs/impl/current/hab.md`; launch checklist lives
  with the human task.

### `hab-mission-pipeline` -- flight track + payload video into the world model

- **Dependencies:** `hab-telemetry-stack`. **Cross-section block:** acceptance on real
  footage is gated by `hab-flight-campaign`; sim + sample-asset acceptance is agent-side.
- **User-visible outcome:** after recovery, one command turns a flight into a mission:
  `ssv --mode local --video flight.mp4 --gps-track $DATA_DIR/sencoop/hab/<id>/track.jsonl`
  aligns frames to the telemetry track, and the run emits a `coverage.geojson`
  (camera ground footprint from altitude + FOV) plus an "HAB flight" report section;
  the production server indexes it as a mission with GPS payloads.
- **Scope boundary:** in scope -- `--gps-track` sidecar ingestion in `ssv_vdp`
  (external track into the fusion/physical-state steps as a first-class pose source,
  time-aligned by timestamp with offset estimation), footprint math in
  `src/ssv_vdp/steps/` (nadir-assumption v1 with tilt passthrough when IMU present),
  `coverage.geojson` artifact + report section, and a helper that POSTs the recovered
  video to `/index/video` with mission GPS metadata. Out of scope -- photogrammetry
  from HAB imagery (existing SfM steps run or degrade per current gating), non-nadir
  rigorous projection.
- **Data and artifact paths:** run outputs under the standard `$DATA_DIR` run tree
  (`coverage.geojson`, report section); sample inputs from `tests/assets/hab/` plus an
  existing small video asset from `tests/assets/`.
- **Execution path:** `ssv --mode local --video tests/assets/<small>.mp4 --gps-track
  tests/assets/hab/ascent_sim_track.jsonl` completes with the artifact; pytest unit
  tests for alignment and footprint math.
- **Acceptance gates:** unit gate green; alignment handles missing/duplicate
  timestamps; geojson validates (shape + CRS); `analysis_summary.json` reports the
  track modality as present.
- **Documentation target:** `docs/impl/current/hab.md` (mission integration);
  `docs/impl/current/local-pipeline.md` step-family table.

### `xc7z020-rf-frontend` -- FPGA spectral front-end (sim-first)

- **Dependencies:** `firmware-workspace` (layout conventions only). **Cross-section
  block:** hardware verification is gated by human task `zynq-hardware-bringup`; every
  agent-side gate runs in simulation.
- **User-visible outcome:** a PYNQ-Z2 (XC7Z020) node computes averaged RF spectra in
  fabric (AXI DMA + FFT LogiCORE) and publishes
  `sensors/rf_spectrum/{node}` JSON (`center_hz, span_hz, bins[], noise_floor_dbm,
  peaks[]`) into the mesh; without hardware, the identical publisher runs in `--sim`
  mode over recorded IQ, so the analytics stack develops against real message shapes.
- **Scope boundary:** in scope -- `firmware/fpga/zynq7020/` with committed Vivado
  2022.1 block-design TCL + `Makefile bitstream` running inside a pinned container
  (`firmware/fpga/docker/Dockerfile.vivado`; the Vivado installer/license is a human
  prerequisite documented there -- CI never builds bitstreams); PS-side
  `firmware/fpga/zynq7020/pynq/rf_spectrum_publisher.py` (PYNQ overlay load, DMA
  capture, Welch-style averaging, peak extraction, MQTT publish) with `--sim` numpy
  path over `tests/assets/rf/iq_sample.npz`; subscriber-side handling of
  `sensors/rf_spectrum/#` -> `SensorEvent(sensor_type="rf_spectrum")`. Out of scope --
  RF front-end hardware selection (test vectors from PS memory first), demodulation,
  direction finding, Vitis AI / neural overlays.
- **Data and artifact paths:** IQ fixture `tests/assets/rf/iq_sample.npz`; bitstreams
  land under `$DATA_DIR/sencoop/fpga/` (never committed; sha256 recorded in
  `mesh_firmware`).
- **Execution path:** `python firmware/fpga/zynq7020/pynq/rf_spectrum_publisher.py
  --sim --mqtt localhost` against `make sencoop-up-min`; pytest on peak extraction and
  message schema; bitstream build documented as heavy host-only step.
- **Acceptance gates:** unit gate green; sim publisher emits schema-valid messages at
  the configured rate; known synthetic tone in the IQ fixture is found within one bin;
  ingestion test covers the new sensor_type.
- **Documentation target:** `docs/impl/current/firmware.md` (FPGA section);
  `docs/sencoop/sensor-integration.md`.

### `rf-threat-analytics` -- baselines, anomalies, and the RF field map

- **Dependencies:** `esp32-sigint-scanner` and/or `xc7z020-rf-frontend` and/or
  `meshtastic-mesh-bridge` (any RF-ish source); `device-registry-core`.
- **User-visible outcome:** `GET /site/rf` returns per-node RF baselines (presence
  counts, band power), current anomaly scores, and an optional Gaussian-process field
  map over the site grid; sustained anomalies feed the threat aggregator as
  `ThreatEvent(sensor_type="rf")` -- turning point RF readings into spatial risk, the
  future-directions "environmental fields" theme made concrete.
- **Scope boundary:** in scope -- `src/sencoop/mesh/rf_baseline.py` (per-node,
  per-band EWMA baseline keyed by hour-of-day, robust z-score anomaly with env-tunable
  thresholds and a minimum-duration gate), GPR field estimate
  (`sklearn.gaussian_process`, optional dep in the `sencoop` extra; grid over the site
  bbox at the existing ~110 m sector resolution; mean + uncertainty per cell), the
  `/site/rf` route, and threat-aggregator wiring. Out of scope -- emitter
  localization/DF, classification of signal types, cross-mission persistence (that is
  the future `global-threat-persistence` theme).
- **Data and artifact paths:** synthetic series fixtures
  `tests/assets/rf/baseline_series.json`.
- **Execution path:** pytest-driven; live smoke by replaying fixtures through MQTT.
- **Acceptance gates:** unit gate green; injected anomaly is flagged and a quiet
  series is not (bounded false-positive test); GPR path degrades gracefully when
  sklearn is absent; endpoint schema tested.
- **Documentation target:** `docs/impl/current/sencoop-mesh.md` (RF analytics);
  `docs/learning_path/18_future_directions.md` updated to mark field models partially
  implemented.

### `sencoop-agent-go` -- single-binary field gateway agent

- **Dependencies:** `device-registry-core`; consumes artifacts from
  `esp32-ota-updates` when present.
- **User-visible outcome:** one static Go binary on any gateway (amd64/arm64) gives
  `sencoop-agent discover` (serial devices by VID:PID -- CP210x/CH340 for ESP32,
  ST-Link for Nucleo, RAK), `sencoop-agent flash --app sensor_node --board heltec_v3
  --port /dev/ttyUSB0` (orchestrates esptool / dfu-util / meshtastic CLIs, verifies
  manifest sha256, records a `flashed` ledger event), `sencoop-agent inventory` and
  `sencoop-agent health --watch` (MQTT status with LWT + registry heartbeats) -- field
  provisioning without a Python environment.
- **Scope boundary:** in scope -- Go module at `src/sencoop-agent/` (go >= 1.22; deps
  limited to eclipse/paho.mqtt.golang and go.bug.st/serial), subcommands above,
  `SENCOOP_AGENT_*` env + flags config, `make -C src/sencoop-agent build test` with
  cross-compile (linux/amd64, linux/arm64) into `.data/sencoop/agent/bin/`. External
  flash tools are orchestrated, not reimplemented: missing tool -> actionable error
  naming the package to install. Out of scope -- implementing flash protocols in Go,
  Windows support, agent self-update (bundle-managed).
- **Data and artifact paths:** binaries under `.data/sencoop/agent/bin/`; ledger
  events via API.
- **Execution path:** `go vet ./... && go test ./...` (fake serial enumerator +
  `httptest` registry + fake exec runner); manual smoke against live stack.
- **Acceptance gates:** `go vet`/`go test` green and wired into `make ci` by
  `ci-cross-stack`; flash flow test proves sha256-mismatch aborts before invoking the
  tool; LWT/health topic golden-tested.
- **Documentation target:** `docs/impl/current/device-management.md` (agent);
  AGENTS.md layout entry for `src/sencoop-agent/`.

### `ci-cross-stack` -- one gate across Python, firmware, Go, and bundles

- **Dependencies:** `firmware-workspace`, `sencoop-agent-go` (jobs activate per-path;
  the workflow can land earlier with only Python jobs).
- **User-visible outcome:** every PR runs a path-filtered matrix -- Python lint+unit,
  firmware builds + native tests, Go vet+test+cross-build, FPGA lint -- and one local
  command (`make ci`) reproduces it; offline release bundles can embed firmware and
  agent artifacts, so a field kit ships from one build.
- **Scope boundary:** in scope -- `.github/workflows/ci.yml` with jobs: `python`
  (ruff + `pytest tests/unit`), `firmware` (paths `firmware/**`: `pio run` all envs +
  `pio test -e native`, PlatformIO cache), `go-agent` (paths `src/sencoop-agent/**`:
  vet, test, cross-build, upload artifacts), `fpga-lint` (TCL/py static checks only --
  never Vivado); Makefile additions `make ci` (lint + test-unit + firmware native
  tests when `pio` is on PATH + agent tests when `go` is on PATH; each missing
  toolchain prints one warning line and skips), `make firmware`, `make agent`;
  `scripts/sencoop/sencoop-release.sh --with-firmware --with-agent` embedding
  `$DATA_DIR/sencoop/firmware/` manifests and agent binaries into the offline bundle
  with a bundle manifest listing. Out of scope -- hardware-in-the-loop runners
  (revisit after the field pilot), nanochat/sslm CI, bitstream builds in CI.
- **Data and artifact paths:** CI artifacts (firmware .bin, agent binaries); bundle
  output as today under the release script's output tree.
- **Execution path:** `make ci` locally; `actionlint` on the workflow when available;
  a bundle build with both flags and an inspection of its manifest.
- **Acceptance gates:** `make ci` green on a full checkout and on a toolchain-less
  checkout (skips, does not fail); workflow triggers verified by path-filter unit
  cases in a dry-run; bundle contains the declared artifacts.
- **Documentation target:** `docs/impl/current/build-ci-test.md` (rewrites the "Known
  gaps" section); `docs/sencoop/distribution.md`.

## Human-Assisted Tasks

Each task's code and unit tests are agent-buildable; the marked **human step** gates
completion.

### `sigint-privacy-review` -- privacy sign-off for presence sensing

- **Dependencies:** `esp32-sigint-scanner` (built, flag-off). **Blocks:** flipping
  `COOP_PRESENCE_INGEST` default and any claim of production presence sensing.
- **User-visible outcome:** a recorded decision that counts-only presence sensing is
  acceptable for the deployment jurisdictions, with the boundary conditions written
  down where operators will see them.
- **Human step:** owner/legal review and sign-off of the assessment; jurisdictional
  check (GDPR-style device-identifier rules differ even for hashed, discarded scans).
- **Agent-buildable support:** draft assessment at `docs/operations/sigint-privacy.md`
  (data captured, retention = none beyond counts, salt lifecycle, threat model), plus
  the config plumbing already landed with the scanner task.
- **Acceptance gates:** signed-off doc committed; default flip lands as a one-line
  config change referencing the doc.
- **Documentation target:** `docs/operations/sigint-privacy.md`;
  `docs/reference/configuration.md`.

### `field-pilot-mesh-site` -- one real site, seven-day soak

- **Dependencies:** `esp32-sensor-node`, `meshtastic-mesh-bridge`,
  `sencoop-agent-go`, `chirpstack-provisioning`; `node-red-automation` recommended.
- **User-visible outcome:** a documented reference deployment: 3 sensor nodes,
  2 Meshtastic nodes, 1 LoRaWAN gateway, 1 camera, 1 nettop running the stack --
  with a seven-day soak report (uptime, packet loss, battery slope, at least one
  camera+sensor fused incident) that becomes the honesty benchmark for the docs.
- **Human step:** hardware purchase and assembly, EU868 duty-cycle compliance check,
  antenna/node placement, physical install, and the seven-day watch.
- **Agent-buildable support:** BOM + install runbook `docs/runbooks/field-pilot.md`;
  `sencoop-analytics soak --days 7` report additions (per-device uptime, loss,
  battery regression) in `src/sencoop/analytics/`.
- **Acceptance gates:** soak report committed under `docs/sencoop/` with real numbers;
  every deviation filed as a new forward task.
- **Documentation target:** `docs/runbooks/field-pilot.md`;
  `docs/impl/current/sencoop-mesh.md` sizing section updated with measured data.

### `hab-flight-campaign` -- one recovered flight

- **Dependencies:** `hab-telemetry-stack`; `hab-mission-pipeline` for the post-flight
  run. **Blocks:** real-flight acceptance of both HAB tasks.
- **User-visible outcome:** one legal, recovered HAB flight with continuous telemetry
  logged into site state and a post-recovery `ssv` mission run over payload video.
- **Human step:** regulatory clearance (airspace notification/permission per
  jurisdiction), launch logistics, chase and recovery.
- **Agent-buildable support:** launch checklist `docs/runbooks/hab-launch.md`;
  `sencoop-hab-sim preflight` validation (GPS airborne mode confirmed, TX interval
  legal for band, battery margin vs predicted flight time, ground-station lock).
- **Acceptance gates:** flight track archived under `$DATA_DIR/sencoop/hab/<id>/` and
  summarized in docs; mission run artifacts produced; lessons filed as forward tasks.
- **Documentation target:** `docs/impl/current/hab.md` (flight evidence section);
  `docs/runbooks/hab-launch.md`.

### `zynq-hardware-bringup` -- XC7Z020 bench validation

- **Dependencies:** `xc7z020-rf-frontend`. **Blocks:** hardware-verified claims for
  the FPGA front-end.
- **User-visible outcome:** the spectral overlay running on a physical PYNQ-Z2,
  matching the simulation within stated tolerance on a loopback test vector.
- **Human step:** board procurement, Vivado install/license acceptance in the pinned
  container, bench bring-up (boot PYNQ image, load overlay, run loopback IQ).
- **Agent-buildable support:** bring-up runbook `docs/runbooks/pynq-z2.md`;
  sim-vs-hardware comparison script emitting a pass/fail delta report.
- **Acceptance gates:** comparison report committed (per-bin power delta within
  tolerance on the synthetic tone); bitstream sha256 recorded in `mesh_firmware`.
- **Documentation target:** `docs/impl/current/firmware.md` (FPGA verified matrix).

### `operator-dashboard-acceptance` -- operators judge the pane of glass

- **Dependencies:** `openremote-asset-sync`, `node-red-automation`,
  `field-pilot-mesh-site` (live data to look at).
- **User-visible outcome:** OpenRemote dashboards and Node-RED flows that real
  operators used for a week and rated workable -- the "satisfaction for users,
  planners, engineers, operators, and owners" claim backed by evidence.
- **Human step:** at least two operators use the dashboards for a week during the
  soak; structured feedback session; sign-off that alarms are neither silent nor noisy.
- **Agent-buildable support:** seeded dashboard/asset templates, feedback form
  template in docs, and fixes for the feedback that lands as concrete issues.
- **Acceptance gates:** feedback summary committed; every accepted item either fixed
  or filed as a forward task.
- **Documentation target:** `docs/impl/current/automation-platforms.md` (operator
  acceptance section).

## Adding Future Tasks

Add a task only when there is concrete forward work with enough detail for an engineer or an
agent to execute without guessing. Use a stable descriptive id such as `lorawan-fuota`
or `global-threat-persistence`; keep the id only while work remains under it. Place it under
**Agent Implementation Tasks** if it can land with the unit gate and per-stack gates green
using fixtures/fakes (heavy deterministic runs on the CUDA host are fine), or under
**Human-Assisted Tasks** if a human review/judgment or authorization gates completion; either
way give it a `Dependencies` line and mark any cross-section block explicitly.

Each task entry must include:

- Dependencies: prerequisite tasks (by id), any cross-section block, and -- for
  human-assisted tasks -- the specific human step that gates completion.
- User-visible outcome: what new capability or decision the work should create.
- Scope boundary: what is in scope, what is explicitly out of scope, and which existing
  modules or commands should be reused.
- Data and artifact paths: expected fixtures, config, `$DATA_DIR/...` outputs, and any
  committed test assets.
- Execution path: commands, manual run steps, required local services, and any
  heavy/dependent steps that must stay outside quick CI.
- Acceptance gates: tests, lint/type checks, thresholds, or manual evidence required
  before the item leaves this file.
- Documentation target: the narrow `docs/impl/current/*.md` topic and any guide that
  should receive the resulting behavior and run notes.

When a task surfaces new future work, add that as a new forward task. Put current behavior
and durable decisions in current docs, never in this plan.

Known future-task candidates deliberately not scheduled yet: `lorawan-fuota` (fleet OTA
without WiFi), `mcuboot-secure-boot` (STM32 signed boot), `hab-landing-prediction`
(tawhiri integration), `global-threat-persistence` (cross-mission sector history --
future-directions theme 4), `rust-sdr-dsp` (only with an ADR), `hil-runner` (self-hosted
hardware-in-the-loop CI after the field pilot).
