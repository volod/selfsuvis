# Data Layout and Configuration

## Configuration facade

`selfsuvis.config` is a Pure Fabrication package re-exporting three validated
config subsystems with a single `validate_all()` entry point:

```python
from selfsuvis.config import settings, coop_settings, realtime_settings
```

- `settings` -- core runtime (`src/selfsuvis/pipeline/core/config/`).
- `coop_settings` -- proxies `sencoop.config.settings` (`COOP_*` env vars).
- `realtime_settings` -- `src/selfsuvis/realtime/config.py`.

Deep imports keep working; new code should import the facade.

## Environment generation

- `make env` / `make env-interactive` run `ssv-env`
  (`selfsuvis.scripts.generate_env`): packaged presets (`src/selfsuvis/env/*.env`,
  `src/sencoop/env/*.env`, `src/selfsuvis/realtime/env/`) + detected hardware
  (GPU/RAM) -> a project-root `.env`. This is the standard bootstrap; `.env.example`
  documents the surface.
- Model selection honors `auto` values resolved by
  `pipeline/vision/registry.resolve_model_id` against the model catalog
  (`docs/reference/model-catalog.md`).

## Data layout rules (enforced by AGENTS.md)

- All runtime data lives under `$DATA_DIR` (default `.data/`), namespaced by
  module: `.data/nanochat/`, `.data/sslm/`, postgres/qdrant/videos dirs created by
  `make data-dirs`. Never a module-local `.data/` inside `src/`.
- `.data/wheels/` is reserved exclusively for compiled wheel artifacts, keyed by
  ABI dimensions (e.g. `torch2.9.1_cu128`).
- No hardcoded absolute paths in committed code; resolve from the project root
  and honor `.env` / `DATA_DIR`. Caches derive under `$DATA_DIR`
  (e.g. `.data/uv-cache/`).
- Sencoop bind mounts (`$DATA_DIR/...`: postgresql, manager, chirpstack-postgres,
  mosquitto, chirpstack-redis, frigate-media, prometheus, proxy) are created by
  `scripts/sencoop/sencoop-data-dirs.sh` before first start.

## Secrets

- Managed per `docs/reference/secrets-management.md` (separation + rotation).
- Production auth fails closed when required secrets are missing; API keys are
  compared timing-safe; empty `ALLOWED_INDEX_PATHS` disables all path endpoints.
- Key env groups: `API_KEY`, `COOP_MQTT_*`, `REASONING_API_URL` /
  `REASONING_MODEL` / `REASONING_TIMEOUT_SEC`, ChirpStack `CHIRPSTACK_API_SECRET`,
  Mosquitto users/ACLs (`config/sencoop/mosquitto/aclfile`,
  `scripts/sencoop/sencoop-mqtt-users.sh`).

## Reference docs

`docs/reference/configuration.md` (all env vars),
`docs/reference/data_layout.md` (output tree), `docs/reference/model-catalog.md`
(VRAM budgets and model options).
