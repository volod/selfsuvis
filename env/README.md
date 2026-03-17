# Environment configuration

Env files define defaults per environment. Set `APP_ENV` to select:

- `dev` — local development (default)
- `test` — integration tests (Docker)
- `prod` — production / `make up`

Files: `env/dev.env`, `env/test.env`, `env/prod.env`. Environment variables override file values.

For local overrides, create `.env` in the project root; it is loaded as fallback when the selected env file is missing.
