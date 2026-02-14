.PHONY: up down logs venv test test-unit

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

venv:
	uv venv .venv
	./scripts/install_requirements.sh requirements/requirements_dev.txt .venv

# Integration tests (require API + worker + Qdrant)
test:
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.test.yml down --remove-orphans

# Unit tests (no services required)
test-unit:
	pytest tests/unit/ -v
