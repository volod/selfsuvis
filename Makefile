.PHONY: up down logs venv
.PHONY: test

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

venv:
	uv venv .venv
	./scripts/install_requirements.sh requirements/requirements_dev.txt .venv

test:
	INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
	INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.test.yml down --remove-orphans

test-dir:
	INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
	INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.test.yml down --remove-orphans
