.PHONY: help up down logs data-dirs fix-data venv venv-pip docker-check test test-no-gpu test-unit test-unit-no-cv2 test-dir lint

# Default target: show help when no target is given
help:
	@echo "=============================================="
	@echo "  Video Semantic Search - Make targets"
	@echo "=============================================="
	@echo ""
	@echo "  Stack (Docker)"
	@echo "  ---------------"
	@echo "  make up              Start API, worker, Qdrant, and UI (docker compose up --build)"
	@echo "  make down            Stop all containers"
	@echo "  make logs            Stream container logs (last 100 lines)"
	@echo "  make docker-check    Check that Docker daemon is reachable (run if you get permission denied)"
	@echo ""
	@echo "  Local dev (venv)"
	@echo "  -----------------"
	@echo "  make venv            Create .venv, install pip, and install Python deps (requires uv on PATH)"
	@echo "  make venv-pip        Install pip into an existing .venv (e.g. after uv venv .venv)"
	@echo ""
	@echo "  Tests"
	@echo "  -----"
	@echo "  make test            Full integration tests in Docker (API + worker + Qdrant; needs GPU or test-no-gpu)"
	@echo "  make test-no-gpu     Same as test but without GPU (use if NVIDIA Container Toolkit is not installed)"
	@echo "  make test-dir        Same as test; set INDEX_DIR_PATH for directory-indexing tests"
	@echo "  make test-unit       Unit tests on host (no Docker; use .venv if present)"
	@echo "  make test-unit-no-cv2  Unit tests skipping cv2-dependent tests (if numpy/opencv version mismatch)"
	@echo ""
	@echo "  Code quality"
	@echo "  --------------"
	@echo "  make lint            Run ruff check and ruff format --check (install ruff if needed)"
	@echo ""
	@echo "  Troubleshooting"
	@echo "  ----------------"
	@echo "  Docker permission denied:  sudo usermod -aG docker \$$USER  then log out and back in (or newgrp docker)"
	@echo "  GPU driver error:           sudo ./scripts/install_nvidia_docker.sh  or  make test-no-gpu"
	@echo "  Unable to open database:   sudo chown -R \$$(id -u):\$$(id -g) data_test cache_test"
	@echo "  Root-owned data/cache:    make fix-data"
	@echo ""
	@echo "  Run  make <target>  or  make help  to show this again."

# Ensure data/cache dirs exist and are owned by current user (avoids root-owned files from containers)
# Pre-create Qdrant Snapshots dir to avoid "Permission denied" when running as non-root
data-dirs:
	@docker run --rm -v "$(CURDIR):/host" -w /host -e HOST_UID=$$(id -u) -e HOST_GID=$$(id -g) alpine sh -c 'mkdir -p data/qdrant/Snapshots cache && chown -R $$HOST_UID:$$HOST_GID data cache' 2>/dev/null && echo "Data directories data and cache are ready." || (mkdir -p data/qdrant/Snapshots cache && echo "Created data and cache. If Qdrant fails with Permission denied, run: make fix-data")

up: docker-check data-dirs
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/docker-compose.yml up --build

down: docker-check
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/docker-compose.yml down

logs: docker-check
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/docker-compose.yml logs -f --tail=100

venv:
	uv venv .venv
	./scripts/ensure_venv_pip.sh .venv
	./scripts/install_requirements.sh requirements/requirements_dev.txt .venv

# Install pip into existing .venv (when uv created it without pip)
venv-pip:
	./scripts/ensure_venv_pip.sh .venv

# Fix ownership of data and cache (run if Qdrant fails with "Permission denied" on Snapshots)
fix-data:
	@echo "Fixing ownership of data/ and cache/..."
	@sudo chown -R $$(id -u):$$(id -g) data cache 2>/dev/null && echo "Done. Run make up again." || echo "Run: sudo chown -R $$(id -u):$$(id -g) data cache"

# Verify Docker daemon is reachable (fixes permission-denied before running test/up)
docker-check:
	@if ! docker info >/dev/null 2>&1; then \
		echo ""; \
		echo "Docker is not accessible (permission denied or daemon not running)."; \
		echo ""; \
		echo "Safe fix: add your user to the docker group:"; \
		echo "  sudo usermod -aG docker $$USER"; \
		echo "Then log out and back in, or in this terminal run: newgrp docker"; \
		echo ""; \
		exit 1; \
	fi
	@echo "Docker access OK."

# Ensure test data dirs exist and are owned by current user (avoids "unable to open database file")
# Uses a one-off container so chown works even when dirs were previously created by Docker as root
test-dirs:
	@docker run --rm -v "$(CURDIR):/host" -w /host -e HOST_UID=$$(id -u) -e HOST_GID=$$(id -g) alpine sh -c 'mkdir -p data_test cache_test && chown $$HOST_UID:$$HOST_GID data_test cache_test' 2>/dev/null && echo "Test directories data_test and cache_test are ready." || (mkdir -p data_test cache_test && echo "Created data_test and cache_test. If api/worker fail with 'unable to open database file', run: sudo chown -R $$(id -u):$$(id -g) data_test cache_test")

# Integration tests (require API + worker + Qdrant). Runs docker-check first. Uses GPU by default.
test: docker-check test-dirs
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.test.yml down --remove-orphans

# Integration tests without GPU (use if NVIDIA Container Toolkit is not installed)
test-no-gpu: docker-check test-dirs
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.no-gpu.yml -f docker/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/docker-compose.yml -f docker/docker-compose.no-gpu.yml -f docker/docker-compose.test.yml down --remove-orphans

# Directory integration test (same as test; set INDEX_DIR_PATH for custom path)
test-dir:
	$(MAKE) test

# Unit tests (no services required). Use .venv if present.
# If you see numpy/opencv import errors, run: make test-unit-no-cv2
test-unit:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m pytest tests/unit/ -v,pytest tests/unit/ -v)

# Unit tests excluding cv2-dependent tests (use when numpy 2.x breaks opencv)
test-unit-no-cv2:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m pytest tests/unit/ -v --ignore=tests/unit/test_frame_extractor.py --ignore=tests/unit/test_heuristics.py,pytest tests/unit/ -v --ignore=tests/unit/test_frame_extractor.py --ignore=tests/unit/test_heuristics.py)

# Lint (requires: pip install ruff)
lint:
	ruff check .
	ruff format --check .
