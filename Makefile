.PHONY: help up down logs data-dirs fix-data env env-interactive venv venv-cuda venv-pip venv-rebuild-xformers docker-check test test-no-gpu test-unit test-unit-no-cv2 test-dir lint cvat-up cvat-down cvat-logs cvat-admin mapper-logs utlz-install utlz utlz-endpoints export-openapi sencoop-up sencoop-up-min sencoop-up-video sencoop-down sencoop-logs sencoop-status sencoop-metrics-up sencoop-release sencoop-release-min sencoop-release-video sencoop-release-metrics sslm sslm-quick sslm-rebuild sslm-run sslm-venv sslm-benchmark sslm-benchmark-quick sslm-dashboard

# Base data directory — overridden by DATA_DIR in .env or shell environment.
DATA_DIR ?= .data
export DATA_DIR

# Default target: show help when no target is given
help:
	@echo "=============================================="
	@echo "  Video Semantic Search - Make targets"
	@echo "=============================================="
	@echo ""
	@echo "  Stack (Docker)"
	@echo "  ---------------"
	@echo "  make up              Start main stack + mapper ICP service (docker-compose.override.yml auto-loaded)"
	@echo "  make sencoop-up         Bootstrap + start sencoop IoT stack (LoRaWAN, Frigate, MQTT)"
	@echo "  make sencoop-down       Stop sencoop IoT stack"
	@echo "  make sencoop-logs       Stream sencoop stack logs"
	@echo "  make sencoop-status     Show sencoop container status and resource usage"
	@echo "  make sencoop-metrics-up      Start sencoop stack + Prometheus / Grafana / cAdvisor (profile: metrics)"
	@echo "  make sencoop-up-min          Start min bundle (LoRaWAN + MQTT only, no video)"
	@echo "  make sencoop-up-video        Start video bundle (MQTT + Frigate only)"
	@echo "  make sencoop-release         Build standard offline bundle (amd64); set VERSION= to tag"
	@echo "  make sencoop-release-min     Build min bundle (no Frigate images)"
	@echo "  make sencoop-release-video   Build video-only bundle"
	@echo "  make sencoop-release-metrics Build standard bundle + Prometheus/Grafana images"
	@echo "  make cvat-up         Start CVAT annotation service (http://localhost:8091)"
	@echo "  make cvat-down       Stop CVAT services"
	@echo "  make cvat-admin      Create CVAT superuser (first-time setup)"
	@echo "  make mapper-logs     Stream mapper (ICP fusion) container logs"
	@echo "  make down            Stop all containers"
	@echo "  make logs            Stream container logs (last 100 lines)"
	@echo "  make docker-check    Check that Docker daemon is reachable (run if you get permission denied)"
	@echo ""
	@echo "  Local dev (venv)"
	@echo "  -----------------"
	@echo "  make env             Generate .data/.env (auto-detects GPU/RAM, picks models)"
	@echo "  make env-interactive Generate .data/.env with interactive prompts (profile, sidecars, models)"
	@echo "  make venv                    Create .venv and install deps; if .venv exists, prompts to recreate or update"
	@echo "  make venv-cuda               Same as venv but forces CUDA wheel install (use if nvidia-smi is absent but GPU present)"
	@echo "  make venv-pip                Install pip into an existing .venv (e.g. after uv venv .venv)"
	@echo "  make venv-rebuild-xformers   Rebuild xformers from source for common GPU arches (RTX 2000/3000/4000, H100)"
	@echo "  make utlz-install            Install optional Utilyze GPU profiler (Linux amd64, NVIDIA Ampere+)"
	@echo "  make utlz                    Run Utilyze with selfsuvis-safe defaults (disables upstream metrics by default)"
	@echo "  make utlz-endpoints          Show Utilyze-discovered inference endpoints per GPU"
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
	@echo "  LLM benchmarks (sslm)"
	@echo "  ----------------------"
	@echo "  make sslm                 Full end-to-end: venv + Open LLM v2 benchmarks (cached)"
	@echo "  make sslm-quick           Same pipeline with quick suite: GSM8K + ARC-C + NQ-Open (~20 min/model, cached)"
	@echo "  make sslm-rebuild         Force-rebuild Docker images then run quick suite"
	@echo "  make sslm-venv            Create .venv-sslm with eval + dashboard extras"
	@echo "  make sslm-benchmark       Run Open LLM Leaderboard v2 (venv must exist)"
	@echo "  make sslm-benchmark-quick Run fast subset: GSM8K + ARC-C + NQ-Open (~20 min/model)"
	@echo "  make sslm-dashboard       Launch Streamlit leaderboard at http://localhost:8501"
	@echo ""
	@echo "  Troubleshooting"
	@echo "  ----------------"
	@echo "  Docker permission denied:  sudo usermod -aG docker \$$USER  then log out and back in (or newgrp docker)"
	@echo "  GPU driver error:           sudo ./scripts/install/install_nvidia_docker.sh  or  make test-no-gpu"
	@echo "  Unable to open database:   sudo chown -R \$$(id -u):\$$(id -g) .data"
	@echo "  Root-owned data:           make fix-data"
	@echo ""
	@echo "  Run  make <target>  or  make help  to show this again."

# Ensure data dirs exist and are owned by current user (avoids root-owned files from containers)
# Pre-create Qdrant Snapshots dir to avoid "Permission denied" when running as non-root
data-dirs:
	@mkdir -p "$(DATA_DIR)/postgres" "$(DATA_DIR)/qdrant/Snapshots" "$(DATA_DIR)/videos" "$(DATA_DIR)/.cache" && chown -R $$(id -u):$$(id -g) "$(DATA_DIR)" 2>/dev/null && echo "Data directories ready ($(DATA_DIR))." || echo "Created $(DATA_DIR). If Qdrant fails with Permission denied, run: make fix-data"

up: docker-check data-dirs
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/core/docker-compose.yml up --build

down: docker-check
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/core/docker-compose.yml down

logs: docker-check
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/core/docker-compose.yml logs -f --tail=100

env:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m selfsuvis.scripts.generate_env --env dev,python -m selfsuvis.scripts.generate_env --env dev)

env-interactive:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m selfsuvis.scripts.generate_env --interactive,python -m selfsuvis.scripts.generate_env --interactive)

venv:
	@if [ -d .venv ]; then \
		printf "\n  .venv already exists.\n"; \
		printf "  [r] Recreate — remove and create a fresh .venv\n"; \
		printf "  [u] Update   — install/upgrade requirements in the existing .venv\n"; \
		printf "\n  Choice [r/u]: "; \
		read choice; \
		case "$$choice" in \
			r|R) \
				echo "Removing existing .venv..."; \
				rm -rf .venv; \
				uv venv .venv; \
				./scripts/install/install_requirements.sh vision,dev .venv \
				;; \
			u|U) \
				echo "Updating requirements in existing .venv..."; \
				./scripts/install/install_requirements.sh vision,dev .venv \
				;; \
			*) \
				echo "Invalid choice '$$choice'. Run  make venv  again and enter r or u."; \
				exit 1 \
				;; \
		esac \
	else \
		uv venv .venv; \
		./scripts/install/install_requirements.sh vision,dev .venv; \
	fi

# Force CUDA torch wheels regardless of nvidia-smi detection (use when GPU is present but nvidia-smi absent)
venv-cuda:
	uv venv .venv
	FORCE_CUDA=1 ./scripts/install/install_requirements.sh vision,dev .venv

# Rebuild xformers from source targeting the GPU present on this machine.
# Auto-detects compute capability via nvidia-smi; falls back to a safe multi-arch
# list (up to sm_90) when no GPU is found, avoiding compute_120 failures on older nvcc.
# Run when python -m xformers.info shows your GPU arch as unavailable.
# Expected build time: 20-60 min.
venv-rebuild-xformers:
	@echo "Rebuilding xformers from source (20-60 min)..."
	@uv pip install --python .venv pip
	@_CC=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '); \
	_ARCH="$${_CC:+$${_CC}+PTX}"; \
	_ARCH="$${_ARCH:-7.5;8.0;8.6;8.9;9.0+PTX}"; \
	echo "  TORCH_CUDA_ARCH_LIST=$${_ARCH}"; \
	TORCH_CUDA_ARCH_LIST="$${_ARCH}" \
	MAX_JOBS=$$(.venv/bin/python src/selfsuvis/scripts/shell_helpers.py max-jobs) \
	.venv/bin/python -m pip install xformers \
	  --no-build-isolation --no-deps --no-binary xformers --force-reinstall --no-cache-dir
	@echo "Done. Verify:  .venv/bin/python -m xformers.info"

# Install pip into existing .venv (when uv created it without pip)
venv-pip:
	uv pip install --python .venv pip

# Fix ownership of data dir (run if Qdrant fails with "Permission denied" on Snapshots)
fix-data:
	@echo "Fixing ownership of $(DATA_DIR)/..."
	@sudo chown -R $$(id -u):$$(id -g) "$(DATA_DIR)" 2>/dev/null && echo "Done. Run make up again." || echo "Run: sudo chown -R $$(id -u):$$(id -g) $(DATA_DIR)"

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
test-dirs:
	@mkdir -p "$(DATA_DIR)/postgres" "$(DATA_DIR)/qdrant/Snapshots" "$(DATA_DIR)/videos" "$(DATA_DIR)/.cache" "$(DATA_DIR)/cache_test" && chown -R $$(id -u):$$(id -g) "$(DATA_DIR)" 2>/dev/null && echo "Test directories ready ($(DATA_DIR))." || echo "Created $(DATA_DIR). If api/worker fail with 'unable to open database file', run: sudo chown -R $$(id -u):$$(id -g) $(DATA_DIR)"

# Integration tests (require API + worker + Qdrant). Runs docker-check first. Uses GPU by default.
test: docker-check test-dirs
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/core/docker-compose.yml -f docker/test/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/core/docker-compose.yml -f docker/test/docker-compose.test.yml down --remove-orphans

# Integration tests without GPU (use if NVIDIA Container Toolkit is not installed)
test-no-gpu: docker-check test-dirs
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/core/docker-compose.yml -f docker/core/docker-compose.no-gpu.yml -f docker/test/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets docker compose -f docker/core/docker-compose.yml -f docker/core/docker-compose.no-gpu.yml -f docker/test/docker-compose.test.yml down --remove-orphans

# Directory integration test (same as test; set INDEX_DIR_PATH for custom path)
test-dir:
	$(MAKE) test

export-openapi:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -c "import json; from selfsuvis.app.main import app; print(json.dumps(app.openapi(), indent=2))" > docs/api/v1-openapi.json,python -c "import json; from selfsuvis.app.main import app; print(json.dumps(app.openapi(), indent=2))" > docs/api/v1-openapi.json)
	@echo "OpenAPI spec written to docs/api/v1-openapi.json"

# Unit tests (no services required). Use .venv if present.
# If you see numpy/opencv import errors, run: make test-unit-no-cv2
test-unit:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m pytest tests/unit/ -v,pytest tests/unit/ -v)

# Unit tests excluding cv2-dependent tests (use when numpy 2.x breaks opencv)
test-unit-no-cv2:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m pytest tests/unit/ -v --ignore=tests/unit/test_frame_extractor.py --ignore=tests/unit/test_heuristics.py,pytest tests/unit/ -v --ignore=tests/unit/test_frame_extractor.py --ignore=tests/unit/test_heuristics.py)

sencoop-up: docker-check
	COMPOSE_PROFILES=lorawan,video ./scripts/sencoop/sencoop-bootstrap.sh up -d

sencoop-up-min: docker-check
	COMPOSE_PROFILES=lorawan ./scripts/sencoop/sencoop-bootstrap.sh up -d

sencoop-up-video: docker-check
	COMPOSE_PROFILES=video ./scripts/sencoop/sencoop-bootstrap.sh up -d

sencoop-down: docker-check
	./scripts/sencoop/sencoop-compose.sh down

sencoop-logs: docker-check
	./scripts/sencoop/sencoop-compose.sh logs -f --tail=100

sencoop-status: docker-check
	./scripts/sencoop/sencoop-compose.sh ps
	@echo ""
	@docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
	  $$(./scripts/sencoop/sencoop-compose.sh ps -q 2>/dev/null) 2>/dev/null || true

sencoop-metrics-up: docker-check
	COMPOSE_PROFILES=lorawan,video,metrics ./scripts/sencoop/sencoop-bootstrap.sh up -d

sencoop-release:
	./scripts/sencoop/sencoop-release.sh --arch amd64 --bundle standard $(if $(VERSION),--version $(VERSION),) --yes

sencoop-release-min:
	./scripts/sencoop/sencoop-release.sh --arch amd64 --bundle min $(if $(VERSION),--version $(VERSION),) --yes

sencoop-release-video:
	./scripts/sencoop/sencoop-release.sh --arch amd64 --bundle video $(if $(VERSION),--version $(VERSION),) --yes

sencoop-release-metrics:
	./scripts/sencoop/sencoop-release.sh --arch amd64 --bundle standard --with-metrics $(if $(VERSION),--version $(VERSION),) --yes

cvat-up: docker-check
	docker compose -f docker/cvat/docker-compose.cvat.yml up -d
	@echo ""
	@echo "CVAT starting at http://localhost:8090"
	@echo "First time? Run: make cvat-admin"

cvat-down: docker-check
	docker compose -f docker/cvat/docker-compose.cvat.yml down

cvat-logs: docker-check
	docker compose -f docker/cvat/docker-compose.cvat.yml logs -f --tail=100

cvat-admin: docker-check
	docker compose -f docker/cvat/docker-compose.cvat.yml exec cvat_server python manage.py createsuperuser

mapper-logs: docker-check
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/core/docker-compose.yml -f docker/core/docker-compose.override.yml logs -f --tail=100 mapper

# Lint (requires: pip install ruff)
lint:
	ruff check .
	ruff format --check .

utlz-install:
	./scripts/install/install_utilyze.sh

utlz:
	./scripts/ssv/ssv-utilyze.sh

utlz-endpoints:
	./scripts/ssv/ssv-utilyze.sh --endpoints

# -- SSLM benchmark playground -------------------------------------------------

define sslm-dashboard-hint
	@echo ""
	@echo "Benchmark complete. Launch the dashboard with:"
	@echo "  make sslm-dashboard"
	@echo "  # or: SSLM_VENV=.venv-sslm .venv-sslm/bin/sslm dashboard --results-dir .data/sslm/results"
endef

# End-to-end: create venv, build sidecar images, run Open LLM v2.
sslm: docker-check
	scripts/sslm/setup-venv.sh eval,dashboard
	scripts/sslm/run-benchmark.sh --suite open_llm_v2
	$(sslm-dashboard-hint)

sslm-quick: docker-check
	scripts/sslm/setup-venv.sh eval,dashboard
	scripts/sslm/run-benchmark.sh --suite reasoning_quick
	$(sslm-dashboard-hint)

sslm-rebuild: docker-check
	scripts/sslm/setup-venv.sh eval,dashboard
	scripts/sslm/run-benchmark.sh --build --suite reasoning_quick
	$(sslm-dashboard-hint)

sslm-run: docker-check
	scripts/sslm/setup-venv.sh eval,dashboard
	scripts/sslm/run-benchmark.sh --suite open_llm_v2
	$(sslm-dashboard-hint)

sslm-venv:
	scripts/sslm/setup-venv.sh eval,dashboard

sslm-benchmark: docker-check
	scripts/sslm/run-benchmark.sh --suite open_llm_v2

sslm-benchmark-quick: docker-check
	scripts/sslm/run-benchmark.sh --suite reasoning_quick

sslm-dashboard:
	scripts/sslm/run-dashboard.sh
