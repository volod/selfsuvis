# Testing Guide

This guide covers how to run tests and validate your Stack A Pilot deployment.

## Prerequisites

Set up the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Health checks only
pytest tests/test_stack_health.py -v

# MQTT tests only
pytest tests/test_mqtt.py -v

# With coverage report
pytest tests/ --cov=coop_stack_analytics --cov-report=html
```

## Test Categories

### Container Health Tests (`test_stack_health.py`)

Tests verify:

- **All containers running** - Checks all expected containers exist
- **Container health status** - Verifies healthcheck status
- **Restart count** - Ensures containers aren't crash-looping
- **Service endpoints** - HTTP accessibility of web interfaces
- **Database connectivity** - PostgreSQL and Redis connections

Example output:
```
tests/test_stack_health.py::TestContainerHealth::test_all_containers_running PASSED
tests/test_stack_health.py::TestContainerHealth::test_containers_healthy PASSED
tests/test_stack_health.py::TestServiceEndpoints::test_chirpstack_ui_accessible PASSED
tests/test_stack_health.py::TestDatabaseConnectivity::test_redis_accessible PASSED
```

### MQTT Tests (`test_mqtt.py`)

Tests verify:

- **Health user connection** - Basic MQTT connectivity
- **ChirpStack user connection** - Service account access
- **ACL enforcement** - Topic access control
- **Message throughput** - Performance baseline

Example:
```
tests/test_mqtt.py::TestMQTTConnectivity::test_health_user_connection PASSED
tests/test_mqtt.py::TestMQTTTopicACL::test_health_user_can_read_sys_topics PASSED
tests/test_mqtt.py::TestMQTTPerformance::test_message_throughput PASSED
```

## Load Testing

### Using Locust

Run load tests against stack services:

```bash
# Interactive mode (web UI at http://localhost:8089)
locust -f tests/locustfile.py

# Headless mode
locust -f tests/locustfile.py --headless -u 10 -r 2 -t 60s
```

Parameters:
- `-u 10` - 10 concurrent users
- `-r 2` - Spawn 2 users per second
- `-t 60s` - Run for 60 seconds

### Load Test Targets

The locustfile includes test users for:

- **ChirpStackUser** - Tests ChirpStack REST API
- **OpenRemoteUser** - Tests OpenRemote Manager API
- **FrigateUser** - Tests Frigate NVR API

### Custom Load Tests

Create your own Locust file:

```python
from locust import HttpUser, task, between

class MyUser(HttpUser):
    host = "http://localhost:8080"
    wait_time = between(1, 3)

    @task
    def my_test(self):
        self.client.get("/api/endpoint")
```

## Manual Testing

### Test MQTT Connectivity

```bash
# From inside the Mosquitto container
docker exec coop-mosquitto mosquitto_sub \
  -h 127.0.0.1 -p 1883 \
  -t '$SYS/broker/version' -C 1 \
  -u health -P 'your-password'
```

### Test ChirpStack API

```bash
curl http://localhost:8080/api/internal/login
curl http://localhost:8090/api/devices
```

### Test Database

```bash
# ChirpStack PostgreSQL
docker exec coop-cs-postgres pg_isready -U chirpstack

# Redis
docker exec coop-cs-redis redis-cli ping
```

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Stack Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements/test.txt

      - name: Start stack
        run: |
          ./scripts/coop/coop-env.sh test
          APP_ENV=test ./scripts/coop/coop-bootstrap.sh
          sleep 60  # Wait for services

      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest tests/ -v --tb=short
```

## Troubleshooting Tests

### Tests Timing Out

Increase timeout in `tests/pytest.ini`:
```ini
timeout = 120
```

Or per-test:
```python
@pytest.mark.timeout(120)
def test_slow_operation():
    ...
```

### Connection Refused Errors

Check if services are healthy:
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### MQTT Authentication Failures

Verify passwords match between `.env` and `config/coop/mosquitto/pwfile`:
```bash
# Regenerate password file
./scripts/coop/coop-mqtt-users.sh
docker compose restart mosquitto
```
