# Troubleshooting Guide

Common issues and solutions for Stack A Pilot.

## Container Issues

### Container Won't Start

**Symptoms:** Container stuck in "Created" or "Restarting" state.

**Solutions:**

1. Check logs:
   ```bash
   docker logs <container-name>
   ```

2. Check dependencies are healthy:
   ```bash
   docker ps --format "table {{.Names}}\t{{.Status}}"
   ```

3. Verify configuration files exist and are readable:
   ```bash
   ls -la config/<service>/
   ```

4. Ensure data directories exist (bind mounts):
   ```bash
   ./scripts/ensure_data_dirs.sh
   ```

5. If containers fail with "Operation not permitted" or "Unable to open" on data dirs:
   - PUID/PGID are set dynamically by `compose.sh` from your user
   - For a fresh start: `./scripts/coop/clean_data.sh`, then `./scripts/coop/bootstrap.sh`

6. If you cannot remove `./data/*` (permission denied):
   - `./scripts/coop/clean_data.sh` fixes ownership and removes data

### Container Keeps Restarting

**Symptoms:** Container shows "Restarting (X)" status.

**Common causes:**

1. **Configuration errors** - Check logs for syntax errors
2. **Missing files** - Ensure all required config files exist
3. **Permission issues** - Check file permissions
4. **Resource limits** - Container may be OOM killed

**Solutions:**

```bash
# Check exit code
docker inspect <container> --format='{{.State.ExitCode}}'

# Check OOM kill
docker inspect <container> --format='{{.State.OOMKilled}}'

# View last logs
docker logs --tail 50 <container>
```

### Health Check Failures

**Symptoms:** Container is "unhealthy".

**Solutions:**

1. Run health check manually:
   ```bash
   docker exec <container> <healthcheck-command>
   ```

2. Check service is actually running inside container:
   ```bash
   docker exec <container> ps aux
   ```

## Mosquitto MQTT Issues

### "per_listener_settings must be set before any other security settings"

**Cause:** Mosquitto 2.x changed config ordering requirements.

**Solution:** Ensure `per_listener_settings` comes before `allow_anonymous` in `mosquitto.conf`:
```
# This is deprecated in 2.x - use per-listener settings instead
listener 1883
allow_anonymous false
password_file /mosquitto/config/pwfile
```

### "Permission denied" on config files (mosquitto exit 13)

**Cause:** Mosquitto runs as non-root (PUID:PGID). Config, pwfile, and certs must be readable by that user.

**Solution:**

1. Run bootstrap (creates TLS and pwfile as your user when missing):
   ```bash
   ./scripts/coop/bootstrap.sh
   ```

2. If data dirs have wrong ownership, run:
   ```bash
   ./scripts/ensure_data_dirs.sh
   ```

3. Ensure TLS and pwfile were created by your user (not sudo):
   - `./scripts/gen_mosquitto_selfsigned_tls.sh` - run as normal user
   - `./scripts/coop/init_mosquitto_users.sh` - creates pwfile as your user

### MQTT Authentication Failures

**Symptoms:** Clients can't connect, "not authorized" errors.

**Solutions:**

1. Verify password file exists:
   ```bash
   ls -la config/coop/mosquitto/pwfile
   ```

2. Regenerate passwords:
   ```bash
   ./scripts/coop/init_mosquitto_users.sh
   docker compose restart mosquitto
   ```

3. Check ACL file for topic permissions:
   ```bash
   cat config/coop/mosquitto/aclfile
   ```

## ChirpStack Issues

### ChirpStack Can't Connect to MQTT

**Symptoms:** ChirpStack logs show MQTT connection errors.

**Solutions:**

1. Verify MQTT credentials in `.env` match `pwfile`
2. Check `chirpstack.toml` has correct MQTT settings:
   ```toml
   [integration.mqtt]
   server = "tcp://mosquitto:1883/"
   username = "$CHIRPSTACK_MQTT_USERNAME"
   password = "$CHIRPSTACK_MQTT_PASSWORD"
   ```

3. Ensure Mosquitto is healthy before ChirpStack starts

### Database Connection Errors

**Cause:** PostgreSQL not ready when ChirpStack starts.

**Solution:** Docker Compose handles this with `depends_on: condition: service_healthy`. If issues persist:
```bash
docker compose restart chirpstack
```

## Frigate Issues

### "Read-only file system" Errors

**Cause:** Config directory mounted as read-only but Frigate needs write access.

**Solution:** Remove `:ro` from volume mount in `docker/docker-compose.yml`:
```yaml
volumes:
  - ./config/coop/frigate:/config  # Not :ro
```

### MQTT Password Not Working

**Cause:** Frigate doesn't support environment variable substitution in config.

**Solution:** Set password directly in `config/coop/frigate/config.yml`:
```yaml
mqtt:
  password: "actual-password-here"
```

Or use the setup script to inject it:
```bash
PASS=$(grep FRIGATE_MQTT_PASSWORD .env | cut -d= -f2)
sed -i "s/{FRIGATE_MQTT_PASSWORD}/$PASS/" config/coop/frigate/config.yml
```

### Camera Connection Failures

**Symptoms:** Camera shows offline or no video.

**Solutions:**

1. Test RTSP URL directly:
   ```bash
   ffprobe rtsp://user:pass@ip:554/stream
   ```

2. Check network connectivity from container:
   ```bash
   docker exec coop-frigate ping camera-ip
   ```

3. Verify camera credentials in config

## OpenRemote Issues

### Keycloak Health Check Fails

**Symptoms:** Keycloak shows "unhealthy", Manager can't start.

**Solution:** OpenRemote's Keycloak uses `/auth` path. Verify healthcheck:
```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -s http://127.0.0.1:8080/auth | grep -q . || exit 1"]
```

### Manager Startup Takes Forever

**Cause:** Manager waits for Keycloak, which waits for PostgreSQL.

**Solution:** Increase `start_period` in healthchecks:
```yaml
healthcheck:
  start_period: 120s
```

### SSL Certificate Errors

**Symptoms:** Browser shows certificate warning.

**Solutions:**

1. For development, accept self-signed cert
2. For production, set `OR_EMAIL_ADMIN` for Let's Encrypt
3. Check domain resolves correctly: `nslookup $OR_HOSTNAME`

## Database Issues

### PostgreSQL "too many connections"

**Solution:** Increase connection limit in `.env`:
```
OR_POSTGRES_MAX_CONNECTIONS=300
```

### Redis Memory Issues

**Solution:** Add memory limit to Redis config:
```bash
docker exec coop-cs-redis redis-cli CONFIG SET maxmemory 256mb
```

## Network Issues

### Containers Can't Reach Each Other

**Solutions:**

1. Check all containers are on same network:
   ```bash
   docker network inspect coop-stack-a-net
   ```

2. Use container names (not localhost) for inter-service communication

3. Restart Docker networking:
   ```bash
   docker compose down
   docker network prune
   docker compose up -d
   ```

### Port Conflicts

**Symptoms:** "port is already allocated" error.

**Solution:** Change port in `.env`:
```
CHIRPSTACK_UI_PORT=8081  # Instead of 8080
```

## Resource Issues

### Out of Memory (OOM)

**Symptoms:** Containers killed randomly, system slow.

**Solutions:**

1. Check system memory:
   ```bash
   free -h
   ```

2. Reduce container limits in `docker/docker-compose.yml`

3. Add swap space:
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Disk Space

**Symptoms:** Containers fail to start, write errors.

**Solutions:**

1. Check disk space:
   ```bash
   df -h
   ```

2. Clean Docker resources:
   ```bash
   docker system prune -a --volumes
   ```

3. Rotate/truncate logs:
   ```bash
   truncate -s 0 /var/lib/docker/containers/*/*-json.log
   ```

## Getting Help

If issues persist:

1. Collect diagnostics:
   ```bash
   docker compose logs > stack-logs.txt
   python -m coop_stack_analytics.cli --format json --output diagnostics.json
   ```

2. Check versions:
   ```bash
   docker --version
   docker compose version
   python --version
   ```

3. Review container events:
   ```bash
   docker events --since 1h --filter 'type=container'
   ```
