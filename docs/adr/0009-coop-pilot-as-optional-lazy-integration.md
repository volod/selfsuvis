# ADR-0009: Keep coop as an Optional, Lazy-Started Integration Layer

Date: 2026-05-02  
Status: Accepted

## Context

The repository serves two related but different deployment shapes:
- the core selfsuvis indexing and retrieval stack
- the coop edge-monitoring stack built around MQTT, Frigate, MediaMTX, and site
  state synthesis

The core API must still start and function when coop-specific dependencies,
brokers, or cameras are absent.

## Decision

Treat `coop` as an optional integration layer that is imported and started
lazily from the main app lifecycle.

Current implementation:
- startup wiring in `src/selfsuvis/app/main.py`
- background lifecycle management in `src/selfsuvis/app/services/coop_streams.py`
- coop runtime under `src/selfsuvis/coop/`

Startup failures in MQTT, Frigate discovery, or optional coop modules are logged
and do not block the base API process.

## Consequences

Positive:
- One repository and API can support both plain selfsuvis and coop deployments
- Developers can run the core stack without the full IoT edge environment
- coop services can evolve without turning every install into a hard dependency

Trade-offs:
- Optional imports and soft-failure behavior increase startup complexity
- Runtime capability depends on environment shape, not just installed code
- Operators need clear observability to distinguish “feature disabled” from
  “feature failed to start”
