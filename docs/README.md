# selfsuvis — Documentation

selfsuvis is a three-playground outdoor autonomy perception stack.
See [README.md](../README.md) for the project overview and three-playground architecture.

---

## Quick navigation

### Getting started
| | |
|---|---|
| [Quick start](quickstart/quickstart.md) | Run any stack with Docker or locally |
| [Setup](quickstart/setup.md) | GPU setup, CVAT, system deps |
| [Local learning path](quickstart/local_path.md) | 36-step pipeline walkthrough |
| [Learning path deep dives](learning_path/README.md) | Per-phase study guides |
| [Coop getting started](sencoop/getting-started.md) | IoT sensor mesh setup |

### Reference
| | |
|---|---|
| [Architecture](reference/architecture.md) | System topology and service map |
| [Pipeline reference](reference/pipeline.md) | 36-step data flow |
| [API reference](reference/api.md) | HTTP endpoints |
| [Configuration](reference/configuration.md) | All env vars |
| [Secrets management](reference/secrets-management.md) | Secrets separation and rotation |
| [Data layout](reference/data_layout.md) | Output directory structure |
| [Model catalog](reference/model-catalog.md) | VRAM budgets, model options, SSL |
| [Analytics](reference/analytics.md) | Post-run reports and visualization |
| [Performance](reference/performance.md) | Latency and tuning |

### Development
| | |
|---|---|
| [Examples](development/examples.md) | Code examples: DAE, anomaly, active learning |
| [Developer guide](development/develop.md) | Running, testing, contributing |
| [Tests](development/tests.md) | Unit and integration test guide |
| [Runbooks](runbooks/README.md) | Per-component runbooks |

### Operations
| | |
|---|---|
| [Troubleshooting](operations/troubleshooting.md) | Common errors and fixes |
| [Operations guide](operations/operations.md) | Deployment constraints |

### Implementation state
| | |
|---|---|
| [Current implementation](impl/current.md) | What exists, where it lives, how flows run |
| [Implementation plan](impl/plan.md) | Forward work: scope, execution plan, task specs |

### Design history
| | |
|---|---|
| [Architecture decisions](adr/README.md) | ADR log |
| [Coop docs](sencoop/getting-started.md) | IoT sensor mesh |
