# Current Implementation

This index is for agents and maintainers who need the current implementation shape: what exists,
where it lives, how the major flows run, and why the important design choices were made.

The system is three interconnected playgrounds (see the root [README](../../README.md)):
a production server that answers queries, a local research pipeline that builds world-model
understanding, and an IoT sensor mesh that collects ground truth from the physical world.
The mesh collects, the pipeline understands, the server serves.

## Topic Map

| Need | Read |
| --- | --- |
| API, worker, UI, storage, query surface, v1 ops API, realtime bridges | [current/production-server.md](current/production-server.md) |
| 36-step local research pipeline (`ssv_vdp`): phases, steps, SSL, threat layer, artifacts | [current/local-pipeline.md](current/local-pipeline.md) |
| IoT sensor mesh (`sencoop`): MQTT/ChirpStack/Frigate/OpenRemote stack, site state, threat feed, and its deliberate gaps | [current/sencoop-mesh.md](current/sencoop-mesh.md) |
| Environments, heavy native builds, Docker composition, tests, CI workflows | [current/build-ci-test.md](current/build-ci-test.md) |
| Config facade, `.env` generation, `$DATA_DIR` layout rules, secrets | [current/data-config.md](current/data-config.md) |
| Model playgrounds: nanochat (LLM training), sslm (reasoning benchmarks) | [current/model-playgrounds.md](current/model-playgrounds.md) |

## Scope of these documents

- **Current behavior and durable decisions only.** Forward work lives in
  [plan.md](plan.md) and moves here when it ships.
- Topic files cite real module paths and commands so an agent can verify claims
  against the tree instead of trusting prose.
- Design history stays in `docs/adr/`; operator guides stay under `docs/`
  (reference, quickstart, sencoop, runbooks). These impl docs link out rather
  than duplicate.

## Maintenance

When a forward task from [plan.md](plan.md) lands: move its durable behavior into
the matching `current/*.md` topic (create a new topic file only when none fits),
update the Topic Map row if the summary changed, and delete the task from the plan.
Keep every claim verifiable: path, command, or endpoint.
