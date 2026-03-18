# Video Semantic Search (POC)

Semantic search over video content: 
index videos, then search by text or image. 
Uses OpenCLIP for embeddings and Qdrant for vector storage.

## Quick start

```bash
make up
```

Then open the Streamlit UI (default: http://localhost:8501), upload a video or provide a URL, start indexing, and run text or image queries.

## Docs
- [Overview](docs/overview.md)
- [Setup](docs/setup.md)
- [Develop](docs/develop.md)
- [API](docs/api.md)
- [UI](docs/ui.md)
- [Helpers](docs/helpers.md)
- [Configuration](docs/configuration.md)
- [Pipeline](docs/pipeline.md)
- [Architecture](docs/architecture.md)
- [Examples](docs/examples.md)
- [Data layout](docs/data_layout.md)
- [Performance](docs/performance.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Licensing](docs/licensing.md)
- [Tests](docs/tests.md)

## Sources 
- [build-your-own-x](https://github.com/codecrafters-io/build-your-own-x)

- [dinov3 github](https://github.com/facebookresearch/dinov3)
- [dinov3 paper](https://ai.meta.com/research/publications/dinov3/)

- [self-supervised pretraining](https://arxiv.org/html/2502.11831v1)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [IntPhys 2](https://arxiv.org/abs/2506.09849)

# Skils
/office-hours → /plan-ceo-review → /plan-eng-review → [build] → /review → /qa → /ship
