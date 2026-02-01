# Setup

## Prerequisites
- Docker + Docker Compose
- NVIDIA Container Toolkit for GPU access (optional but recommended)

## Start services
```bash
make up
```

If you need GPU inside containers:
```bash
docker compose up --build --remove-orphans
```

## Open UI
- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000
