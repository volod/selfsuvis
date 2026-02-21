from fastapi import FastAPI

from app.routers.health import router as health_router
from app.routers.index import router as index_router
from app.routers.jobs import router as jobs_router
from app.routers.query import router as query_router

app = FastAPI()
app.include_router(health_router)
app.include_router(index_router)
app.include_router(jobs_router)
app.include_router(query_router)
