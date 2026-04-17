from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from selfsuvis.app.db import close_db_pool, init_db_pool
from selfsuvis.app.routers.admin import router as admin_router
from selfsuvis.app.routers.cvat import cvat_admin_router, webhook_router
from selfsuvis.app.routers.health import router as health_router
from selfsuvis.app.routers.index import router as index_router
from selfsuvis.app.routers.jobs import router as jobs_router
from selfsuvis.app.routers.query import router as query_router
from selfsuvis.app.routers.realtime import router as realtime_router
from selfsuvis.app.routers.robot import router as robot_router
from selfsuvis.app.routers.scene import router as scene_router
from selfsuvis.app.services.form_templates import get_index_form_html

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down shared resources."""
    await init_db_pool(app)
    try:
        yield
    finally:
        await close_db_pool(app)


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    return response


app.include_router(admin_router)
app.include_router(cvat_admin_router)
app.include_router(health_router)
app.include_router(index_router)
app.include_router(jobs_router)
app.include_router(query_router)
app.include_router(realtime_router)
app.include_router(robot_router)
app.include_router(scene_router)
app.include_router(webhook_router)


@app.get("/index/form", response_class=HTMLResponse, include_in_schema=False)
async def index_form():
    """Simple HTML form to upload a local video or submit a URL for indexing."""
    return get_index_form_html()
