from fastapi import Request
from fastapi.responses import HTMLResponse

from app.routers.admin import router as admin_router
from app.routers.cvat import cvat_admin_router, webhook_router
from app.routers.health import router as health_router
from app.routers.index import router as index_router
from app.routers.jobs import router as jobs_router
from app.routers.query import router as query_router
from app.routers.robot import router as robot_router
from app.services.form_templates import get_index_form_html

app = FastAPI()


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
app.include_router(robot_router)
app.include_router(webhook_router)


@app.get("/index/form", response_class=HTMLResponse, include_in_schema=False)
async def index_form():
    """Simple HTML form to upload a local video or submit a URL for indexing."""
    return get_index_form_html()
