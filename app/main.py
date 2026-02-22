from fastapi import FastAPI, Request

from app.routers.health import router as health_router
from app.routers.index import router as index_router
from app.routers.jobs import router as jobs_router
from app.routers.query import router as query_router

app = FastAPI()


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    return response


app.include_router(health_router)
app.include_router(index_router)
app.include_router(jobs_router)
app.include_router(query_router)
