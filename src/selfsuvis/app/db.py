"""Database pool utilities for FastAPI app lifecycle."""

from __future__ import annotations

from typing import Optional
import asyncpg
from fastapi import HTTPException, Request
from fastapi import FastAPI

from selfsuvis.pipeline.core import get_logger, settings

logger = get_logger(__name__)


async def init_db_pool(app: FastAPI) -> None:
    """Initialize asyncpg pool and attach it to app state."""
    db_url = settings.DATABASE_URL
    if not db_url:
        app.state.db_pool = None
        logger.warning("DATABASE_URL not configured; API DB operations unavailable")
        return
    app.state.db_pool = await asyncpg.create_pool(
        dsn=db_url,
        min_size=1,
        max_size=10,
        timeout=10,
    )


async def close_db_pool(app: FastAPI) -> None:
    """Close asyncpg pool if it exists."""
    pool: Optional[asyncpg.Pool] = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()


def get_db_pool(request: Request) -> asyncpg.Pool:
    """Return DB pool from request app state or raise 503."""
    pool: Optional[asyncpg.Pool] = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")
    return pool


def get_db_pool_optional(request: Request) -> Optional[asyncpg.Pool]:
    """Return DB pool from request app state, or None if not configured."""
    return getattr(request.app.state, "db_pool", None)
