#!/usr/bin/env python3
import asyncio
import json
import logging
import os

import asyncpg

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis",
)


async def main() -> None:
    conn = await asyncpg.connect(DATABASE_URL, timeout=5)
    try:
        rows = await conn.fetch(
            """
            SELECT file_hash, video_id, path, size_bytes, mtime, status, meta_json, created_at, updated_at
            FROM processed_files
            ORDER BY updated_at DESC
            LIMIT 200
            """
        )
    finally:
        await conn.close()

    for row in rows:
        item = dict(row)
        if isinstance(item.get("meta_json"), str):
            item["meta_json"] = json.loads(item["meta_json"])
        logger.info("%s", json.dumps(item, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
