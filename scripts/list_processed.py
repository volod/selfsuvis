import sqlite3
import os
import json
import logging

DB_PATH = os.getenv("PROCESSED_DB", "./data/processed.db")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT * FROM processed ORDER BY updated_at DESC LIMIT 200").fetchall()
for row in rows:
    item = dict(row)
    item["meta_json"] = json.loads(item.get("meta_json") or "{}")
    logger.info("%s", json.dumps(item, indent=2))
