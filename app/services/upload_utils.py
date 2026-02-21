import hashlib
import io
from typing import Tuple

from fastapi import UploadFile


async def read_upload_limited(file: UploadFile, max_bytes: int, chunk_size: int = 1024 * 1024) -> bytes:
    buf = io.BytesIO()
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Upload exceeds max size {max_bytes} bytes")
        buf.write(chunk)
    return buf.getvalue()


async def hash_upload_limited(file: UploadFile, max_bytes: int, chunk_size: int = 1024 * 1024) -> Tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Upload exceeds max size {max_bytes} bytes")
        h.update(chunk)
    return h.hexdigest(), total
