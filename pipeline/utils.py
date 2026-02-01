import hashlib
import os
import time
from typing import Dict, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stable_point_id(*parts: Any) -> int:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int(h.hexdigest()[:16], 16)


def now_ts() -> float:
    return time.time()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


class RateTimer:
    def __init__(self):
        self.start = time.time()
        self.count = 0

    def tick(self, n: int = 1) -> None:
        self.count += n

    def rate(self) -> float:
        elapsed = time.time() - self.start
        if elapsed <= 0:
            return 0.0
        return self.count / elapsed

