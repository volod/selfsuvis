import os
from typing import Optional

import requests

from pipeline.config import settings
from pipeline.utils import ensure_dir


def download_url(
    url: str,
    dest_path: str,
    chunk_size: int = 1024 * 1024,
    max_bytes: Optional[int] = None,
) -> None:
    """Download URL to dest_path. Stops after max_bytes if set."""
    max_bytes = max_bytes if max_bytes is not None else settings.MAX_DOWNLOAD_BYTES
    ensure_dir(os.path.dirname(dest_path))
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        content_length = r.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError(
                f"Content-Length {content_length} exceeds max download size {max_bytes}"
            )
        written = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(
                            f"Download exceeded max size {max_bytes} bytes"
                        )
                    f.write(chunk)
