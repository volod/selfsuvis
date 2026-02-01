import os
import requests

from pipeline.utils import ensure_dir


def download_url(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
    ensure_dir(os.path.dirname(dest_path))
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
