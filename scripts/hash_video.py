import sys
import logging
from pipeline.utils import file_sha256

if len(sys.argv) < 2:
    logging.basicConfig(level="INFO")
    logging.getLogger(__name__).info("Usage: python scripts/hash_video.py /path/to/video")
    raise SystemExit(1)

logging.basicConfig(level="INFO")
logging.getLogger(__name__).info("sha256=%s", file_sha256(sys.argv[1]))
