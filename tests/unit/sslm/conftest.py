from __future__ import annotations

import sys
from pathlib import Path

SSLM_ROOT = Path(__file__).resolve().parents[3] / "src" / "sslm"
sys.path.insert(0, str(SSLM_ROOT))
