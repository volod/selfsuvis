#!/usr/bin/env python3
"""Convenience wrapper for the packaged Florence smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __name__ == "__main__":
    from selfsuvis.scripts.smoke_florence import main

    main()
