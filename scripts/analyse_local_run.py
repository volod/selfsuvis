#!/usr/bin/env python3
"""Analyse a local pipeline run and produce charts + an HTML report."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project src is on the path when run directly
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __name__ == "__main__":
    from selfsuvis.scripts.analyse_local_run import main

    main()
