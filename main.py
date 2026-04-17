"""Dev-only shim — only useful when running from the repo root without pip install.

After `pip install -e .` (or `make venv`), use the installed entry point instead:
  selfsuvis [args]          # registered console_script
  python -m selfsuvis [args]  # module invocation
"""
from selfsuvis.cli import main

if __name__ == "__main__":
    main()
