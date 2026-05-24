"""Backward-compatibility shim: selfsuvis.coop_pilot -> selfsuvis.coop.

This package was renamed to ``selfsuvis.coop`` in 2026-05.
Imports of ``selfsuvis.coop_pilot`` and all its sub-modules are transparently
redirected to ``selfsuvis.coop``.  Existing code continues to work.

Migration:  s/selfsuvis\\.coop_pilot/selfsuvis.coop/g
"""

import importlib
import sys
import warnings


def _warn() -> None:
    warnings.warn(
        "selfsuvis.coop_pilot is deprecated; use selfsuvis.coop instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class _CoopPilotRedirector:
    """Meta-path finder that maps selfsuvis.coop_pilot.* imports to selfsuvis.coop.*"""

    _PREFIX = "selfsuvis.coop_pilot"
    _TARGET = "selfsuvis.coop"

    def find_module(self, name: str, path=None):  # Python <3.12 API
        if name == self._PREFIX or name.startswith(self._PREFIX + "."):
            return self
        return None

    def load_module(self, name: str):
        new_name = name.replace(self._PREFIX, self._TARGET, 1)
        _warn()
        module = importlib.import_module(new_name)
        sys.modules[name] = module
        return module

    # Python 3.4+ spec-based API (preferred)
    def find_spec(self, name: str, path, target=None):
        if name == self._PREFIX or name.startswith(self._PREFIX + "."):
            new_name = name.replace(self._PREFIX, self._TARGET, 1)
            _warn()
            spec = importlib.util.find_spec(new_name)
            if spec is not None:
                spec.name = name
                sys.modules[name] = importlib.import_module(new_name)
            return spec
        return None


_redirector = _CoopPilotRedirector()
if not any(isinstance(f, _CoopPilotRedirector) for f in sys.meta_path):
    sys.meta_path.append(_redirector)

# Make this module itself behave as selfsuvis.coop
import selfsuvis.coop as _coop  # noqa: E402
sys.modules[__name__] = _coop
