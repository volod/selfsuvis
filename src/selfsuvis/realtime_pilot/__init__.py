"""Backward-compatibility shim: selfsuvis.realtime_pilot -> selfsuvis.realtime.

This package was renamed to ``selfsuvis.realtime`` in 2026-05.
Imports of ``selfsuvis.realtime_pilot`` and all its sub-modules are transparently
redirected to ``selfsuvis.realtime``.  Existing code continues to work.

Migration:  s/selfsuvis\\.realtime_pilot/selfsuvis.realtime/g
"""

import importlib
import sys
import warnings


def _warn() -> None:
    warnings.warn(
        "selfsuvis.realtime_pilot is deprecated; use selfsuvis.realtime instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class _RealtimePilotRedirector:
    _PREFIX = "selfsuvis.realtime_pilot"
    _TARGET = "selfsuvis.realtime"

    def find_module(self, name: str, path=None):
        if name == self._PREFIX or name.startswith(self._PREFIX + "."):
            return self
        return None

    def load_module(self, name: str):
        new_name = name.replace(self._PREFIX, self._TARGET, 1)
        _warn()
        module = importlib.import_module(new_name)
        sys.modules[name] = module
        return module

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


_redirector = _RealtimePilotRedirector()
if not any(isinstance(f, _RealtimePilotRedirector) for f in sys.meta_path):
    sys.meta_path.append(_redirector)

import selfsuvis.realtime as _realtime  # noqa: E402
sys.modules[__name__] = _realtime
