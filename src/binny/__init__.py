from __future__ import annotations

from importlib import import_module

_api = import_module("binny.api")

__all__ = list(getattr(_api, "__all__", []))

for name in __all__:
    globals()[name] = getattr(_api, name)
