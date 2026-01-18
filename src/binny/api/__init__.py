"""Initialization of the Binny API package."""

from __future__ import annotations

from importlib import import_module

_API_MODULES = ("binny.api.nz_tomography",)

__all__: list[str] = []

for mod_name in _API_MODULES:
    mod = import_module(mod_name)
    names = list(getattr(mod, "__all__", []))
    for name in names:
        globals()[name] = getattr(mod, name)
    __all__.extend(names)
