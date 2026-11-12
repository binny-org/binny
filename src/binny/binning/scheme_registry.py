"""Registry for tomographic bin-edge generation schemes."""

from __future__ import annotations

from collections.abc import Callable

from binny.utils.types import FloatArray

BinningFunction = Callable[..., FloatArray]

_SCHEMES: dict[str, BinningFunction] = {}


def register_binning_scheme(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> Callable[[BinningFunction], BinningFunction]:
    """Register a bin-edge generation function under one or more names."""

    def decorator(func: BinningFunction) -> BinningFunction:
        names = (name, *aliases)

        for item in names:
            key = item.strip().lower()
            if not key:
                raise ValueError("Binning scheme names must not be empty.")
            if key in _SCHEMES:
                raise ValueError(f"Binning scheme {key!r} is already registered.")

        for item in names:
            _SCHEMES[item.strip().lower()] = func

        return func

    return decorator


def get_binning_scheme(name: str) -> BinningFunction:
    """Retrieve a registered bin-edge generation function."""
    key = name.strip().lower()

    try:
        return _SCHEMES[key]
    except KeyError as error:
        raise ValueError(f"Unsupported binning_scheme: {name!r}.") from error


def available_binning_schemes() -> tuple[str, ...]:
    """Return registered binning scheme names and aliases."""
    return tuple(sorted(_SCHEMES))
