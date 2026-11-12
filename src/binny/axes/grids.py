"""Utilities for constructing one-dimensional sampling grids."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from binny.axes.bin_edges import log_edges
from binny.utils.validators import validate_grid_spec

__all__ = [
    "linear_grid",
    "log_grid",
    "power_grid",
    "chebyshev_grid",
    "sinh_grid",
    "register_grid",
    "grid",
]

_GRID_REGISTRY: dict[str, Callable[..., np.ndarray]] = {}

_GRID_ALIASES = {
    "lin": "linear",
    "uniform": "linear",
    "log_grid": "log",
    "logarithmic": "log",
    "geom": "log",
    "geometric": "log",
}


def register_grid(name: str) -> Callable:
    """Register a custom grid builder under a case-insensitive name.

    Args:
        name: Unique name identifying the grid strategy.

    Returns:
        Decorator that registers the grid builder.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If the name is empty or already registered.
    """
    if not isinstance(name, str):
        raise TypeError("Grid name must be a string.")

    key = name.lower()

    if not key or key in _GRID_REGISTRY or key in _GRID_ALIASES:
        raise ValueError(f"Invalid or already registered grid name {name!r}.")

    def decorator(func: Callable) -> Callable:
        if not callable(func):
            raise TypeError("Grid builder must be callable.")
        if key in _GRID_REGISTRY:
            raise ValueError(f"Grid kind {name!r} is already registered.")
        _GRID_REGISTRY[key] = func
        return func

    return decorator


@register_grid("linear")
def linear_grid(start: float, stop: float, n: int) -> np.ndarray:
    """Construct a linearly spaced grid including both endpoints.

    Args:
        start: Lower endpoint.
        stop: Upper endpoint.
        n: Number of grid points, at least 2.

    Returns:
        One-dimensional array of equally spaced points.
    """
    x0, x1, n_int = validate_grid_spec(start, stop, n, log=False)
    return np.linspace(x0, x1, n_int, dtype=np.float64)


@register_grid("log")
def log_grid(start: float, stop: float, n: int) -> np.ndarray:
    """Construct a logarithmically spaced grid including both endpoints.

    Args:
        start: Positive lower endpoint.
        stop: Upper endpoint, greater than ``start``.
        n: Number of grid points, at least 2.

    Returns:
        One-dimensional array of logarithmically spaced points.
    """
    x0, x1, n_int = validate_grid_spec(start, stop, n, log=True)
    return log_edges(x0, x1, n_int - 1).astype(np.float64, copy=False)


@register_grid("power")
def power_grid(
    start: float,
    stop: float,
    n: int,
    *,
    power: float = 2.0,
) -> np.ndarray:
    """Construct a power-law stretched grid.

    Grid points follow ``x = start + (stop - start) * t**power``,
    where ``t`` is uniformly spaced between zero and one.

    Args:
        start: Lower endpoint.
        stop: Upper endpoint.
        n: Number of grid points, at least 2.
        power: Positive stretching exponent. Values greater than one
            cluster points near ``start``, while values below one
            cluster points near ``stop``. A value of one is linear.

    Returns:
        One-dimensional array of power-law spaced points.

    Raises:
        ValueError: If ``power`` is not positive and finite.
    """
    x0, x1, n_int = validate_grid_spec(start, stop, n, log=False)

    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be positive and finite.")

    t = np.linspace(0.0, 1.0, n_int)
    return x0 + (x1 - x0) * t**power


@register_grid("chebyshev")
def chebyshev_grid(start: float, stop: float, n: int) -> np.ndarray:
    """Construct Chebyshev-Lobatto nodes including both endpoints.

    Nodes cluster near both endpoints, making them useful for
    polynomial interpolation and spectral methods.

    Args:
        start: Lower endpoint.
        stop: Upper endpoint.
        n: Number of grid points, at least 2.

    Returns:
        One-dimensional array of Chebyshev-Lobatto nodes.
    """
    x0, x1, n_int = validate_grid_spec(start, stop, n, log=False)

    theta = np.linspace(0.0, np.pi, n_int)
    return x0 + 0.5 * (x1 - x0) * (1.0 - np.cos(theta))


@register_grid("sinh")
def sinh_grid(
    start: float,
    stop: float,
    n: int,
    *,
    strength: float = 3.0,
    concentration: Literal["midpoint", "endpoints"] = "midpoint",
) -> np.ndarray:
    """Construct a sinh-based grid with configurable point concentration.

    The transformation is symmetric around the interval midpoint.
    The ``concentration`` parameter selects whether points cluster
    near the midpoint or near both endpoints.

    Larger ``strength`` values increase the concentration.
    Zero strength produces linear spacing in either mode.

    Args:
        start: Lower endpoint.
        stop: Upper endpoint.
        n: Number of grid points, at least 2.
        strength: Nonnegative stretching parameter.
        concentration: Where grid points are concentrated.
            Either ``"midpoint"`` or ``"endpoints"``.

    Returns:
        One-dimensional array of strictly increasing grid points.

    Raises:
        ValueError: If ``strength`` or ``concentration`` is invalid.
    """
    x0, x1, n_int = validate_grid_spec(start, stop, n, log=False)

    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be nonnegative and finite.")

    if concentration not in {"midpoint", "endpoints"}:
        raise ValueError("concentration must be 'midpoint' or 'endpoints'.")

    if strength == 0:
        return np.linspace(x0, x1, n_int, dtype=np.float64)

    t = np.linspace(-1.0, 1.0, n_int)

    if concentration == "midpoint":
        u = np.sinh(strength * t) / np.sinh(strength)
    else:
        u = np.arcsinh(np.sinh(strength) * t) / strength

    return x0 + 0.5 * (x1 - x0) * (u + 1.0)


def grid(kind: str, /, **kwargs: Any) -> np.ndarray:
    """Construct a one-dimensional grid using a registered strategy.

    Built-in strategies are ``"linear"``, ``"log"``, ``"power"``,
    ``"chebyshev"``, and ``"sinh"``. Custom strategies can be added
    using :func:`register_grid`.

    Strategy names are case-insensitive.

    Args:
        kind: Registered grid strategy or supported alias.
        **kwargs: Arguments passed to the selected grid builder.

    Returns:
        Finite, strictly increasing one-dimensional array of grid
        points with dtype ``float64``.

    Raises:
        ValueError: If the strategy is unknown or its output is invalid.
    """
    key = kind.lower()
    key = _GRID_ALIASES.get(key, key)

    try:
        builder = _GRID_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown grid kind {kind!r}.") from exc

    result = np.asarray(builder(**kwargs), dtype=np.float64)

    if (
        result.ndim != 1
        or result.size < 2
        or not np.all(np.isfinite(result))
        or not np.all(np.diff(result) > 0)
    ):
        raise ValueError(
            f"Grid builder {kind!r} must return a finite, "
            "strictly increasing 1D array with at least two points."
        )

    return result
