"""Registry for named redshift-distribution models.

This module maps string model names (e.g. "smail", "gaussian") to callables that
evaluate redshift distributions n(z) on a provided redshift grid.

The registry provides:
- Discovery via `available_models`
- Lookup via `get_model`
- Convenience evaluation via `nz_model`
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from binny.nz.models import (
    gamma_distribution,
    gaussian_distribution,
    gaussian_mixture_distribution,
    lognormal_distribution,
    schechter_like_distribution,
    shifted_smail_distribution,
    skew_normal_distribution,
    smail_like_distribution,
    student_t_distribution,
    tophat_distribution,
)
from binny.utils.types import FloatArray

__all__ = [
    "available_models",
    "nz_model",
    "get_model",
]


class DistFunc(Protocol):
    """Protocol for redshift-distribution model callables.

    Model functions take a redshift grid `z` and model-specific parameters, and
    return an array of the same shape as `z`.

    Methods:
        __call__: Evaluates the distribution model on a redshift grid.
    """

    def __call__(self, z: FloatArray, /, **params: Any) -> FloatArray: ...


_MODELS: dict[str, DistFunc] = {
    # canonical keys
    "smail": smail_like_distribution,
    "gaussian": gaussian_distribution,
    "gaussian_mixture": gaussian_mixture_distribution,
    "gamma": gamma_distribution,
    "schechter": schechter_like_distribution,
    "lognormal": lognormal_distribution,
    "tophat": tophat_distribution,
    "shifted_smail": shifted_smail_distribution,
    "skew_normal": skew_normal_distribution,
    "student_t": student_t_distribution,
}


def available_models() -> list[str]:
    """Lists the supported redshift-distribution model names.

    Returns:
        A sorted list of registry keys that can be passed to `get_model` or
        `nz_model`.
    """
    return sorted(_MODELS.keys())


def get_model(name: str) -> DistFunc:
    """Gets a registered redshift-distribution model by name.

    Args:
        name: Model name (case-insensitive). Must be one of `available_models()`.

    Returns:
        The callable model associated with `name`.

    Raises:
        ValueError: If `name` is not a known model key.
    """
    key = str(name).lower()
    try:
        return _MODELS[key]
    except KeyError as e:
        raise ValueError(
            f"Unknown redshift distribution model '{name}'. "
            f"Available: {', '.join(available_models())}"
        ) from e


def nz_model(name: str, z: Any, /, **params: Any) -> FloatArray:
    """Evaluates a named redshift-distribution model on a redshift grid.

    This is a convenience wrapper around `get_model` that also ensures `z` and
    the returned array are `np.float64`.

    Args:
        name: Model name (case-insensitive). Must be one of `available_models()`.
        z: Redshift grid. Any array-like input accepted by `np.asarray`.
        **params: Model-specific keyword parameters forwarded to the model.

    Returns:
        The model evaluated on `z` as a float64 NumPy array with the same shape
        as `z`.

    Raises:
        ValueError: If `name` is not a known model key.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    fn = get_model(name)
    return np.asarray(fn(z_arr, **params), dtype=np.float64)
