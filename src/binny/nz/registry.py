"""Registry for named redshift-distribution models."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

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

FloatArray = NDArray[np.float64]

__all__ = [
    "available_models",
    "nz_model",
    "get_model",
]


class DistFunc(Protocol):
    """Callable interface for redshift distribution models."""

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
    """Returns the names of supported redshift distribution models."""
    return sorted(_MODELS.keys())


def get_model(name: str) -> DistFunc:
    """Returns the callable model associated with ``name``."""
    key = str(name).lower()
    try:
        return _MODELS[key]
    except KeyError as e:
        raise ValueError(
            f"Unknown redshift distribution model '{name}'. "
            f"Available: {', '.join(available_models())}"
        ) from e


def nz_model(name: str, z: Any, /, **params: Any) -> FloatArray:
    """Evaluates a named redshift distribution model on ``z``."""
    z_arr = np.asarray(z, dtype=np.float64)
    fn = get_model(name)
    return np.asarray(fn(z_arr, **params), dtype=np.float64)
