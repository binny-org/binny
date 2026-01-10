from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from binny.ztomo.distributions import (
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

__all__ = [
    "redshift_distribution",
    "available_redshift_distributions",
]

FloatArray = NDArray[np.float64]


class DistFunc(Protocol):
    """Protocol for redshift distribution functions."""

    def __call__(self, z: NDArray[np.floating], /, **params: Any) -> FloatArray: ...


_DISTS: dict[str, DistFunc] = {
    "smail": smail_like_distribution,
    "smail_like": smail_like_distribution,
    "gaussian": gaussian_distribution,
    "gaussian_mixture": gaussian_mixture_distribution,
    "gamma": gamma_distribution,
    "schechter": schechter_like_distribution,
    "schechter_like": schechter_like_distribution,
    "lognormal": lognormal_distribution,
    "tophat": tophat_distribution,
    "shifted_smail": shifted_smail_distribution,
    "skew_normal": skew_normal_distribution,
    "student_t": student_t_distribution,
}


def available_redshift_distributions() -> list[str]:
    """Lists the names of available redshift distributions."""
    return sorted(_DISTS.keys())


def redshift_distribution(
    name: str, z: NDArray[np.float64], /, **params: Any
) -> FloatArray:
    """Evaluates a named redshift distribution.

    Args:
        name: Name of the redshift distribution. Available options are:
            ``'smail'``, ``'gaussian'``, ``'gaussian_mixture'``, ``'gamma'``,
            ``'schechter'``, ``'lognormal'``, ``'tophat'``, ``'shifted_smail'``,
            ``'skew_normal'``, ``'student_t'``.
        z: Redshift values where to evaluate the distribution.
        **params: Parameters specific to the chosen distribution.

    Returns:
        Array of evaluated distribution values at the input redshifts.

    Raises:
        ValueError: If the specified distribution name is not recognized.
    """
    key = name.lower()
    try:
        fn = _DISTS[key]
    except KeyError as e:
        raise ValueError(
            f"Unknown redshift distribution '{name}'. "
            f"Available: {', '.join(available_redshift_distributions())}"
        ) from e
    z_arr = np.asarray(z, dtype=np.float64)
    return fn(z_arr, **params)
