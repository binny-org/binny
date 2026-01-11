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
    """Callable interface for redshift distribution functions.

    A distribution function maps an array of redshift values to an array of
    non-negative weights of the same shape. Extra keyword parameters control
    the shape of the distribution (e.g., location/scale/shape parameters).

    Implementations are expected to accept ``z`` as the first positional-only
    argument and accept distribution-specific parameters via ``**params``.
    """

    def __call__(self, z: NDArray[np.floating], /, **params: Any) -> FloatArray:
        """Evaluates the distribution function."""
        ...


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
    """Returns the names of supported redshift distribution models.

    This is the authoritative list of distribution identifiers accepted by
    :func:`redshift_distribution`. Names are returned in sorted order and
    include aliases (e.g., ``"smail"`` and ``"smail_like"``).

    Returns:
        Sorted list of distribution names.
    """
    return sorted(_DISTS.keys())


def redshift_distribution(
    name: str, z: NDArray[np.float64], /, **params: Any
) -> FloatArray:
    """Evaluates a named redshift distribution model.

    This is a convenience dispatcher over the concrete distribution functions
    in :mod:`binny.ztomo.distributions`. It selects a distribution by name
    (case-insensitive), converts ``z`` to ``float64``, and evaluates the model.

    The meaning and valid set of ``**params`` depends on the chosen model. For
    a given name, pass exactly the keyword parameters that the underlying
    distribution function expects.

    Args:
        name: Distribution identifier (case-insensitive). Use
            :func:`available_redshift_distributions` to list valid names.
        z: Redshift grid at which to evaluate the model. Any array-like input is
            accepted; it is converted to a ``float64`` NumPy array and the
            output preserves the same shape.
        **params: Model-specific parameters forwarded to the underlying
            distribution implementation.

    Returns:
        Array of distribution values evaluated at ``z`` (``float64``) with the
        same shape as the input ``z``.

    Raises:
        ValueError: If ``name`` is not a supported distribution identifier.

    Examples:
    >>> import numpy as np
    >>> from binny.api.distributions import redshift_distribution
    >>> z = np.linspace(0.0, 2.0, 5)
    >>> y = redshift_distribution("gaussian", z, mu=1.0, sigma=0.2)
    >>> y.shape == z.shape
    True
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
