"""Utilities for sampling mock redshift catalogs from tabulated distributions."""

from __future__ import annotations

from typing import Any

import numpy as np

from binny.utils.normalization import cdf_from_curve
from binny.utils.types import FloatArray, ObservationModel
from binny.utils.validators import (
    validate_axis_and_weights,
    validate_positive_integer,
)

__all__ = [
    "sample_redshifts_from_nz",
    "sample_mock_catalog_from_nz",
]


def _get_rng(
    seed: int | None,
    rng: np.random.Generator | None,
) -> np.random.Generator:
    """Returns a random generator from a seed or an existing generator."""
    if seed is not None and rng is not None:
        raise ValueError("Specify either seed or rng, not both.")

    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        return rng

    return np.random.default_rng(seed)


def sample_redshifts_from_nz(
    z: Any,
    nz: Any,
    n_samples: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Sample continuous redshifts from a tabulated redshift distribution.

    The distribution is interpreted as piecewise linear between grid points.
    Sampling uses inverse-CDF sampling with exact inversion within each
    interval, including on nonuniform grids. The input is normalized
    internally and may represent any redshift distribution.

    Args:
        z: Strictly increasing one-dimensional redshift grid.
        nz: Nonnegative distribution evaluated on ``z``.
        n_samples: Number of redshifts to generate.
        seed: Optional random seed for reproducibility.
        rng: Optional NumPy random generator. Cannot be combined with ``seed``.

    Returns:
        One-dimensional array of sampled redshifts within the input grid.

    Raises:
        TypeError: If ``n_samples`` or ``rng`` has an invalid type.
        ValueError: If the distribution is invalid or has zero integral.
    """
    n_samples = validate_positive_integer(n_samples, name="n_samples")
    generator = _get_rng(seed, rng)

    z_arr, nz_arr = validate_axis_and_weights(z, nz)

    if np.any(z_arr < 0):
        raise ValueError("z must be nonnegative.")

    if np.any(nz_arr < 0):
        raise ValueError("nz must be nonnegative.")

    scale = float(np.max(nz_arr))
    if scale <= 0:
        raise ValueError("nz must have a positive integral.")

    density = nz_arr / scale

    cdf, norm = cdf_from_curve(z_arr, density)
    if not np.isfinite(norm):
        raise ValueError("nz must have a positive finite integral.")

    cdf /= norm

    u = generator.random(n_samples)
    indices = np.searchsorted(cdf[1:], u, side="right")
    indices = np.minimum(indices, z_arr.size - 2)

    left = density[indices]
    right = density[indices + 1]
    width = z_arr[indices + 1] - z_arr[indices]

    interval_probability = cdf[indices + 1] - cdf[indices]
    local_fraction = (u - cdf[indices]) / interval_probability
    local_fraction = np.clip(local_fraction, 0.0, 1.0)

    slope = right - left
    target = 0.5 * local_fraction * (left + right)

    discriminant = np.maximum(left**2 + 2.0 * slope * target, 0.0)
    denominator = left + np.sqrt(discriminant)

    fraction = np.divide(
        2.0 * target,
        denominator,
        out=np.zeros_like(target),
        where=denominator > 0,
    )

    fraction = np.where(left == right, local_fraction, fraction)
    fraction = np.clip(fraction, 0.0, 1.0)

    return z_arr[indices] + fraction * width


def sample_mock_catalog_from_nz(
    z: Any,
    nz: Any,
    n_samples: int,
    *,
    observation_model: ObservationModel | None = None,
    weights: Any | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, FloatArray]:
    """Generate a mock redshift catalog from a tabulated distribution.

    Redshifts are sampled from the supplied distribution without assuming
    a particular measurement technique. An optional callable can generate
    observed redshifts from the sampled values.

    The observation model must accept ``(z_samples, rng)`` and return one
    finite observed redshift per input sample. This permits photo-z, spec-z,
    grism, narrow-band, or user-defined measurement models.

    Args:
        z: Strictly increasing one-dimensional redshift grid.
        nz: Nonnegative distribution evaluated on ``z``.
        n_samples: Number of catalog objects to generate.
        observation_model: Optional callable accepting sampled redshifts
            and a NumPy random generator, returning observed redshifts.
            The input distribution must represent the underlying redshift
            population appropriate for this model.
        weights: Optional scalar or one-dimensional array of object weights.
            Defaults to unit weights.
        seed: Optional random seed for reproducibility.
        rng: Optional NumPy random generator. Cannot be combined with ``seed``.

    Returns:
        Dictionary containing ``z`` and ``weight``. When an observation
        model is provided, also contains ``z_obs``. The ``z`` values
        represent the distribution supplied by the caller; they are
        interpreted as true redshifts only when the input distribution
        is a true-redshift distribution.

    Raises:
        TypeError: If the sample count, random generator, or observation
            model has an invalid type.
        ValueError: If the distribution, weights, or observed redshifts
            are invalid.
    """
    n_samples = validate_positive_integer(n_samples, name="n_samples")
    generator = _get_rng(seed, rng)

    if observation_model is not None and not callable(observation_model):
        raise TypeError("observation_model must be callable.")

    if weights is None:
        weight = np.ones(n_samples, dtype=np.float64)
    else:
        try:
            weight = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must be numeric.") from exc

        if weight.ndim == 0:
            weight = np.full(n_samples, float(weight), dtype=np.float64)
        elif weight.ndim != 1 or weight.size != n_samples:
            raise ValueError("weights must be a scalar or an array of length n_samples.")

        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("weights must be finite and nonnegative.")

    z_samples = sample_redshifts_from_nz(z, nz, n_samples, rng=generator)

    catalog: dict[str, FloatArray] = {
        "z": z_samples,
        "weight": weight,
    }

    if observation_model is not None:
        try:
            z_obs = np.asarray(
                observation_model(z_samples.copy(), generator),
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("observation_model must return numeric redshifts.") from exc

        if z_obs.shape != z_samples.shape:
            raise ValueError("observation_model must return one redshift per sampled object.")

        if not np.all(np.isfinite(z_obs)):
            raise ValueError("observation_model must return finite redshifts.")

        catalog["z_obs"] = z_obs

    return catalog
