"""Metrics for binned redshift distributions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from binny.core.validators import validate_axis_and_weights

__all__ = [
    "bin_moments",
    "summarize_bins",
    "bin_integrals",
    "bin_fractions",
    "n_eff_per_bin",
]


def bin_moments(
    z: Any,
    nz_bin: Any,
) -> tuple[float, float]:
    """Computes the mean and standard deviation of a redshift bin.

    The inputs are interpreted as a redshift grid ``z`` and a corresponding
    binned redshift distribution ``n_i(z)`` evaluated on that grid. Moments are
    computed using trapezoidal integration.

    Args:
        z: One-dimensional redshift grid.
        nz_bin: One-dimensional array of ``n_i(z)`` values evaluated on ``z``.

    Returns:
        A tuple ``(mean, std)`` giving the mean redshift and standard deviation
        of the bin.

    Raises:
        ValueError: If the integral of ``n_i(z)`` over ``z`` is non-positive.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz_bin)

    norm = np.trapezoid(nz_arr, z_arr)
    if norm <= 0:
        raise ValueError("Bin normalization must be positive.")

    mean = np.trapezoid(z_arr * nz_arr, z_arr) / norm
    var = np.trapezoid((z_arr - mean) ** 2 * nz_arr, z_arr) / norm
    return float(mean), float(np.sqrt(var))


def summarize_bins(
    z: Any,
    bins: dict[int, np.ndarray],
    n_eff_per_bin: Mapping[int, float] | None = None,
    sigma_mean: float | Sequence[float] | Mapping[int, float] | None = None,
) -> dict[int, dict[str, float]]:
    """Summarizes per-bin redshift moments and uncertainty on the mean.

    For each entry in ``bins``, this function computes the mean and standard
    deviation of ``n_i(z)`` on the common grid ``z``. An uncertainty on the mean
    (``sigma_mean``) is included either from user-provided values or from
    ``n_eff_per_bin`` using ``std / sqrt(N_eff)``.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to ``n_i(z)`` values evaluated on ``z``.
        n_eff_per_bin: Optional mapping from bin index to an effective number of
            objects (or effective number density) in that bin. Used to compute
            ``sigma_mean`` when ``sigma_mean`` is not provided.
        sigma_mean: Optional uncertainty on the mean per bin. Supported forms are
            a scalar applied to all bins, a sequence with length equal to the
            number of bins (in sorted bin-index order), or a mapping keyed by bin
            index.

    Returns:
        A mapping ``{bin_idx: {"mean": ..., "std": ..., "sigma_mean": ...}}``.
        The ``"sigma_mean"`` field is included when it can be determined from
        ``sigma_mean`` or ``n_eff_per_bin``.

    Raises:
        ValueError: If ``sigma_mean`` is provided as a sequence with length not
            equal to the number of bins.
        ValueError: If ``n_eff_per_bin[idx]`` is non-positive for any bin.
    """
    stats: dict[int, dict[str, float]] = {}
    bin_indices = sorted(bins.keys())

    sigma_mean_dict: dict[int, float] | None = None
    if sigma_mean is not None:
        if isinstance(sigma_mean, (float, int)):
            sigma_mean_dict = {idx: float(sigma_mean) for idx in bin_indices}
        elif isinstance(sigma_mean, Mapping):
            sigma_mean_dict = {idx: float(sigma_mean[idx]) for idx in bin_indices}
        else:
            arr = np.asarray(sigma_mean, dtype=float)
            if arr.shape[0] != len(bin_indices):
                raise ValueError(
                    "sigma_mean sequence must have length equal to number of bins."
                )
            sigma_mean_dict = {
                idx: float(val) for idx, val in zip(bin_indices, arr, strict=False)
            }

    for idx in bin_indices:
        nz_bin = bins[idx]
        mean, std = bin_moments(z, nz_bin)
        entry: dict[str, float] = {"mean": mean, "std": std}

        if sigma_mean_dict is not None:
            entry["sigma_mean"] = sigma_mean_dict[idx]
        elif n_eff_per_bin is not None:
            num = n_eff_per_bin[idx]  # effective number of galaxies in this bin
            if num <= 0:
                raise ValueError(f"n_eff_per_bin[{idx}] must be positive.")
            entry["sigma_mean"] = std / np.sqrt(num)

        stats[idx] = entry

    return stats


def bin_integrals(
    z: Any,
    bins: dict[int, np.ndarray],
) -> dict[int, float]:
    """Computes the integral of ``n_i(z)`` over ``z`` for each bin.

    Integrals are computed by trapezoidal integration on the provided redshift
    grid ``z``.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to ``n_i(z)`` values evaluated on ``z``.

    Returns:
        A mapping ``{bin_idx: integral}`` where ``integral`` is the integral of
        ``n_i(z)`` over ``z``.
    """
    z_arr = np.asarray(z, dtype=float)
    integrals: dict[int, float] = {}

    for idx, nz_bin in bins.items():
        _, nz_arr = validate_axis_and_weights(z_arr, nz_bin)
        integrals[idx] = float(np.trapezoid(nz_arr, z_arr))

    return integrals


def bin_fractions(
    z: Any,
    bins: dict[int, np.ndarray],
) -> dict[int, float]:
    """Computes fractional bin weights from integrated ``n_i(z)``.

    Fractions are defined as:

    ``f_i = integral(n_i(z) dz) / sum_j integral(n_j(z) dz)``.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to ``n_i(z)`` values evaluated on ``z``.

    Returns:
        A mapping ``{bin_idx: f_i}`` such that ``sum_i f_i`` is approximately 1 up
        to numerical integration error.

    Raises:
        ValueError: If the total integrated ``n(z)`` across all bins is
            non-positive.
    """
    integrals = bin_integrals(z, bins)
    total = sum(integrals.values())

    if total <= 0:
        raise ValueError("Total integrated n(z) over all bins must be positive.")

    return {idx: val / total for idx, val in integrals.items()}


def n_eff_per_bin(
    z: Any,
    bins: Mapping[int, Any],
    n_eff_total: float,
    *,
    expect_unnormalized: bool = True,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> tuple[dict[int, float], dict[int, float]]:
    """Computes per-bin effective counts and fractions from binned distributions.

    This function estimates per-bin weights from the integral of ``n_i(z)`` over
    ``z`` and allocates a total effective count (or effective number density)
    ``n_eff_total`` across bins in proportion to those weights.

    When ``expect_unnormalized`` is True, the function checks whether all bins
    appear normalized (integral approximately 1) and raises if so, since normalized
    bins do not encode relative bin populations.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to ``n_i(z)`` values evaluated on ``z``.
        n_eff_total: Total effective count (or effective number density) for the
            full sample.
        expect_unnormalized: Whether to treat normalized bins as an error
            condition.
        rtol: Relative tolerance used when checking ``integral of n_i(z) dz approx 1``.
        atol: Absolute tolerance used when checking ``integral of n_i(z) dz approx 1``.

    Returns:
        A tuple ``(n_eff_per_bin, frac_per_bin)`` where:
        - ``n_eff_per_bin`` maps bin index to allocated effective count.
        - ``frac_per_bin`` maps bin index to the fractional weight derived from
          integrals.

    Raises:
        ValueError: If ``expect_unnormalized`` is True and all bins appear
            normalized within ``rtol`` and ``atol``.
        ValueError: If the total integral ``sum_i integral n_i(z) dz`` is non-positive.
    """
    # Make sure z is an array and compatible with one bin
    z_arr = np.asarray(z, dtype=float)
    # Use validate_axis_and_weights on the first bin we see
    first_idx = next(iter(bins.keys()))
    _, _ = validate_axis_and_weights(z_arr, bins[first_idx])

    integrals: dict[int, float] = {}
    for idx, nz_bin in bins.items():
        nz_arr = np.asarray(nz_bin, dtype=float)
        integrals[idx] = float(np.trapezoid(nz_arr, z_arr))

    if expect_unnormalized:
        if all(np.isclose(val, 1.0, rtol=rtol, atol=atol) for val in integrals.values()):
            raise ValueError(
                "n_eff_from_bins: all bins appear normalised (integral n_i(z) dz approx 1). "
                "You probably passed bins built with normalize_bins=True. "
                "Use normalize_bins=False when building bins for n_eff."
            )

    total_counts = sum(integrals.values())
    if total_counts <= 0:
        raise ValueError("Total integral over all bins must be positive to compute n_eff.")

    frac_per_bin = {idx: val / total_counts for idx, val in integrals.items()}
    neff_per_bin = {idx: n_eff_total * frac for idx, frac in frac_per_bin.items()}

    return neff_per_bin, frac_per_bin
