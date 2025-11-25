# src/binny/bin_stats.py

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from src.binny.utils.validation import validate_axis_and_weights

__all__ = [
    "bin_moments",
    "summarize_bins",
    "bin_integrals",
    "bin_fractions",
    "n_eff_per_bin",
]


def bin_moments(
    z: ArrayLike,
    nz_bin: ArrayLike,
) -> tuple[float, float]:
    """Return mean and std of a single binned distribution."""
    z_arr, nz_arr = validate_axis_and_weights(z, nz_bin)

    norm = np.trapezoid(nz_arr, z_arr)
    if norm <= 0:
        raise ValueError("Bin normalization must be positive.")

    mean = np.trapezoid(z_arr * nz_arr, z_arr) / norm
    var = np.trapezoid((z_arr - mean) ** 2 * nz_arr, z_arr) / norm
    return float(mean), float(np.sqrt(var))


def summarize_bins(
    z: ArrayLike,
    bins: dict[int, np.ndarray],
    n_eff_per_bin: Mapping[int, float] | None = None,
    sigma_mean: float | Sequence[float] | Mapping[int, float] | None = None,
) -> dict[int, dict[str, float]]:
    """Compute mean, std (and optional error on mean) for each bin.

    Args:
        z:
            1D redshift grid.
        bins:
            Dict of bin index -> n_i(z) on the same z-grid.
        n_eff_per_bin:
            Optional dict of bin index -> effective number density or galaxy count.
            If provided and `sigma_mean` is None, the error on the mean is computed
            as std/sqrt(N).
        sigma_mean:
            Optional external error on the mean. Can be:
              * a single float -> same error for all bins
              * a 1D sequence/array of length n_bins -> per-bin errors
              * a dict[int, float] -> per-bin errors keyed by bin index.

    Returns:
        Dict[bin_idx] -> {"mean": ..., "std": ..., "sigma_mean": ... (optional)}.
    """
    stats: dict[int, dict[str, float]] = {}
    bin_indices = sorted(bins.keys())

    # Normalise sigma_mean to a dict[int, float] if provided
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
            sigma_mean_dict = {idx: float(val) for idx, val in zip(bin_indices, arr)}

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
    z: ArrayLike,
    bins: dict[int, np.ndarray],
) -> dict[int, float]:
    """Compute ∫ n_i(z) dz for each bin.

    This is the generic version of your old “integrate n(z) per bin” logic.

    Args:
        z:
            1D redshift grid.
        bins:
            Dict of bin index -> n_i(z) on the same z-grid.

    Returns:
        Dict[bin_idx] -> integral_i, where integral_i = ∫ n_i(z) dz.
    """
    z_arr = np.asarray(z, dtype=float)
    integrals: dict[int, float] = {}

    for idx, nz_bin in bins.items():
        _, nz_arr = validate_axis_and_weights(z_arr, nz_bin)
        integrals[idx] = float(np.trapezoid(nz_arr, z_arr))

    return integrals


def bin_fractions(
    z: ArrayLike,
    bins: dict[int, np.ndarray],
) -> dict[int, float]:
    """Compute fraction of galaxies per bin from n_i(z).

    Fractions are defined as:
        f_i = ∫ n_i(z) dz / Σ_j ∫ n_j(z) dz.

    This is the generic version of your old get_n_eff_frac_* helpers.

    Args:
        z:
            1D redshift grid.
        bins:
            Dict of bin index -> n_i(z) on the same z-grid.

    Returns:
        Dict[bin_idx] -> f_i, such that Σ_i f_i = 1 (up to numerical noise).
    """
    integrals = bin_integrals(z, bins)
    total = sum(integrals.values())

    if total <= 0:
        raise ValueError("Total integrated n(z) over all bins must be positive.")

    return {idx: val / total for idx, val in integrals.items()}


def n_eff_per_bin(
    z: ArrayLike,
    bins: Mapping[int, ArrayLike],
    n_eff_total: float,
    *,
    expect_unnormalized: bool = True,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute n_eff per bin and fractional n_eff from *unnormalised* bins.

    Args:
        z:
            1D redshift grid.
        bins:
            Mapping bin_idx -> n_i(z). Should be unnormalised if
            expect_unnormalized=True.
        n_eff_total:
            Total effective number density (or total number of galaxies)
            for the sample.
        expect_unnormalized:
            If True, raise if all bins look normalised (∫ n_i dz ≈ 1).
        rtol, atol:
            Tolerances for deciding whether a bin is 'normalised'.

    Returns:
        (n_eff_per_bin, frac_per_bin) where both are dict[bin_idx, float].
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
        # If *all* bins integrate to ~1, they are almost certainly normalised.
        if all(np.isclose(val, 1.0, rtol=rtol, atol=atol) for val in integrals.values()):
            raise ValueError(
                "n_eff_from_bins: all bins appear normalised (∫ n_i(z) dz ≈ 1). "
                "You probably passed bins built with normalize_bins=True. "
                "Use normalize_bins=False when building bins for n_eff."
            )

    total_counts = sum(integrals.values())
    if total_counts <= 0:
        raise ValueError("Total integral over all bins must be positive to compute n_eff.")

    frac_per_bin = {idx: val / total_counts for idx, val in integrals.items()}
    n_eff_per_bin = {idx: n_eff_total * frac for idx, frac in frac_per_bin.items()}

    return n_eff_per_bin, frac_per_bin