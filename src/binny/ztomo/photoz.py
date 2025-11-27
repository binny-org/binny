"""Functions to build photometric redshift smeared n(z) distributions per tomographic bin."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf

from src.binny.core.validation import validate_axis_and_weights, validate_n_bins
from src.binny.utils.normalization import normalize_1d
from src.binny.utils.broadcasting import as_per_bin

__all__ = [
    "build_photoz_bins",
    "true_redshift_distribution",
]


def build_photoz_bins(
    z: ArrayLike,
    nz: ArrayLike,
    bin_edges: ArrayLike,
    sigma_z_per_bin: Sequence[float],
    z_bias_per_bin: Sequence[float],
    *,
    c_bias_per_bin: Sequence[float] | float = 1.0,
    f_out_per_bin: Sequence[float] | float = 0.0,
    sigma_out_per_bin: Sequence[float] | float | None = None,
    z_bias_out_per_bin: Sequence[float] | float = 0.0,
    c_bias_out_per_bin: Sequence[float] | float = 1.0,
    # normalisation controls
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: str = "trapz",
) -> dict[int, np.ndarray]:
    """Return photo-z–smeared n(z) per tomographic bin."""
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bins = bin_edges.size - 1
    validate_n_bins(n_bins)

    if len(sigma_z_per_bin) != n_bins or len(z_bias_per_bin) != n_bins:
        raise ValueError("sigma_z_per_bin and z_bias_per_bin must have length n_bins.")

    # --- guard against double-normalisation of the parent nz ---
    if normalize_input:
        total = np.trapezoid(n_arr, z_arr)
        if np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
            raise ValueError(
                "build_photoz_bins: normalize_input=True but intrinsic nz already looks "
                f"normalised (∫ n(z) dz ≈ {total:.4f}). "
                "Set normalize_input=False if nz is already normalised."
            )
        n_arr = normalize_1d(z_arr, n_arr, method=norm_method)

    c_b       = as_per_bin(c_bias_per_bin,     n_bins, "c_bias_per_bin")
    f_out     = as_per_bin(f_out_per_bin,      n_bins, "f_out_per_bin")
    sigma_out = as_per_bin(sigma_out_per_bin,  n_bins, "sigma_out_per_bin")
    z_b_out   = as_per_bin(z_bias_out_per_bin, n_bins, "z_bias_out_per_bin")
    c_b_out   = as_per_bin(c_bias_out_per_bin, n_bins, "c_bias_out_per_bin")

    bins: dict[int, np.ndarray] = {}

    for i, (z_min, z_max) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        nz_bin = true_redshift_distribution(
            z_arr,
            n_arr,
            bin_min=z_min,
            bin_max=z_max,
            sigma_z=sigma_z_per_bin[i],
            z_bias=z_bias_per_bin[i],
            c_bias=c_b[i],
            f_out=f_out[i],
            sigma_out=sigma_out[i],
            z_bias_out=z_b_out[i],
            c_bias_out=c_b_out[i],
        )
        if normalize_bins:
            nz_bin = normalize_1d(z_arr, nz_bin, method=norm_method)

        bins[i] = nz_bin

    return bins


def true_redshift_distribution(
    z: ArrayLike,
    nz: ArrayLike,
    bin_min: float,
    bin_max: float,
    sigma_z: float,
    z_bias: float,
    *,
    c_bias: float = 1.0,
    f_out: float = 0.0,
    sigma_out: float | None = None,
    z_bias_out: float = 0.0,
    c_bias_out: float = 1.0,
) -> np.ndarray:
    """Convolves intrinsic n(z) with photo-z errors within a given photo-z bin."""
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    if not (0.0 <= f_out <= 1.0):
        raise ValueError("f_out must lie in [0, 1].")

    scatter_core = np.maximum(sigma_z * (1.0 + z_arr), 1e-10)
    mu_core = c_bias * z_arr - z_bias

    sqrt2 = np.sqrt(2.0)
    upper_core = (bin_max - mu_core) / (sqrt2 * scatter_core)
    lower_core = (bin_min - mu_core) / (sqrt2 * scatter_core)
    p_core = 0.5 * (erf(upper_core) - erf(lower_core))

    if f_out > 0.0 and sigma_out is not None:
        scatter_out = np.maximum(sigma_out * (1.0 + z_arr), 1e-10)
        mu_out = c_bias_out * z_arr + z_bias_out

        upper_out = (bin_max - mu_out) / (sqrt2 * scatter_out)
        lower_out = (bin_min - mu_out) / (sqrt2 * scatter_out)
        p_out = 0.5 * (erf(upper_out) - erf(lower_out))

        p_bin_given_z = (1.0 - f_out) * p_core + f_out * p_out
    else:
        p_bin_given_z = p_core

    return n_arr * p_bin_given_z
