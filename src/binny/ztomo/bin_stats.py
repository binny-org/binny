"""Basic statistics of binned redshift distributions."""

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
    "bin_overlaps",
    "bin_overlap_percent",
    "overlapping_bin_pairs",
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
    neff_per_bin: Mapping[int, float] | None = None,
    sigma_mean: float | Sequence[float] | Mapping[int, float] | None = None,
) -> dict[int, dict[str, float]]:
    """Summarizes per-bin redshift moments and uncertainty on the mean.

    For each entry in ``bins``, this function computes the mean and standard
    deviation of ``n_i(z)`` on the common grid ``z``. An uncertainty on the mean
    (``sigma_mean``) is included either from user-provided values or from
    ``neff_per_bin`` using ``std / sqrt(N_eff)``.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to ``n_i(z)`` values evaluated on ``z``.
        neff_per_bin: Optional mapping from bin index to an effective number of
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
        ValueError: If ``neff_per_bin[idx]`` is non-positive for any bin.
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
        elif neff_per_bin is not None:
            num = neff_per_bin[idx]  # effective number of galaxies in this bin
            if num <= 0:
                raise ValueError(f"neff_per_bin[{idx}] must be positive.")
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
        if all(
                np.isclose(val, 1.0, rtol=rtol, atol=atol)
                for val in integrals.values()
        ):
            raise ValueError(
                "All bins appear normalized (integral n_i(z) dz approx 1). "
                "You likely built bins with normalize_bins=True. "
                "Build bins with normalize_bins=False when computing n_eff."
            )

    total_counts = sum(integrals.values())
    if total_counts <= 0:
        raise ValueError("Total integral must be positive to compute n_eff.")

    frac_per_bin = {idx: val / total_counts for idx, val in integrals.items()}
    neff_per_bin = {idx: n_eff_total * frac for idx, frac in frac_per_bin.items()}

    return neff_per_bin, frac_per_bin


def bin_overlaps(
    z: Any,
    bins: Mapping[int, Any],
    *,
    method: str = "min",
    assume_normalized: bool = True,
    normalize_if_needed: bool = True,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> dict[int, dict[int, float]]:
    """Compute a pairwise overlap matrix for binned redshift distributions.

    This function computes a symmetric overlap measure between bins based on
    their one-dimensional distributions evaluated on a shared redshift grid.

    Supported methods:

    - ``method="min"``:
      ``O_ij = integral min(p_i(z), p_j(z)) dz``.
      If each ``p_i`` is normalized (integral equals 1), then ``O_ij`` lies in
      ``[0, 1]`` with ``O_ii = 1``.

    - ``method="cosine"``:
      Cosine similarity under the continuous inner product:

      ``O_ij = (∫ p_i(z) p_j(z) dz) / sqrt((∫ p_i(z)^2 dz) (∫ p_j(z)^2 dz))``.
      For nonnegative distributions, ``O_ij`` lies in ``[0, 1]`` and ``O_ii = 1``
      when ``p_i`` is not identically zero.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to ``n_i(z)`` values evaluated on ``z``.
        method: Overlap metric to use. Supported values are ``"min"`` and
            ``"cosine"``.
        assume_normalized: Only used for method="min".
            If True, bins must have unit integral
            (or will be normalized if normalize_if_needed=True).
        normalize_if_needed: If True, normalize bins that do not appear
            normalized (based on ``rtol`` and ``atol``). If False and
            ``assume_normalized`` is True, a non-normalized bin raises.
        rtol: Relative tolerance for the normalization check.
        atol: Absolute tolerance for the normalization check.

    Returns:
        A nested mapping ``overlaps[i][j] = O_ij`` for all bin indices ``i, j``
        in sorted order. The result is symmetric.

    Raises:
        ValueError: If ``method`` is not supported.
        ValueError: If a bin has non-positive integral (cannot normalize).
        ValueError: If ``assume_normalized`` is True and a bin does not appear
            normalized and ``normalize_if_needed`` is False.
    """
    z_arr = np.asarray(z, dtype=float)
    if len(bins) == 0:
        return {}

    if method not in {"min", "cosine"}:
        raise ValueError('method must be "min" or "cosine".')

    bin_indices = sorted(bins.keys())

    # Validate against the first bin and cast all bins to float arrays.
    first_idx = bin_indices[0]
    _, _ = validate_axis_and_weights(z_arr, bins[first_idx])

    p: dict[int, np.ndarray] = {}
    for idx in bin_indices:
        _, nz_arr = validate_axis_and_weights(z_arr, bins[idx])
        p[idx] = nz_arr.astype(float, copy=False)

    # Optional normalization handling (mainly relevant for method="min").
    if method == "min" and assume_normalized:
        for idx in bin_indices:
            area = float(np.trapezoid(p[idx], z_arr))
            if area <= 0.0:
                raise ValueError(f"bin {idx} has non-positive integral: {area}.")
            if not np.isclose(area, 1.0, rtol=rtol, atol=atol):
                if normalize_if_needed:
                    p[idx] = p[idx] / area
                else:
                    raise ValueError(
                        f"bin {idx} does not appear normalized (integral={area}). "
                        "Set normalize_if_needed=True or assume_normalized=False."
                    )

    overlaps: dict[int, dict[int, float]] = {i: {} for i in bin_indices}

    if method == "min":
        for i in bin_indices:
            for j in bin_indices:
                if j < i:
                    continue
                val = float(np.trapezoid(np.minimum(p[i], p[j]), z_arr))
                overlaps[i][j] = val
                overlaps[j][i] = val
    else:  # method == "cosine"
        norms: dict[int, float] = {}
        for i in bin_indices:
            n2 = float(np.trapezoid(p[i] * p[i], z_arr))
            norms[i] = float(np.sqrt(max(n2, 0.0)))

        for i in bin_indices:
            for j in bin_indices:
                if j < i:
                    continue
                denom = norms[i] * norms[j]
                if denom == 0.0:
                    val = 0.0
                else:
                    num = float(np.trapezoid(p[i] * p[j], z_arr))
                    val = num / denom
                overlaps[i][j] = float(val)
                overlaps[j][i] = float(val)

    return overlaps


def bin_overlap_percent(
    z: Any,
    bins: Mapping[int, Any],
    *,
    assume_normalized: bool = True,
    normalize_if_needed: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> dict[int, dict[int, float]]:
    """Compute pairwise bin overlap as a percentage.

    This returns ``100 * integral min(p_i(z), p_j(z)) dz`` for all bin pairs. When
    each ``p_i`` is normalized to unit integral, the result lies in ``[0, 100]``
    with 100 on the diagonal.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to ``n_i(z)`` values evaluated on ``z``.
        assume_normalized: Whether to assume each bin distribution is normalized.
        normalize_if_needed: If True, normalize bins that do not appear normalized.
        rtol: Relative tolerance for the normalization check.
        atol: Absolute tolerance for the normalization check.

    Returns:
        A nested mapping ``percent[i][j]`` giving the overlap percentage between
        bins ``i`` and ``j``.

    Raises:
        ValueError: If a bin has non-positive integral (cannot normalize).
        ValueError: If ``assume_normalized`` is True and a bin does not appear
            normalized and ``normalize_if_needed`` is False.
    """
    overlaps = bin_overlaps(
        z,
        bins,
        method="min",
        assume_normalized=assume_normalized,
        normalize_if_needed=normalize_if_needed,
        rtol=rtol,
        atol=atol,
    )
    return {
        i: {j: 100.0 * val for j, val in row.items()} for i, row in overlaps.items()
    }


def overlapping_bin_pairs(
    z: Any,
    bins: Mapping[int, Any],
    *,
    threshold_percent: float = 10.0,
    assume_normalized: bool = True,
    normalize_if_needed: bool = False,
) -> list[tuple[int, int, float]]:
    """Returns the list of bin pairs with overlap above a threshold.

    This function computes the pairwise bin overlap percentages and returns a
    list of bin index pairs whose overlap exceeds ``threshold_percent``. Each entry
    in the output list is a tuple ``(i, j, overlap_percent)`` with ``i < j``.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to ``n_i(z)`` values evaluated on ``z``.
        threshold_percent: Minimum overlap percentage for a bin pair to be included
            in the output.
        assume_normalized: Whether to assume each bin distribution is normalized.
        normalize_if_needed: If True, normalize bins that do not appear normalized.

    Returns:
        A list of tuples ``(i, j, overlap_percent)`` for all bin pairs ``i < j``
        with overlap percentage at or above ``threshold_percent``. The list is sorted
        in descending order of overlap percentage.

    Raises:
        ValueError: If a bin has non-positive integral (cannot normalize).
        ValueError: If ``assume_normalized`` is True and a bin does not appear
            normalized and ``normalize_if_needed`` is False.
    """
    pct = bin_overlap_percent(
        z,
        bins,
        assume_normalized=assume_normalized,
        normalize_if_needed=normalize_if_needed,
    )

    out: list[tuple[int, int, float]] = []
    indices = sorted(pct.keys())

    for a, i in enumerate(indices):
        for j in indices[a + 1:]:
            val = float(pct[i][j])
            if val >= threshold_percent:
                out.append((i, j, val))

    out.sort(key=lambda t: t[2], reverse=True)
    return out
