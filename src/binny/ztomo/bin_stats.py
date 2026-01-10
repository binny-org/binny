"""Basic statistics of binned redshift distributions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from binny.utils.normalization import (
    cdf_from_curve,
    integrate_bins,
    weighted_quantile_from_cdf,
)
from binny.utils.validators import validate_axis_and_weights

__all__ = [
    "bin_moments",
    "bin_centers",
    "summarize_bins",
    "galaxy_fraction_per_bin",
    "galaxy_density_per_bin",
    "galaxy_count_per_bin",
    "bin_quantiles",
    "in_range_fraction",
    "in_range_fraction_per_bin",
    "peak_flags",
    "peak_flags_per_bin",
]

_MSG_NORMALIZED_BINS = (
    "Bin curves appear normalized, so their integrals do not encode "
    "relative bin populations. Use unnormalized bin curves or provide "
    "explicit per-bin weights."
)


def bin_moments(z: Any, nz_bin: Any) -> dict[str, float]:
    """Computes summary statistics for a single bin distribution.

    The inputs are a redshift grid z and a corresponding bin distribution nz_bin
    evaluated on that grid.

    The returned dictionary contains:

    - "mean": weighted mean redshift.
    - "median": weighted median redshift.
    - "mode": redshift value on the input grid where nz_bin is maximal.
    - "std": weighted standard deviation.
    - "skewness": standardized third central moment (dimensionless).
    - "kurtosis": standardized fourth central moment minus 3 (dimensionless).
    - "iqr": interquartile range (q75 - q25).
    - "width_68": central 68 percent interval width (q84 - q16).

    Notes:
        - nz_bin does not need to be normalized.
        - Percentile-based summaries depend on the sampling of the grid.
        - The mode is grid-dependent by definition.

    Args:
        z: One-dimensional redshift grid.
        nz_bin: Values of the bin distribution evaluated on z.

    Returns:
        Dictionary of summary statistics.

    Raises:
        ValueError: If the total weight is not positive.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz_bin)

    norm = float(np.trapezoid(nz_arr, x=z_arr))
    if norm <= 0.0:
        raise ValueError("Total weight must be positive.")

    mean = float(np.trapezoid(z_arr * nz_arr, x=z_arr) / norm)

    # Central moments
    m2 = float(np.trapezoid(((z_arr - mean) ** 2) * nz_arr, x=z_arr) / norm)
    m3 = float(np.trapezoid(((z_arr - mean) ** 3) * nz_arr, x=z_arr) / norm)
    m4 = float(np.trapezoid(((z_arr - mean) ** 4) * nz_arr, x=z_arr) / norm)

    std = float(np.sqrt(max(m2, 0.0)))

    if std > 0.0:
        skewness = float(m3 / (std**3))
        kurtosis = float(m4 / (std**4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0

    mode = float(z_arr[int(np.nanargmax(nz_arr))])

    # Quantiles for median + effective widths
    cdf, norm = cdf_from_curve(z_arr, nz_arr)
    q16 = weighted_quantile_from_cdf(z_arr, cdf, norm, 0.16)
    q25 = weighted_quantile_from_cdf(z_arr, cdf, norm, 0.25)
    q50 = weighted_quantile_from_cdf(z_arr, cdf, norm, 0.50)
    q75 = weighted_quantile_from_cdf(z_arr, cdf, norm, 0.75)
    q84 = weighted_quantile_from_cdf(z_arr, cdf, norm, 0.84)

    iqr = float(q75 - q25)
    width_68 = float(q84 - q16)

    return {
        "mean": mean,
        "median": float(q50),
        "mode": mode,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "iqr": iqr,
        "width_68": width_68,
    }


def bin_centers(
    z: Any,
    bins: Mapping[int, Any],
    *,
    method: str = "mean",
    decimal_places: int | None = 2,
) -> dict[int, float]:
    """Computes one center value per tomographic bin.

    For each bin, this function returns a single representative redshift.

    Supported methods:
        - "mean": weighted mean redshift.
        - "median": weighted median redshift.
        - "mode": redshift value on the input grid where the bin curve is
            maximal.
        - "pXX": weighted percentile center, where XX is a number between
            0 and 100 (for example, "p50" for the median, "p16" and "p84"
            for common bounds).

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin distributions evaluated on z.
        method: Center definition.
        decimal_places: If not None, round each returned center to this many
            decimal places. If None, return full-precision floats.

    Returns:
        Mapping from bin index to the chosen center.

    Raises:
        ValueError: If method is not supported.
        ValueError: If a bin has non-positive total weight when needed.
        ValueError: If method is not supported.
    """
    method_l = method.lower()

    # Parse percentile method like "p50"
    percentile: float | None = None
    if method_l.startswith("p") and len(method_l) > 1:
        try:
            percentile = float(method_l[1:])
        except ValueError as e:
            raise ValueError('percentile methods must look like "p50" or "p16".') from e
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("percentile in method='pXX' must be between 0 and 100.")

    centers: dict[int, float] = {}
    for bin_idx, distribution in bins.items():
        if percentile is not None:
            qs = bin_quantiles(z, distribution, [percentile / 100.0])
            val = float(qs[percentile / 100.0])
        else:
            stats = bin_moments(z, distribution)
            if method_l in {"mean", "median", "mode"}:
                val = float(stats["median" if method_l == "median" else method_l])
            else:
                raise ValueError(
                    'method must be "mean", "median", "mode", or a percentile'
                    ' like "p50".'
                )

        if decimal_places is not None:
            centers[int(bin_idx)] = round(val, int(decimal_places))
        else:
            centers[int(bin_idx)] = val

    return centers


def summarize_bins(
    z: Any,
    bins: Mapping[int, Any],
    *,
    count_per_bin: Mapping[int, float] | None = None,
    density_per_bin: Mapping[int, float] | None = None,
    survey_area: float | None = None,
    sigma_mean: float | Sequence[float] | Mapping[int, float] | None = None,
) -> dict[int, dict[str, float]]:
    """Summarizes per-bin redshift moments and the uncertainty on the mean.

    For each bin distribution, this function computes the weighted mean
    redshift and the weighted standard deviation on the shared grid ``z``.
    If an uncertainty on the mean is available, it is included as
    ``sigma_mean``. This happens in one of three ways:

    - If ``sigma_mean`` is provided, it is used directly.
    - If count_per_bin is provided, ``sigma_mean`` is computed as std divided by the
      square root of the count for that bin.
    - If density_per_bin and survey_area are provided, an effective count is
      inferred as density_per_bin times survey_area, and ``sigma_mean`` is computed
      as std divided by the square root of that inferred count.

    Conventions:
        - ``density_per_bin`` is interpreted as a surface density in galaxies
            per square arcminute.
        - ``survey_area`` is interpreted as an area in square arcminutes.
        - ``count_per_bin`` is an effective (dimensionless) count.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to n_i(z) values evaluated on ``z``.
        count_per_bin: Optional mapping from bin index to effective counts.
        density_per_bin: Optional mapping from bin index to surface density in
            galaxies per square arcminute.
        survey_area: Survey area in square arcminutes. Required if
            ``density_per_bin`` is used to infer counts.
        sigma_mean: Optional uncertainty on the mean per bin. May be a scalar
            applied to all bins, a sequence aligned with sorted bin indices,
            or a mapping keyed by bin index.

    Returns:
        Mapping of the form ``{bin_idx: {"mean": ..., "std": ..., "sigma_mean": ...}}``.
        The ``"sigma_mean"`` field is included only when it can be determined.

    Raises:
        ValueError: If survey_area is required but not provided or not positive.
        ValueError: If any required bin index is missing from ``sigma_mean``,
            count_per_bin, or density_per_bin.
        ValueError: If any provided count is not positive.
    """
    stats: dict[int, dict[str, float]] = {}
    bin_indices = sorted(bins.keys())

    sigma_mean_dict: dict[int, float] | None = None
    if sigma_mean is not None:
        if isinstance(sigma_mean, float | int):
            sigma_mean_dict = {idx: float(sigma_mean) for idx in bin_indices}
        elif isinstance(sigma_mean, Mapping):
            try:
                sigma_mean_dict = {idx: float(sigma_mean[idx]) for idx in bin_indices}
            except KeyError as e:
                raise ValueError(f"sigma_mean is missing bin index {e.args[0]}.") from e
        else:
            arr = np.asarray(sigma_mean, dtype=float)
            if arr.shape[0] != len(bin_indices):
                raise ValueError(
                    "sigma_mean sequence must have length equal to number of bins."
                )
            sigma_mean_dict = {
                idx: float(val) for idx, val in zip(bin_indices, arr, strict=True)
            }

    inferred_count_per_bin: dict[int, float] | None = None
    if (
        sigma_mean_dict is None
        and count_per_bin is None
        and density_per_bin is not None
    ):
        if survey_area is None:
            raise ValueError("survey_area must be provided when using density_per_bin.")
        if survey_area <= 0.0:
            raise ValueError("survey_area must be positive.")
        inferred_count_per_bin = {}
        for idx in bin_indices:
            try:
                dens = float(density_per_bin[idx])
            except KeyError as e:
                raise ValueError(
                    f"density_per_bin is missing bin index {e.args[0]}."
                ) from e
            inferred_count_per_bin[idx] = dens * float(survey_area)

    for idx in bin_indices:
        nz_bin = bins[idx]
        mom = bin_moments(z, nz_bin)
        mean = mom["mean"]
        std = mom["std"]
        entry: dict[str, float] = {"mean": mean, "std": std}

        if sigma_mean_dict is not None:
            entry["sigma_mean"] = sigma_mean_dict[idx]
        elif count_per_bin is not None:
            try:
                count = float(count_per_bin[idx])
            except KeyError as e:
                raise ValueError(
                    f"count_per_bin is missing bin index {e.args[0]}."
                ) from e
            if count <= 0.0:
                raise ValueError(f"count_per_bin[{idx}] must be positive.")
            entry["sigma_mean"] = std / np.sqrt(count)
        elif inferred_count_per_bin is not None:
            count = float(inferred_count_per_bin[idx])
            if count <= 0.0:
                raise ValueError(
                    f"Inferred count for bin {idx} is non-positive ({count})."
                )
            entry["sigma_mean"] = std / np.sqrt(count)

        stats[idx] = entry

    return stats


def galaxy_fraction_per_bin(
    z: Any,
    bins: Mapping[int, Any],
) -> dict[int, float]:
    """Computes per-bin galaxy fractions from the integrated bin curves.

    This function assigns each bin a fraction based on the total weight of its
    curve n_i(z) integrated over z, divided by the summed weight over all bins.

    This is meaningful when the bin curves carry relative population information,
    for example unnormalized selection-weighted curves or absolute number-density
    curves. If each bin curve is a normalized probability density (its integral
    over z is 1), then the returned fractions will be close to equal by
    construction and will not represent relative bin populations.

    For these fractions to represent relative bin populations, the bin curves
    must encode relative populations (i.e., they should not all be individually
    normalized PDFs). If each bin curve integrates to 1, the returned fractions
    will be uniform by construction.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to n_i(z) values evaluated on z.

    Returns:
        Mapping from bin index to a fraction. The sum of fractions is close to 1,
        up to numerical integration error.

    Raises:
        ValueError: If the total integrated weight over all bins is not positive.
    """
    weights = integrate_bins(z, bins)
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError(
            f"Total integrated n(z) over all bins must be positive (got {total})."
        )

    vals = np.asarray(list(weights.values()), dtype=float)
    if vals.size > 0:
        frac_norm = float(np.mean(np.isclose(vals, 1.0, rtol=1e-2, atol=1e-3)))
        if frac_norm >= 0.8:
            raise ValueError(_MSG_NORMALIZED_BINS)

    return {idx: val / total for idx, val in weights.items()}


def galaxy_density_per_bin(
    z: Any,
    bins: Mapping[int, Any],
    density_total: float,
    *,
    frac_per_bin: Mapping[int, float] | None = None,
    normalize: bool = True,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> tuple[dict[int, float], dict[int, float]]:
    """Allocates a total effective surface density across bins.

    This function returns per-bin surface densities by splitting a total surface
    density across bins in proportion to per-bin weights.

    You can provide the weights explicitly using frac_per_bin. If frac_per_bin is
    provided, it is treated as non-negative bin weights and is renormalized to sum
    to 1 before allocation.

    If frac_per_bin is not provided, the weights are derived from the integral of
    each bin curve n_i(z) over z. In that case, the bin curves must carry relative
    population information (typically they are not individually normalized).
    If you want per-bin population fractions inferred from integrals, the bin curves
    must encode relative populations (i.e., they should not all be individually
    normalized PDFs).

    Conventions:
        - density_total and the returned densities are in galaxies per square
          arcminute.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to n_i(z) values evaluated on z.
        density_total: Total effective surface density in galaxies per square
            arcminute.
        frac_per_bin: Optional mapping from bin index to fractional weights used
            for the allocation.
        normalize: If True and frac_per_bin is provided, renormalize frac_per_bin
            to sum to 1 before allocating density_total. If False, frac_per_bin is
            used as-is (but must sum to 1 within tolerance).
        rtol: Relative tolerance used when checking whether frac_per_bin sums to 1
            when normalize is False.
        atol: Absolute tolerance used when checking whether frac_per_bin sums to 1
            when normalize is False.

    Returns:
        A tuple (density_per_bin, frac_per_bin), where density_per_bin maps bin
        index to allocated surface density, and frac_per_bin maps bin index to the
        fractional weights used for the allocation.

    Raises:
        ValueError: If bins is empty.
        ValueError: If density_total is not positive.
        ValueError: If frac_per_bin is missing any bin index or sums to a
            non-positive value.
        ValueError: If normalize is False and frac_per_bin does not sum to 1.
        ValueError: If the total weight inferred from integrals is not positive.
    """
    if density_total <= 0.0:
        raise ValueError("density_total must be positive.")
    if len(bins) == 0:
        raise ValueError("bins must not be empty.")

    z_arr = np.asarray(z, dtype=float)

    bin_indices = sorted(bins.keys())
    first_idx = bin_indices[0]
    _, _ = validate_axis_and_weights(z_arr, bins[first_idx])

    if frac_per_bin is not None:
        frac: dict[int, float] = {}
        for idx in bin_indices:
            try:
                frac[idx] = float(frac_per_bin[idx])
            except KeyError as e:
                raise ValueError(
                    f"frac_per_bin is missing bin index {e.args[0]}."
                ) from e

        total_frac = sum(frac.values())
        if total_frac <= 0.0:
            raise ValueError("Sum of frac_per_bin must be positive.")

        if normalize:
            frac = {idx: val / total_frac for idx, val in frac.items()}
        else:
            if not np.isclose(total_frac, 1.0, rtol=rtol, atol=atol):
                raise ValueError(
                    "normalize=False requires frac_per_bin weights to sum to 1 "
                    f"(got {total_frac})."
                )

        dens = {idx: float(density_total) * f for idx, f in frac.items()}
        return dens, frac

    integrals: dict[int, float] = {}
    for idx in bin_indices:
        _, nz_arr = validate_axis_and_weights(z_arr, bins[idx])
        integrals[idx] = float(np.trapezoid(nz_arr, x=z_arr))

    total_weight = float(sum(integrals.values()))
    if total_weight <= 0.0:
        raise ValueError("Total integral must be positive to allocate density.")

    vals = np.asarray(list(integrals.values()), dtype=float)
    if vals.size > 0:
        frac_norm = float(np.mean(np.isclose(vals, 1.0, rtol=1e-2, atol=1e-3)))
        if frac_norm >= 0.8:
            raise ValueError(_MSG_NORMALIZED_BINS)

    frac = {idx: val / total_weight for idx, val in integrals.items()}
    dens = {idx: float(density_total) * f for idx, f in frac.items()}
    return dens, frac


def galaxy_count_per_bin(
    density_per_bin: Mapping[int, float],
    survey_area: float,
) -> dict[int, float]:
    """Converts per-bin surface density to effective counts using a survey area.

    This function multiplies each per-bin surface density by the survey area to
    produce an effective count per bin.

    Conventions:
        - density_per_bin values are in galaxies per square arcminute.
        - survey_area is in square arcminutes.
        - returned counts are dimensionless effective counts.

    Args:
        density_per_bin: Mapping from bin index to surface density in galaxies
            per square arcminute.
        survey_area: Survey area in square arcminutes.

    Returns:
        Mapping from bin index to effective counts.

    Raises:
        ValueError: If survey_area is not positive.
        ValueError: If any density is negative.
    """
    if survey_area <= 0.0:
        raise ValueError("survey_area must be positive.")

    out: dict[int, float] = {}
    for idx, dens in density_per_bin.items():
        dens_f = float(dens)
        if dens_f < 0.0:
            raise ValueError(f"density_per_bin[{idx}] must be non-negative.")
        out[int(idx)] = dens_f * float(survey_area)

    return out


def bin_quantiles(z: Any, nz_bin: Any, qs: Sequence[float]) -> dict[float, float]:
    """Computes weighted quantiles of a single bin distribution.

    This function treats nz_bin as a nonnegative weight curve evaluated on z and
    returns the redshift values where the cumulative integral reaches the given
    fractions of the total weight.

    Args:
        z: One-dimensional redshift grid.
        nz_bin: Values of the bin distribution evaluated on z.
        qs: Quantiles to compute, as fractions between 0 and 1 (for example, 0.5
            for the median).

    Returns:
        Mapping from each requested quantile q to the corresponding redshift
        value.

    Raises:
        ValueError: If any q is outside [0, 1].
        ValueError: If the total weight is not positive.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz_bin)
    cdf, norm = cdf_from_curve(z_arr, nz_arr)

    out: dict[float, float] = {}
    for q in qs:
        out[float(q)] = weighted_quantile_from_cdf(z_arr, cdf, norm, float(q))
    return out


def in_range_fraction(z: Any, nz_bin: Any, z_min: float, z_max: float) -> float:
    """Computes the fraction of a bin distribution contained in a redshift range.

    This function integrates nz_bin over the full z grid and over the interval
    [z_min, z_max], and returns the ratio (range weight / total weight).

    Args:
        z: One-dimensional redshift grid.
        nz_bin: Values of the bin distribution evaluated on z.
        z_min: Lower bound of the range.
        z_max: Upper bound of the range.

    Returns:
        Fraction of the total bin weight contained in [z_min, z_max].

    Raises:
        ValueError: If z_max is not greater than z_min.
        ValueError: If the total weight is not positive.
    """
    if not (z_max > z_min):
        raise ValueError("z_max must be greater than z_min.")

    z_arr, nz_arr = validate_axis_and_weights(z, nz_bin)
    total = float(np.trapezoid(nz_arr, x=z_arr))
    if total <= 0.0:
        raise ValueError("Total weight must be positive.")

    mask = (z_arr >= float(z_min)) & (z_arr <= float(z_max))
    if mask.sum() < 2:
        return 0.0

    inside = float(np.trapezoid(nz_arr[mask], x=z_arr[mask]))
    return float(inside / total)


def in_range_fraction_per_bin(
    z: Any,
    bins: Mapping[int, Any],
    bin_edges: Mapping[int, tuple[float, float]] | Sequence[float],
) -> dict[int, float]:
    """Computes the in-range fraction for each bin given nominal edges.

    You can provide bin edges in either form:
        - a mapping {bin_idx: (z_min, z_max)}, or
        - a sequence of edges [e0, e1, ..., eN] where bin i uses [e_i, e_{i+1}].

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin distributions evaluated on z.
        bin_edges: Nominal edges, as a mapping or an edge sequence.

    Returns:
        Mapping from bin index to the fraction of its weight inside its
        nominal range.

    Raises:
        ValueError: If required edges are missing.
    """
    indices = sorted(bins.keys())

    # Normalize bin_edges into a mapping idx -> (low, high)
    edges_map: dict[int, tuple[float, float]] = {}
    if isinstance(bin_edges, Mapping):
        for idx in indices:
            try:
                lo, hi = bin_edges[idx]
            except KeyError as e:
                raise ValueError(f"bin_edges is missing bin index {e.args[0]}.") from e
            edges_map[int(idx)] = (float(lo), float(hi))
    else:
        edges_arr = np.asarray(bin_edges, dtype=float)
        if edges_arr.ndim != 1 or edges_arr.size < 2:
            raise ValueError("bin_edges must be a 1D sequence with length at least 2.")
        # assumes bins are 0..N-1 if using a sequence
        for idx in indices:
            i = int(idx)
            if i + 1 >= edges_arr.size:
                raise ValueError("bin_edges sequence is too short for the bin indices.")
            edges_map[i] = (float(edges_arr[i]), float(edges_arr[i + 1]))

    return {i: in_range_fraction(z, bins[i], *edges_map[i]) for i in indices}


def peak_flags(
    z: Any,
    nz_bin: Any,
    *,
    min_rel_height: float = 0.1,
) -> dict[str, float]:
    """Computes simple peak diagnostics for a single bin distribution.

    This function identifies local maxima on the input grid and returns basic
    indicators of whether the curve is single-peaked or has multiple peaks.

    A point is treated as a peak if it is greater than its immediate neighbors.
    Peaks below min_rel_height times the global maximum are ignored.

    Args:
        z: One-dimensional redshift grid.
        nz_bin: Values of the bin distribution evaluated on z.
        min_rel_height: Minimum peak height as a fraction of the global maximum.

    Returns:
        Dictionary with:
            - "mode": redshift value of the global maximum on the grid.
            - "mode_height": value of nz_bin at the mode.
            - "num_peaks": number of detected peaks passing the height cut.
            - "second_peak_ratio": second-highest peak height divided by the
              highest (0 if there is no second peak).

    Raises:
        ValueError: If nz_bin is empty or invalid.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz_bin)

    if nz_arr.size < 3:
        mode = float(z_arr[int(np.nanargmax(nz_arr))])
        height = float(np.nanmax(nz_arr))
        return {
            "mode": mode,
            "mode_height": height,
            "num_peaks": 1.0,
            "second_peak_ratio": 0.0,
        }

    maxv = float(np.nanmax(nz_arr))
    if not np.isfinite(maxv) or maxv <= 0.0:
        mode = float(z_arr[int(np.nanargmax(nz_arr))])
        return {
            "mode": mode,
            "mode_height": float(maxv),
            "num_peaks": 0.0,
            "second_peak_ratio": 0.0,
        }

    # Local maxima
    left = nz_arr[1:-1] > nz_arr[:-2]
    right = nz_arr[1:-1] >= nz_arr[2:]
    is_peak = left & right

    peak_idx = np.where(is_peak)[0] + 1
    if peak_idx.size == 0:
        mode_i = int(np.nanargmax(nz_arr))
        return {
            "mode": float(z_arr[mode_i]),
            "mode_height": float(nz_arr[mode_i]),
            "num_peaks": 0.0,
            "second_peak_ratio": 0.0,
        }

    heights = nz_arr[peak_idx]
    keep = heights >= float(min_rel_height) * maxv
    peak_idx = peak_idx[keep]
    heights = heights[keep]

    if heights.size == 0:
        mode_i = int(np.nanargmax(nz_arr))
        return {
            "mode": float(z_arr[mode_i]),
            "mode_height": float(nz_arr[mode_i]),
            "num_peaks": 0.0,
            "second_peak_ratio": 0.0,
        }

    order = np.argsort(heights)[::-1]
    heights_sorted = heights[order]
    mode_i = int(np.nanargmax(nz_arr))

    second_ratio = 0.0
    if heights_sorted.size >= 2 and heights_sorted[0] > 0.0:
        second_ratio = float(heights_sorted[1] / heights_sorted[0])

    return {
        "mode": float(z_arr[mode_i]),
        "mode_height": float(nz_arr[mode_i]),
        "num_peaks": float(heights_sorted.size),
        "second_peak_ratio": float(second_ratio),
    }


def peak_flags_per_bin(
    z: Any,
    bins: Mapping[int, Any],
    *,
    min_rel_height: float = 0.1,
) -> dict[int, dict[str, float]]:
    """Computes peak diagnostics for each bin distribution."""
    return {
        int(i): peak_flags(z, bins[i], min_rel_height=min_rel_height)
        for i in sorted(bins.keys())
    }
