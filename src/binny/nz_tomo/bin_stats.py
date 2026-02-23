"""Basic statistics of binned redshift distributions.

This module provides two families of diagnostics for tomographic bin curves:

- **Shape statistics**: summaries that depend only on the *shape* of each bin curve
  (moments, quantiles, peaks, centers, in-range fractions). These are safe to compute
  even when each bin curve is individually normalized.

- **Population statistics**: quantities that depend on *relative bin populations*
  (fractions, optional surface densities and counts). In Binny, these should be taken
  from tomographic-builder metadata (photo-z / spec-z), not inferred from normalized
  bin curves.

The public entry points are :func:`shape_stats` and :func:`population_stats`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from binny.utils.metadata import round_floats
from binny.utils.normalization import cdf_from_curve, weighted_quantile_from_cdf
from binny.utils.validators import validate_axis_and_weights

__all__ = [
    "bin_moments",
    "bin_quantiles",
    "bin_centers",
    "in_range_fraction",
    "in_range_fraction_per_bin",
    "peak_flags",
    "peak_flags_per_bin",
    "galaxy_fraction_per_bin",
    "galaxy_count_per_bin",
    "galaxy_density_per_bin",
    "shape_stats",
    "population_stats",
]


def _bin_widths(bin_edges: np.ndarray) -> list[float]:
    """Compute bin widths from bin edges."""
    w = np.diff(np.asarray(bin_edges, dtype=float))
    return [float(x) for x in w]


def _width_summary(widths: list[float]) -> dict[str, float]:
    """Compute summary statistics for bin widths."""
    w = np.asarray(widths, dtype=float)
    return {
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
    }


def _equidistant_score(widths: list[float]) -> float:
    """Compute the equidistant width score."""
    w = np.asarray(widths, dtype=float)
    m = float(np.mean(w))
    if m <= 0.0:
        return 0.0
    return float(np.max(np.abs(w - m)) / m)


def bin_moments(z: Any, nz_bin: Any) -> dict[str, float]:
    """Compute standard shape summaries for a single bin curve.

    The input is a redshift grid ``z`` and a corresponding bin curve ``nz_bin``
    evaluated on that grid. The curve is treated as a nonnegative weight function;
    it does *not* need to be normalized.

    Returned keys:
        - ``"mean"``: weighted mean redshift.
        - ``"median"``: weighted median redshift.
        - ``"mode"``: redshift value on the input grid where ``nz_bin`` is maximal.
        - ``"std"``: weighted standard deviation.
        - ``"skewness"``: standardized third central moment (dimensionless).
        - ``"kurtosis"``: standardized fourth central moment minus 3 (dimensionless).
        - ``"iqr"``: interquartile range (q75 - q25).
        - ``"width_68"``: central 68% interval width (q84 - q16).

    Notes:
        - Percentile-based summaries depend on the sampling of the grid.
        - The mode is grid-dependent by definition.

    Args:
        z: One-dimensional redshift grid.
        nz_bin: Bin curve values evaluated on ``z``.

    Returns:
        Dictionary of summary statistics.

    Raises:
        ValueError: If the total weight (integral of ``nz_bin``) is not positive.
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
    cdf, norm2 = cdf_from_curve(z_arr, nz_arr)
    q16 = weighted_quantile_from_cdf(z_arr, cdf, norm2, 0.16)
    q25 = weighted_quantile_from_cdf(z_arr, cdf, norm2, 0.25)
    q50 = weighted_quantile_from_cdf(z_arr, cdf, norm2, 0.50)
    q75 = weighted_quantile_from_cdf(z_arr, cdf, norm2, 0.75)
    q84 = weighted_quantile_from_cdf(z_arr, cdf, norm2, 0.84)

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


def bin_quantiles(z: Any, nz_bin: Any, qs: Sequence[float]) -> dict[float, float]:
    """Compute weighted quantiles of a single bin curve.

    The curve ``nz_bin`` is treated as a nonnegative weight curve evaluated on ``z``.
    For each requested quantile ``q`` in ``[0, 1]``, this returns the redshift value
    where the cumulative integral reaches ``q`` of the total weight.

    Args:
        z: One-dimensional redshift grid.
        nz_bin: Bin curve values evaluated on ``z``.
        qs: Quantiles to compute, as fractions between 0 and 1.

    Returns:
        Mapping from each requested quantile to the corresponding redshift value.

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


def bin_centers(
    z: Any,
    bins: Mapping[int, Any],
    *,
    method: str = "mean",
    decimal_places: int | None = 2,
) -> dict[int, float]:
    """Compute one center value per tomographic bin.

    Supported methods:
        - ``"mean"``: weighted mean redshift.
        - ``"median"``: weighted median redshift.
        - ``"mode"``: grid-point mode.
        - ``"pXX"``: weighted percentile, where XX is between 0 and 100
            (e.g. ``"p50"``).

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin curves evaluated on ``z``.
        method: Center definition.
        decimal_places: If not None, round each returned center to this many decimal
            places. If None, return full-precision floats.

    Returns:
        Mapping from bin index to the chosen center value.

    Raises:
        ValueError: If method is invalid.
        ValueError: If any bin has non-positive total weight when needed.
    """
    method_l = method.lower()

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
        idx = int(bin_idx)

        if percentile is not None:
            q = percentile / 100.0
            val = float(bin_quantiles(z, distribution, [q])[q])
        else:
            stats = bin_moments(z, distribution)
            if method_l in {"mean", "median", "mode"}:
                key = "median" if method_l == "median" else method_l
                val = float(stats[key])
            else:
                raise ValueError('method must be "mean", "median", "mode", or "pXX".')

        centers[idx] = round(val, int(decimal_places)) if decimal_places is not None else val

    return centers


def in_range_fraction(z: Any, nz_bin: Any, z_min: float, z_max: float) -> float:
    """Compute the fraction of a bin curve contained in a redshift range.

    This integrates ``nz_bin`` over the full grid and over ``[z_min, z_max]`` and
    returns ``inside / total``.

    Args:
        z: One-dimensional redshift grid.
        nz_bin: Bin curve values evaluated on ``z``.
        z_min: Lower bound of the range.
        z_max: Upper bound of the range.

    Returns:
        Fraction of the total bin weight contained in ``[z_min, z_max]``.

    Raises:
        ValueError: If ``z_max <= z_min``.
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
    """Compute the in-range fraction for each bin given nominal edges.

    ``bin_edges`` can be either:
        - mapping ``{bin_idx: (z_min, z_max)}``, or
        - a sequence of edges ``[e0, e1, ..., eN]`` where bin ``i`` uses
          ``[e_i, e_{i+1}]``.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin curves evaluated on ``z``.
        bin_edges: Nominal edges mapping or edge sequence.

    Returns:
        Mapping from bin index to fraction of its weight inside its nominal range.

    Raises:
        ValueError: If required edges are missing or malformed.
    """
    indices = sorted(int(i) for i in bins.keys())

    edges_map: dict[int, tuple[float, float]] = {}
    if isinstance(bin_edges, Mapping):
        for idx in indices:
            try:
                lo, hi = bin_edges[idx]
            except KeyError as e:
                raise ValueError(f"bin_edges is missing bin index {e.args[0]}.") from e
            edges_map[idx] = (float(lo), float(hi))
    else:
        edges_arr = np.asarray(bin_edges, dtype=float)
        if edges_arr.ndim != 1 or edges_arr.size < 2:
            raise ValueError("bin_edges must be a 1D sequence with length at least 2.")
        for idx in indices:
            if idx + 1 >= edges_arr.size:
                raise ValueError("bin_edges sequence is too short for the bin indices.")
            edges_map[idx] = (float(edges_arr[idx]), float(edges_arr[idx + 1]))

    return {i: in_range_fraction(z, bins[i], *edges_map[i]) for i in indices}


def peak_flags(
    z: Any,
    nz_bin: Any,
    *,
    min_rel_height: float = 0.1,
) -> dict[str, float | None]:
    """Compute simple peak diagnostics for a single bin curve.

    A point is treated as a peak if it is greater than its immediate neighbors.
    Peaks below ``min_rel_height * global_max`` are ignored.

    Returned keys:
        - ``"mode"``: redshift of the global maximum on the grid.
        - ``"mode_height"``: curve value at the mode.
        - ``"num_peaks"``: number of detected peaks passing the height cut.
        - ``"second_peak_ratio"``: second-highest peak height divided by the highest
          (None if no second peak).
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz_bin)

    i0 = int(np.nanargmax(nz_arr))
    mode = float(z_arr[i0])
    maxv = float(nz_arr[i0])

    # Degenerate / invalid
    if not np.isfinite(maxv) or maxv <= 0.0:
        return {
            "mode": mode,
            "mode_height": float(maxv),
            "num_peaks": 0.0,
            "second_peak_ratio": None,
        }

    # Too short to define local maxima robustly; treat as single-peaked.
    if nz_arr.size < 3:
        return {
            "mode": mode,
            "mode_height": float(maxv),
            "num_peaks": 1.0,
            "second_peak_ratio": None,
        }

    # Candidate local maxima indices (1..n-2)
    left = nz_arr[1:-1] > nz_arr[:-2]
    right = nz_arr[1:-1] >= nz_arr[2:]
    peak_idx = np.where(left & right)[0] + 1

    if peak_idx.size == 0:
        return {
            "mode": mode,
            "mode_height": float(maxv),
            "num_peaks": 0.0,
            "second_peak_ratio": None,
        }

    heights = nz_arr[peak_idx]

    # Height cut relative to global maximum (safe under normalization)
    keep = heights >= float(min_rel_height) * maxv
    heights = heights[keep]

    if heights.size == 0:
        return {
            "mode": mode,
            "mode_height": float(maxv),
            "num_peaks": 0.0,
            "second_peak_ratio": None,
        }

    heights_sorted = np.sort(heights)[::-1]

    second_ratio: float | None = None
    if heights_sorted.size >= 2 and heights_sorted[0] > 0.0:
        second_ratio = float(heights_sorted[1] / heights_sorted[0])

    return {
        "mode": mode,
        "mode_height": float(maxv),
        "num_peaks": float(heights_sorted.size),
        "second_peak_ratio": second_ratio,
    }


def peak_flags_per_bin(
    z: Any,
    bins: Mapping[int, Any],
    *,
    min_rel_height: float = 0.1,
) -> dict[int, dict[str, float | None]]:
    """Compute peak diagnostics for each bin curve."""
    return {
        int(i): peak_flags(z, bins[i], min_rel_height=min_rel_height) for i in sorted(bins.keys())
    }


def galaxy_fraction_per_bin(meta: Mapping[str, Any]) -> dict[int, float]:
    """Extract and normalize per-bin population fractions from tomo metadata."""
    frac = meta.get("frac_per_bin", None)

    if frac is None and isinstance(meta.get("bins"), Mapping):
        frac = meta["bins"].get("frac_per_bin", None)

    if frac is None or not isinstance(frac, Mapping):
        raise ValueError("metadata must contain a mapping 'frac_per_bin'.")

    out: dict[int, float] = {int(k): float(v) for k, v in frac.items()}
    s = float(sum(out.values()))
    if s <= 0.0:
        raise ValueError("Sum of metadata frac_per_bin must be positive.")
    return {i: f / s for i, f in out.items()}


def galaxy_count_per_bin(
    density_per_bin: Mapping[int, float],
    survey_area: float,
) -> dict[int, float]:
    """Convert per-bin surface densities to effective counts using a survey area.

    Conventions:
        - ``density_per_bin`` values are in galaxies per square arcminute.
        - ``survey_area`` is in square arcminutes.
        - returned counts are dimensionless effective counts.

    Args:
        density_per_bin: Mapping from bin index to surface density (gal/arcmin^2).
        survey_area: Survey area (arcmin^2).

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
        d = float(dens)
        if d < 0.0:
            raise ValueError(f"density_per_bin[{idx}] must be non-negative.")
        out[int(idx)] = d * float(survey_area)
    return out


def galaxy_density_per_bin(
    metadata: Mapping[str, Any],
    density_total: float,
) -> dict[int, float]:
    """Computes per-bin surface densities from tomo metadata and total
    effective density..

    Args:
        metadata: Tomography metadata containing ``"frac_per_bin"``.
        density_total: Total effective surface density (gal/arcmin^2).

    Returns:
        Mapping {bin_idx: gal/arcmin^2}.

    Raises:
        ValueError: If density_total is negative.
        ValueError: If metadata does not contain valid fractions.
    """
    if density_total < 0.0:
        raise ValueError("density_total must be non-negative.")
    frac = galaxy_fraction_per_bin(metadata)
    return {i: float(density_total) * float(f) for i, f in frac.items()}


def shape_stats(
    z: Any,
    bins: Mapping[int, Any],
    *,
    center_method: str = "mean",
    decimal_places: int | None = 2,
    quantiles: Sequence[float] = (0.16, 0.50, 0.84),
    min_rel_height: float = 0.1,
    bin_edges: Mapping[int, tuple[float, float]] | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute shape-only summary statistics for tomographic bin curves.

    This function computes redshift-shape diagnostics that depend only on the
    per-bin curve shapes, not on relative bin populations. It is safe to call on
    individually normalized bin PDFs.

    Per bin, it returns:
        - moments from :func:`bin_moments`
        - quantiles from :func:`bin_quantiles`
        - peak diagnostics from :func:`peak_flags`
        - a single center value from :func:`bin_centers`

    Optionally, if ``bin_edges`` are supplied, it also returns the fraction of each
    bin curve contained within its nominal redshift range via
    :func:`in_range_fraction_per_bin`.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin curves evaluated on ``z``.
        center_method: Center definition ("mean", "median", "mode", or "pXX").
        decimal_places: If not None, round centers to this many decimal places.
        quantiles: Quantiles to compute per bin (fractions in [0, 1]).
        min_rel_height: Peak height threshold relative to the global maximum.
        bin_edges: Optional nominal edges for in-range fractions.

    Returns:
        Nested mapping with keys:
            - ``"centers"``: {bin_idx: center}
            - ``"peaks"``: {bin_idx: peak_flags_dict}
            - ``"per_bin"``: {bin_idx: {"moments": ..., "center": ...,
                "quantiles": ..., "peaks": ...}}
            - ``"in_range_fraction"``: {bin_idx: fraction} if ``bin_edges``
                is not None

    Raises:
        ValueError: If bins is empty.
        ValueError: If any bin has non-positive total weight.
        ValueError: If center_method is invalid.
        ValueError: If quantiles are outside [0, 1].
        ValueError: If bin_edges are malformed when provided.
    """
    indices = sorted(int(i) for i in bins.keys())
    if len(indices) == 0:
        raise ValueError("bins must not be empty.")

    z_arr = np.asarray(z, dtype=float)
    _ = validate_axis_and_weights(z_arr, bins[indices[0]])

    centers = bin_centers(z_arr, bins, method=center_method, decimal_places=decimal_places)
    peaks = peak_flags_per_bin(z_arr, bins, min_rel_height=min_rel_height)

    per_bin: dict[int, dict[str, Any]] = {}
    for i in indices:
        mom = bin_moments(z_arr, bins[i])
        qs = bin_quantiles(z_arr, bins[i], quantiles)

        tail_asymmetry: float | None = None
        if 0.16 in qs and 0.5 in qs and 0.84 in qs:
            left = float(qs[0.5] - qs[0.16])
            right = float(qs[0.84] - qs[0.5])
            tail_asymmetry = float(right / max(left, 1e-12))

        per_bin[i] = {
            "moments": mom,
            "center": float(centers[i]),
            "quantiles": qs,
            "peaks": peaks[i],
            "tail_asymmetry": tail_asymmetry,
        }

    out: dict[str, Any] = {
        "centers": centers,
        "peaks": peaks,
        "per_bin": per_bin,
    }

    if bin_edges is not None:
        out["in_range_fraction"] = in_range_fraction_per_bin(z_arr, bins, bin_edges)

        edges_info: dict[str, Any] = {}

        if isinstance(bin_edges, Mapping):
            # bin_edges is {bin_idx: (lo, hi)}
            widths_per_bin = {int(i): float(hi) - float(lo) for i, (lo, hi) in bin_edges.items()}
            widths = [widths_per_bin[i] for i in indices]  # ordered like bins
            edges_info["widths_per_bin"] = widths_per_bin
        else:
            # bin_edges is a 1D edge sequence [e0, e1, ..., eN]
            edges_arr = np.asarray(bin_edges, dtype=float)
            widths = _bin_widths(edges_arr)
            edges_info["widths"] = widths

        edges_info["width_summary"] = _width_summary(widths)
        edges_info["equidistant_score"] = _equidistant_score(widths)
        out["edges"] = edges_info

    return round_floats(out, decimal_places)


def population_stats(
    bins: Mapping[int, Any],
    metadata: Mapping[str, Any],
    *,
    density_total: float | None = None,
    survey_area: float | None = None,
    normalize_frac: bool = True,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    decimal_places: int | None = 2,
) -> dict[str, Any]:
    """Compute population / normalization statistics for tomographic bins.

    This function computes quantities that depend on *relative bin populations*,
    using metadata produced by Binny's tomo builders (photo-z / spec-z).

    The primary input is ``metadata["frac_per_bin"]`` (mapping bin index -> fraction).
    Fractions are always checked against the bin indices present in ``bins``.

    Optional survey-level allocation:

    - If ``density_total`` (gal/arcmin^2) is provided, return per-bin allocated
      surface densities.
    - If ``survey_area`` (arcmin^2) is also provided, return per-bin effective counts.

    Args:
        bins: Mapping from bin index to bin curves.
        metadata: Tomography metadata containing ``"frac_per_bin"``.
        density_total: Optional total effective surface density (gal/arcmin^2).
        survey_area: Optional survey area (arcmin^2). Requires ``density_total``.
        normalize_frac: If True, renormalize metadata fractions to sum to 1.
            If False, require they sum to 1 within (rtol, atol).
        rtol: Relative tolerance for sum-to-one checks when normalize_frac=False.
        atol: Absolute tolerance for sum-to-one checks when normalize_frac=False.
        decimal_places: Rounding precision for returned values.

    Returns:
        Mapping with keys:

        - ``"fractions"``: {bin_idx: fraction}

        And optionally:

        - ``"density_total"``: float
        - ``"density_per_bin"``: {bin_idx: gal/arcmin^2}
        - ``"survey_area"``: float
        - ``"count_per_bin"``: {bin_idx: effective count}

    Raises:
        ValueError: If bins is empty.
        ValueError: If metadata does not contain ``"frac_per_bin"``.
        ValueError: If metadata fractions are missing any bin index.
        ValueError: If survey_area is provided without density_total.
        ValueError: If fractions cannot be normalized or validated.
    """
    indices = sorted(int(i) for i in bins.keys())
    if len(indices) == 0:
        raise ValueError("bins must not be empty.")

    if survey_area is not None and density_total is None:
        raise ValueError("survey_area requires density_total to compute counts.")

    frac_meta = galaxy_fraction_per_bin(metadata)

    frac: dict[int, float] = {}
    for i in indices:
        if i not in frac_meta:
            raise ValueError(f"metadata frac_per_bin missing bin index {i}.")
        frac[i] = float(frac_meta[i])

    s = float(sum(frac.values()))
    if s <= 0.0:
        raise ValueError("Sum of per-bin fractions must be positive.")

    if not normalize_frac:
        if not np.isclose(s, 1.0, rtol=rtol, atol=atol):
            raise ValueError(f"normalize_frac=False requires fractions to sum to 1 (got {s}).")
    else:
        frac = {i: f / s for i, f in frac.items()}

    out: dict[str, Any] = {"fractions": frac}

    if density_total is not None:
        density_per_bin = {i: float(density_total) * frac[i] for i in indices}
        out["density_total"] = float(density_total)
        out["density_per_bin"] = density_per_bin

        if survey_area is not None:
            out["survey_area"] = float(survey_area)
            out["count_per_bin"] = galaxy_count_per_bin(density_per_bin, float(survey_area))

    return round_floats(out, decimal_places)
