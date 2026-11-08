"""Compare tomographic redshift distributions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from binny.tomography.composition import align_bin_grids
from binny.utils.metadata import round_floats
from binny.utils.normalization import (
    normalize_edges,
    prepare_metric_inputs,
    trapz_weights,
)
from binny.utils.pairwise_metrics import (
    apply_unit,
    fill_symmetric,
    pair_cosine,
    pair_hellinger,
    pair_js,
    pair_min,
    pair_tv,
)
from binny.utils.types import MetricUnit
from binny.utils.validators import validate_axis_and_weights

__all__ = [
    "bin_overlap",
    "overlap_pairs",
    "leakage_matrix",
    "pearson_matrix",
    "between_bin_overlap",
    "between_overlap_pairs",
    "between_interval_mass_matrix",
    "between_pearson_matrix",
]

_SPECS: dict[str, tuple[bool, bool]] = {
    "min": (True, False),
    "cosine": (False, False),
    "js": (True, True),
    "hellinger": (True, True),
    "tv": (True, True),
}


def _validate_method(method: str) -> str:
    """Validate a pairwise comparison metric.

    Args:
        method: Name of the comparison metric.

    Returns:
        Lowercase metric name.

    Raises:
        ValueError: If the metric is not supported.
    """
    method_l = method.lower()
    if method_l not in _SPECS:
        raise ValueError('method must be "min", "cosine", "js", "hellinger", or "tv".')
    return method_l


def _curve_norm_mode(
    *,
    normalize: bool,
    requires_norm: bool,
) -> Literal["none", "normalize"]:
    """Select the normalization mode for bin distributions.

    Args:
        normalize: Whether normalization is requested.
        requires_norm: Whether the metric requires normalized curves.

    Returns:
        Normalization mode for the distributions.
    """
    return "normalize" if normalize and requires_norm else "none"


def _prepare_curve_inputs(
    z: Any,
    bins: Mapping[int, Any],
    *,
    normalize: bool,
    requires_norm: bool,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, list[int], dict[int, np.ndarray]]:
    """Prepare bin distributions for comparison.

    Args:
        z: Redshift grid.
        bins: Redshift distributions indexed by bin.
        normalize: Whether normalization is requested.
        requires_norm: Whether the metric requires normalized curves.
        rtol: Relative normalization tolerance.
        atol: Absolute normalization tolerance.

    Returns:
        Redshift grid, sorted bin indices, and prepared distributions.
    """
    indices = sorted(int(k) for k in bins)
    z_m, curves = prepare_metric_inputs(
        np.asarray(z, dtype=float),
        bins,
        mode="curves",
        curve_norm=_curve_norm_mode(
            normalize=normalize,
            requires_norm=requires_norm,
        ),
        rtol=rtol,
        atol=atol,
    )
    return z_m, indices, curves


def _prepare_mass_inputs(
    z: Any,
    bins: Mapping[int, Any],
    *,
    normalize: bool,
    requires_norm: bool,
    rtol: float,
    atol: float,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Prepare bin probabilities for distance comparisons.

    Args:
        z: Redshift grid.
        bins: Redshift distributions indexed by bin.
        normalize: Whether normalization is requested.
        requires_norm: Whether the metric requires normalized curves.
        rtol: Relative normalization tolerance.
        atol: Absolute normalization tolerance.

    Returns:
        Sorted bin indices and segment-mass probability distributions.
    """
    indices = sorted(int(k) for k in bins)
    _, masses = prepare_metric_inputs(
        np.asarray(z, dtype=float),
        bins,
        mode="segments_prob",
        curve_norm=_curve_norm_mode(
            normalize=normalize,
            requires_norm=requires_norm,
        ),
        rtol=rtol,
        atol=atol,
    )
    return indices, masses


def _rectangular_from_pair_value(
    row_indices: list[int],
    col_indices: list[int],
    pair_value: Callable[[int, int], float],
) -> dict[int, dict[int, float]]:
    """Build a comparison matrix between two sets of bins.

    Args:
        row_indices: Indices of the first bin set.
        col_indices: Indices of the second bin set.
        pair_value: Function returning a value for each bin pair.

    Returns:
        Nested dictionary of pairwise comparison values.
    """
    return {i: {j: float(pair_value(i, j)) for j in col_indices} for i in row_indices}


def _pair_min_between(
    z: np.ndarray,
    curves_a: Mapping[int, np.ndarray],
    curves_b: Mapping[int, np.ndarray],
) -> Callable[[int, int], float]:
    """Create a minimum-overlap comparison between two bin sets.

    Args:
        z: Shared redshift grid.
        curves_a: Distributions of the first bin set.
        curves_b: Distributions of the second bin set.

    Returns:
        Function returning the minimum overlap for a bin pair.
    """

    def evaluate(i: int, j: int) -> float:
        return float(np.trapezoid(np.minimum(curves_a[i], curves_b[j]), x=z))

    return evaluate


def _pair_cosine_between(
    z: np.ndarray,
    curves_a: Mapping[int, np.ndarray],
    curves_b: Mapping[int, np.ndarray],
) -> Callable[[int, int], float]:
    """Create a cosine-similarity comparison between two bin sets.

    Args:
        z: Shared redshift grid.
        curves_a: Distributions of the first bin set.
        curves_b: Distributions of the second bin set.

    Returns:
        Function returning the cosine similarity for a bin pair.
    """
    w = trapz_weights(z)

    def evaluate(i: int, j: int) -> float:
        fa = curves_a[i]
        fb = curves_b[j]
        num = float(np.sum(w * fa * fb))
        den_a = float(np.sum(w * fa * fa))
        den_b = float(np.sum(w * fb * fb))
        den = float(np.sqrt(max(den_a, 0.0) * max(den_b, 0.0)))
        return float(num / den) if den > 0.0 else 0.0

    return evaluate


def _pair_js_between(
    masses_a: Mapping[int, np.ndarray],
    masses_b: Mapping[int, np.ndarray],
) -> Callable[[int, int], float]:
    """Create a Jensen-Shannon distance comparison between bin sets.

    Args:
        masses_a: Bin probabilities of the first sample.
        masses_b: Bin probabilities of the second sample.

    Returns:
        Function returning the Jensen-Shannon distance for a bin pair.
    """

    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate the Kullback-Leibler divergence.

        Args:
            p: First probability distribution.
            q: Second probability distribution.

        Returns:
            Kullback-Leibler divergence.
        """
        mask = p > 0.0
        if not np.any(mask):
            return 0.0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    def evaluate(i: int, j: int) -> float:
        return_value: float
        p = masses_a[i]
        q = masses_b[j]
        m = 0.5 * (p + q)
        divergence = 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
        return_value = float(np.sqrt(max(divergence, 0.0)))
        return return_value

    return evaluate


def _pair_hellinger_between(
    masses_a: Mapping[int, np.ndarray],
    masses_b: Mapping[int, np.ndarray],
) -> Callable[[int, int], float]:
    """Create a Hellinger-distance comparison between bin sets.

    Args:
        masses_a: Bin probabilities of the first sample.
        masses_b: Bin probabilities of the second sample.

    Returns:
        Function returning the Hellinger distance for a bin pair.
    """

    def evaluate(i: int, j: int) -> float:
        p = masses_a[i]
        q = masses_b[j]
        return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

    return evaluate


def _pair_tv_between(
    masses_a: Mapping[int, np.ndarray],
    masses_b: Mapping[int, np.ndarray],
) -> Callable[[int, int], float]:
    """Create a total-variation comparison between bin sets.

    Args:
        masses_a: Bin probabilities of the first sample.
        masses_b: Bin probabilities of the second sample.

    Returns:
        Function returning the total variation distance for a bin pair.
    """

    def evaluate(i: int, j: int) -> float:
        return float(0.5 * np.sum(np.abs(masses_a[i] - masses_b[j])))

    return evaluate


def _format_matrix(
    matrix: dict[int, dict[int, float]],
    unit: MetricUnit | None,
    decimal_places: int | None,
) -> dict[int, dict[int, float]]:
    """Format a matrix of comparison values.

    Args:
        matrix: Matrix of comparison values.
        unit: Output unit, or None to keep the original values.
        decimal_places: Number of decimal places, or None.

    Returns:
        Matrix with the requested units and precision.
    """
    out = apply_unit(matrix, unit) if unit is not None else matrix
    if decimal_places is not None:
        return round_floats(out, decimal_places=decimal_places)
    return out


def _select_pairs(
    matrix: Mapping[int, Mapping[int, float]],
    *,
    threshold: float,
    direction: Literal["high", "low"],
    decimal_places: int | None,
    unique: bool,
) -> list[tuple[int, int, float]]:
    """Find bin pairs that meet a comparison threshold.

    Args:
        matrix: Matrix of pairwise comparison values.
        threshold: Value used to select bin pairs.
        direction: Select values above or below the threshold.
        decimal_places: Number of decimal places, or None.
        unique: Whether to include only unique off-diagonal pairs.

    Returns:
        Sorted list of matching bin pairs and their values.

    Raises:
        ValueError: If the threshold direction is invalid.
    """
    if direction not in {"high", "low"}:
        raise ValueError('direction must be "high" or "low".')

    pairs = [
        (int(i), int(j), float(value))
        for i, row in matrix.items()
        for j, value in row.items()
        if (not unique or i < j)
        and (value >= threshold if direction == "high" else value <= threshold)
    ]
    pairs.sort(key=lambda pair: pair[2], reverse=direction == "high")

    if decimal_places is not None:
        return [(i, j, float(np.round(value, decimal_places))) for i, j, value in pairs]
    return pairs


def _interval_mass(
    z: Any,
    bins: Mapping[int, Any],
    edges: Mapping[int, tuple[float, float]] | Sequence[float] | np.ndarray,
    *,
    unit: MetricUnit,
    decimal_places: int | None,
    edge_name: str,
    match_bin_indices: bool,
) -> dict[int, dict[int, float]]:
    """Measure the fraction of each bin within target intervals.

    Args:
        z: Redshift grid.
        bins: Redshift distributions indexed by bin.
        edges: Redshift intervals to measure.
        unit: Fraction or percent.
        decimal_places: Number of decimal places, or None.
        edge_name: Name used in validation messages.
        match_bin_indices: Whether intervals use the input bin indices.

    Returns:
        Matrix of mass fractions within each interval.

    Raises:
        ValueError: If units, intervals, or bin masses are invalid.
    """
    if unit not in {"fraction", "percent"}:
        raise ValueError('unit must be "fraction" or "percent".')

    if not bins:
        return {}

    z_arr = np.asarray(z, dtype=float)
    input_indices = sorted(int(k) for k in bins)

    if match_bin_indices:
        target_indices = input_indices
    elif isinstance(edges, Mapping):
        target_indices = sorted(int(k) for k in edges)
    else:
        edges_arr = np.asarray(edges, dtype=float)
        if edges_arr.ndim != 1 or len(edges_arr) < 2:
            raise ValueError(f"{edge_name} must define at least one interval.")
        target_indices = list(range(len(edges_arr) - 1))

    edges_map = normalize_edges(target_indices, edges)
    curves = {
        i: validate_axis_and_weights(z_arr, bins[i])[1].astype(float, copy=False)
        for i in input_indices
    }

    matrix: dict[int, dict[int, float]] = {i: {} for i in input_indices}

    for i in input_indices:
        total = float(np.trapezoid(curves[i], x=z_arr))
        if total <= 0.0:
            raise ValueError(f"bin {i} has non-positive total mass: {total}.")

        for j in target_indices:
            lo, hi = edges_map[j]
            if not hi > lo:
                raise ValueError(f"{edge_name}[{j}] must satisfy hi > lo (got {lo}, {hi}).")

            mask = (z_arr >= lo) & (z_arr <= hi)
            if int(mask.sum()) < 2:
                frac = 0.0
            else:
                inside = float(np.trapezoid(curves[i][mask], x=z_arr[mask]))
                frac = inside / total

            matrix[i][j] = float(frac)

    return _format_matrix(matrix, unit, decimal_places)


def _pearson_between_curves(
    z: np.ndarray,
    indices_a: list[int],
    curves_a: Mapping[int, np.ndarray],
    indices_b: list[int],
    curves_b: Mapping[int, np.ndarray],
) -> dict[int, dict[int, float]]:
    """Calculate Pearson correlations between two sets of bins.

    Args:
        z: Shared redshift grid.
        indices_a: Bin indices of the first sample.
        curves_a: Distributions of the first sample.
        indices_b: Bin indices of the second sample.
        curves_b: Distributions of the second sample.

    Returns:
        Matrix of Pearson correlation coefficients.

    Raises:
        ValueError: If the redshift integration weights are invalid.
    """
    w = trapz_weights(z)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise ValueError("Non-positive integration weights; check z grid.")

    means_a = {i: float(np.sum(w * curves_a[i]) / wsum) for i in indices_a}
    means_b = {j: float(np.sum(w * curves_b[j]) / wsum) for j in indices_b}

    centered_a = {i: curves_a[i] - means_a[i] for i in indices_a}
    centered_b = {j: curves_b[j] - means_b[j] for j in indices_b}

    std_a = {
        i: float(
            np.sqrt(
                max(
                    float(np.sum(w * centered_a[i] ** 2) / wsum),
                    0.0,
                )
            )
        )
        for i in indices_a
    }
    std_b = {
        j: float(
            np.sqrt(
                max(
                    float(np.sum(w * centered_b[j] ** 2) / wsum),
                    0.0,
                )
            )
        )
        for j in indices_b
    }

    matrix: dict[int, dict[int, float]] = {i: {} for i in indices_a}

    for i in indices_a:
        for j in indices_b:
            den = std_a[i] * std_b[j]
            if den == 0.0:
                value = 0.0
            else:
                cov = float(np.sum(w * centered_a[i] * centered_b[j]) / wsum)
                value = float(cov / den)

            matrix[i][j] = value

    return matrix


def bin_overlap(
    z: Any,
    bins: Mapping[int, Any],
    *,
    method: str = "min",
    unit: MetricUnit = "fraction",
    normalize: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Compare all pairs of bins within one tomographic sample.

    Args:
        z: Redshift grid.
        bins: Redshift distributions indexed by bin.
        method: Comparison metric: min, cosine, js, hellinger, or tv.
        unit: Fraction or percent.
        normalize: Whether to normalize the distributions.
        rtol: Relative normalization tolerance.
        atol: Absolute normalization tolerance.
        decimal_places: Number of decimal places, or None.

    Returns:
        Symmetric matrix of pairwise comparison values.

    Raises:
        ValueError: If the metric or input distributions are invalid.
    """
    if not bins:
        return {}

    method_l = _validate_method(method)
    requires_norm, uses_masses = _SPECS[method_l]

    if uses_masses:
        indices, masses = _prepare_mass_inputs(
            z,
            bins,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )
        evaluators = {
            "js": pair_js,
            "hellinger": pair_hellinger,
            "tv": pair_tv,
        }
        pair_value = evaluators[method_l](masses)
    else:
        z_m, indices, curves = _prepare_curve_inputs(
            z,
            bins,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )
        pair_value = pair_min(z_m, curves) if method_l == "min" else pair_cosine(z_m, curves)

    return _format_matrix(
        fill_symmetric(indices, pair_value),
        unit,
        decimal_places,
    )


def overlap_pairs(
    z: Any,
    bins: Mapping[int, Any],
    *,
    threshold: float = 10.0,
    unit: MetricUnit = "percent",
    method: str = "min",
    direction: Literal["high", "low"] = "high",
    normalize: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    decimal_places: int | None = 2,
) -> list[tuple[int, int, float]]:
    """Find bin pairs whose comparison values pass a threshold.

    Args:
        z: Redshift grid.
        bins: Redshift distributions indexed by bin.
        threshold: Minimum or maximum accepted value.
        unit: Fraction or percent, including the threshold.
        method: Comparison metric: min, cosine, js, hellinger, or tv.
        direction: Select values above or below the threshold.
        normalize: Whether to normalize the distributions.
        rtol: Relative normalization tolerance.
        atol: Absolute normalization tolerance.
        decimal_places: Number of decimal places, or None.

    Returns:
        Sorted list of matching bin pairs and their values.

    Raises:
        ValueError: If the direction, metric, or inputs are invalid.
    """
    if direction not in {"high", "low"}:
        raise ValueError('direction must be "high" or "low".')

    matrix = bin_overlap(
        z,
        bins,
        method=method,
        unit=unit,
        normalize=normalize,
        rtol=rtol,
        atol=atol,
        decimal_places=None,
    )
    return _select_pairs(
        matrix,
        threshold=threshold,
        direction=direction,
        decimal_places=decimal_places,
        unique=True,
    )


def leakage_matrix(
    z: Any,
    bins: Mapping[int, Any],
    bin_edges: Mapping[int, tuple[float, float]] | Sequence[float] | np.ndarray,
    *,
    unit: MetricUnit = "fraction",
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Measure how much of each bin lies within each nominal bin interval.

    Args:
        z: Redshift grid.
        bins: Redshift distributions indexed by bin.
        bin_edges: Nominal redshift intervals of the bins.
        unit: Fraction or percent.
        decimal_places: Number of decimal places, or None.

    Returns:
        Matrix where entry (i, j) gives the fraction of bin i
        within the nominal interval of bin j.

    Raises:
        ValueError: If the intervals, units, or bin masses are invalid.
    """
    return _interval_mass(
        z,
        bins,
        bin_edges,
        unit=unit,
        decimal_places=decimal_places,
        edge_name="bin_edges",
        match_bin_indices=True,
    )


def pearson_matrix(
    z: Any,
    bins: Mapping[int, Any],
    *,
    normalize: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Calculate Pearson correlations between bins in one sample.

    Args:
        z: Redshift grid.
        bins: Redshift distributions indexed by bin.
        normalize: Whether to normalize the distributions.
        rtol: Relative normalization tolerance.
        atol: Absolute normalization tolerance.
        decimal_places: Number of decimal places, or None.

    Returns:
        Symmetric matrix of Pearson correlation coefficients.

    Raises:
        ValueError: If the input distributions are invalid.
    """
    if not bins:
        return {}

    z_m, indices, curves = _prepare_curve_inputs(
        z,
        bins,
        normalize=normalize,
        requires_norm=normalize,
        rtol=rtol,
        atol=atol,
    )

    matrix = _pearson_between_curves(z_m, indices, curves, indices, curves)
    return _format_matrix(matrix, None, decimal_places)


def between_bin_overlap(
    z_a: Any,
    bins_a: Mapping[int, Any],
    z_b: Any,
    bins_b: Mapping[int, Any],
    *,
    method: str = "min",
    unit: MetricUnit = "fraction",
    normalize: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Compare bins from two tomographic samples.

    Args:
        z_a: Redshift grid of the first sample.
        bins_a: Distributions of the first sample.
        z_b: Redshift grid of the second sample.
        bins_b: Distributions of the second sample.
        method: Comparison metric: min, cosine, js, hellinger, or tv.
        unit: Fraction or percent.
        normalize: Whether to normalize the distributions.
        rtol: Relative normalization tolerance.
        atol: Absolute normalization tolerance.
        decimal_places: Number of decimal places, or None.

    Returns:
        Matrix comparing bins from the first and second samples.

    Raises:
        ValueError: If the metric or input distributions are invalid.
    """
    if not bins_a or not bins_b:
        return {}

    z, aligned_a, aligned_b = align_bin_grids(z_a, bins_a, z_b, bins_b)
    method_l = _validate_method(method)
    requires_norm, uses_masses = _SPECS[method_l]

    if uses_masses:
        rows, masses_a = _prepare_mass_inputs(
            z,
            aligned_a,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )
        cols, masses_b = _prepare_mass_inputs(
            z,
            aligned_b,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )
        evaluators = {
            "js": _pair_js_between,
            "hellinger": _pair_hellinger_between,
            "tv": _pair_tv_between,
        }
        pair_value = evaluators[method_l](masses_a, masses_b)
    else:
        z_m, rows, curves_a = _prepare_curve_inputs(
            z,
            aligned_a,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )
        _, cols, curves_b = _prepare_curve_inputs(
            z,
            aligned_b,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )
        pair_value = (
            _pair_min_between(z_m, curves_a, curves_b)
            if method_l == "min"
            else _pair_cosine_between(z_m, curves_a, curves_b)
        )

    matrix = _rectangular_from_pair_value(rows, cols, pair_value)
    return _format_matrix(matrix, unit, decimal_places)


def between_overlap_pairs(
    z_a: Any,
    bins_a: Mapping[int, Any],
    z_b: Any,
    bins_b: Mapping[int, Any],
    *,
    threshold: float = 10.0,
    unit: MetricUnit = "percent",
    method: str = "min",
    direction: Literal["high", "low"] = "high",
    normalize: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    decimal_places: int | None = 2,
) -> list[tuple[int, int, float]]:
    """Find bin pairs from two samples that pass a threshold.

    Args:
        z_a: Redshift grid of the first sample.
        bins_a: Distributions of the first sample.
        z_b: Redshift grid of the second sample.
        bins_b: Distributions of the second sample.
        threshold: Minimum or maximum accepted value.
        unit: Fraction or percent, including the threshold.
        method: Comparison metric: min, cosine, js, hellinger, or tv.
        direction: Select values above or below the threshold.
        normalize: Whether to normalize the distributions.
        rtol: Relative normalization tolerance.
        atol: Absolute normalization tolerance.
        decimal_places: Number of decimal places, or None.

    Returns:
        Sorted list of matching cross-sample bin pairs.

    Raises:
        ValueError: If the direction, metric, or inputs are invalid.
    """
    if direction not in {"high", "low"}:
        raise ValueError('direction must be "high" or "low".')

    matrix = between_bin_overlap(
        z_a,
        bins_a,
        z_b,
        bins_b,
        method=method,
        unit=unit,
        normalize=normalize,
        rtol=rtol,
        atol=atol,
        decimal_places=None,
    )
    return _select_pairs(
        matrix,
        threshold=threshold,
        direction=direction,
        decimal_places=decimal_places,
        unique=False,
    )


def between_interval_mass_matrix(
    z: Any,
    bins: Mapping[int, Any],
    target_edges: Mapping[int, tuple[float, float]] | Sequence[float] | np.ndarray,
    *,
    unit: MetricUnit = "fraction",
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Measure how much of each bin lies within target intervals.

    Args:
        z: Redshift grid.
        bins: Input redshift distributions.
        target_edges: Redshift intervals to compare against.
        unit: Fraction or percent.
        decimal_places: Number of decimal places, or None.

    Returns:
        Matrix where entry (i, j) gives the fraction of input bin i
        within target interval j.

    Raises:
        ValueError: If the intervals, units, or bin masses are invalid.
    """
    return _interval_mass(
        z,
        bins,
        target_edges,
        unit=unit,
        decimal_places=decimal_places,
        edge_name="target_edges",
        match_bin_indices=False,
    )


def between_pearson_matrix(
    z_a: Any,
    bins_a: Mapping[int, Any],
    z_b: Any,
    bins_b: Mapping[int, Any],
    *,
    normalize: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Calculate Pearson correlations between bins from two samples.

    Args:
        z_a: Redshift grid of the first sample.
        bins_a: Distributions of the first sample.
        z_b: Redshift grid of the second sample.
        bins_b: Distributions of the second sample.
        normalize: Whether to normalize the distributions.
        rtol: Relative normalization tolerance.
        atol: Absolute normalization tolerance.
        decimal_places: Number of decimal places, or None.

    Returns:
        Matrix of cross-sample Pearson correlation coefficients.

    Raises:
        ValueError: If the input distributions are invalid.
    """
    if not bins_a or not bins_b:
        return {}

    z, aligned_a, aligned_b = align_bin_grids(z_a, bins_a, z_b, bins_b)
    z_m, rows, curves_a = _prepare_curve_inputs(
        z,
        aligned_a,
        normalize=normalize,
        requires_norm=normalize,
        rtol=rtol,
        atol=atol,
    )
    _, cols, curves_b = _prepare_curve_inputs(
        z,
        aligned_b,
        normalize=normalize,
        requires_norm=normalize,
        rtol=rtol,
        atol=atol,
    )

    matrix = _pearson_between_curves(z_m, rows, curves_a, cols, curves_b)
    return _format_matrix(matrix, None, decimal_places)
