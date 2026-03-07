"""Cross-bin comparison metrics for binned redshift distributions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

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
from binny.utils.validators import validate_axis_and_weights

__all__ = [
    "bin_overlap",
    "between_bin_overlap",
    "overlap_pairs",
    "between_overlap_pairs",
    "leakage_matrix",
    "between_interval_mass_matrix",
    "pearson_matrix",
    "between_pearson_matrix",
]

MetricUnit = Literal["fraction", "percent"]

# method spec: (needs_normalized_input, uses_segment_masses)
_SPECS: dict[str, tuple[bool, bool]] = {
    "min": (True, False),
    "cosine": (False, False),
    "js": (True, True),
    "hellinger": (True, True),
    "tv": (True, True),
}


def _validate_method(method: str) -> str:
    """Validates and normalizes a supported pairwise metric name.

    Args:
        method: Name of the pairwise metric.

    Returns:
        Lowercase normalized method name.

    Raises:
        ValueError: If ``method`` is not supported.
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
    """Returns the curve-normalization mode for metric preparation.

    Args:
        normalize: Whether the user requested normalization.
        requires_norm: Whether the chosen metric operates on normalized inputs.

    Returns:
        Normalization mode passed to :func:`prepare_metric_inputs`.
    """
    if requires_norm and normalize:
        return "normalize"
    return "none"


def _prepare_curve_inputs(
    z: Any,
    bins: Mapping[int, Any],
    *,
    normalize: bool,
    requires_norm: bool,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, list[int], dict[int, np.ndarray]]:
    """Prepares curve-valued metric inputs for one bin set.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin distributions evaluated on ``z``.
        normalize: Whether normalization was requested by the user.
        requires_norm: Whether the chosen metric operates on normalized inputs.
        rtol: Relative tolerance for the normalization check.
        atol: Absolute tolerance for the normalization check.

    Returns:
        Tuple ``(z_m, bin_indices, curves)`` of prepared redshift grid,
        sorted bin indices, and prepared curves.
    """
    z_arr = np.asarray(z, dtype=float)
    bin_indices = sorted(int(k) for k in bins.keys())
    curve_norm = _curve_norm_mode(normalize=normalize, requires_norm=requires_norm)

    z_m, curves = prepare_metric_inputs(
        z_arr,
        bins,
        mode="curves",
        curve_norm=curve_norm,
        rtol=rtol,
        atol=atol,
    )
    return z_m, bin_indices, curves


def _prepare_mass_inputs(
    z: Any,
    bins: Mapping[int, Any],
    *,
    normalize: bool,
    requires_norm: bool,
    rtol: float,
    atol: float,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Prepares segment-mass metric inputs for one bin set.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin distributions evaluated on ``z``.
        normalize: Whether normalization was requested by the user.
        requires_norm: Whether the chosen metric operates on normalized inputs.
        rtol: Relative tolerance for the normalization check.
        atol: Absolute tolerance for the normalization check.

    Returns:
        Tuple ``(bin_indices, masses)`` of sorted bin indices and prepared
        segment-mass probability vectors.
    """
    z_arr = np.asarray(z, dtype=float)
    bin_indices = sorted(int(k) for k in bins.keys())
    curve_norm = _curve_norm_mode(normalize=normalize, requires_norm=requires_norm)

    _, masses = prepare_metric_inputs(
        z_arr,
        bins,
        mode="segments_prob",
        curve_norm=curve_norm,
        rtol=rtol,
        atol=atol,
    )
    return bin_indices, masses


def _rectangular_from_pair_value(
    row_indices: list[int],
    col_indices: list[int],
    pair_value: Callable[[int, int], float],
) -> dict[int, dict[int, float]]:
    """Builds a rectangular nested mapping from a pairwise evaluator.

    Args:
        row_indices: Sorted indices for the output rows.
        col_indices: Sorted indices for the output columns.
        pair_value: Callable returning the value for a given ``(i, j)`` pair.

    Returns:
        Nested mapping ``mat[i][j]`` over all row/column index combinations.
    """
    out: dict[int, dict[int, float]] = {i: {} for i in row_indices}
    for i in row_indices:
        for j in col_indices:
            out[i][j] = float(pair_value(i, j))
    return out


def _pair_min_between(
    z: np.ndarray,
    curves_a: Mapping[int, np.ndarray],
    curves_b: Mapping[int, np.ndarray],
) -> Callable[[int, int], float]:
    """Builds a pointwise-minimum overlap evaluator between two bin sets.

    Args:
        z: One-dimensional redshift grid.
        curves_a: Mapping of first-sample curves evaluated on ``z``.
        curves_b: Mapping of second-sample curves evaluated on ``z``.

    Returns:
        Callable returning the pointwise-minimum overlap integral for a pair.
    """

    def evaluate(i: int, j: int) -> float:
        return float(np.trapezoid(np.minimum(curves_a[i], curves_b[j]), x=z))

    return evaluate


def _pair_cosine_between(
    z: np.ndarray,
    curves_a: Mapping[int, np.ndarray],
    curves_b: Mapping[int, np.ndarray],
) -> Callable[[int, int], float]:
    """Builds a cosine-similarity evaluator between two bin sets.

    Args:
        z: One-dimensional redshift grid.
        curves_a: Mapping of first-sample curves evaluated on ``z``.
        curves_b: Mapping of second-sample curves evaluated on ``z``.

    Returns:
        Callable returning the cosine similarity for a pair.
    """
    w = trapz_weights(z)

    def evaluate(i: int, j: int) -> float:
        fa = curves_a[i]
        fb = curves_b[j]

        num = float(np.sum(w * fa * fb))
        den_a = float(np.sum(w * fa * fa))
        den_b = float(np.sum(w * fb * fb))
        den = float(np.sqrt(max(den_a, 0.0) * max(den_b, 0.0)))

        if den == 0.0:
            return 0.0
        return float(num / den)

    return evaluate


def _pair_js_between(
    masses_a: Mapping[int, np.ndarray],
    masses_b: Mapping[int, np.ndarray],
) -> Callable[[int, int], float]:
    """Builds a Jensen--Shannon distance evaluator between two bin sets.

    Args:
        masses_a: Mapping of first-sample segment-mass probability vectors.
        masses_b: Mapping of second-sample segment-mass probability vectors.

    Returns:
        Callable returning the Jensen--Shannon distance for a pair.
    """

    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0.0
        if not np.any(mask):
            return 0.0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    def evaluate(i: int, j: int) -> float:
        p = masses_a[i]
        q = masses_b[j]
        m = 0.5 * (p + q)
        js_div = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
        return float(np.sqrt(max(js_div, 0.0)))

    return evaluate


def _pair_hellinger_between(
    masses_a: Mapping[int, np.ndarray],
    masses_b: Mapping[int, np.ndarray],
) -> Callable[[int, int], float]:
    """Builds a Hellinger-distance evaluator between two bin sets.

    Args:
        masses_a: Mapping of first-sample segment-mass probability vectors.
        masses_b: Mapping of second-sample segment-mass probability vectors.

    Returns:
        Callable returning the Hellinger distance for a pair.
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
    """Builds a total-variation-distance evaluator between two bin sets.

    Args:
        masses_a: Mapping of first-sample segment-mass probability vectors.
        masses_b: Mapping of second-sample segment-mass probability vectors.

    Returns:
        Callable returning the total variation distance for a pair.
    """

    def evaluate(i: int, j: int) -> float:
        p = masses_a[i]
        q = masses_b[j]
        return float(0.5 * np.sum(np.abs(p - q)))

    return evaluate


def _validate_same_grid(z_a: np.ndarray, z_b: np.ndarray) -> None:
    """Validates that two prepared redshift grids are identical.

    Args:
        z_a: First prepared redshift grid.
        z_b: Second prepared redshift grid.

    Raises:
        ValueError: If the two grids are not identical.
    """
    if z_a.shape != z_b.shape or not np.allclose(z_a, z_b, rtol=0.0, atol=0.0):
        raise ValueError("bins_a and bins_b must be evaluated on the same z grid.")


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
    """Computes a pairwise comparison matrix for binned redshift distributions.

    This function compares all correlations of bin distributions evaluated on a shared
    redshift grid and returns a symmetric matrix of values.

    Supported methods:

    * ``"min"``: Integral of the pointwise minimum of the two curves.
      If curves are normalized, values lie in [0, 1] and the diagonal is 1.
    * ``"cosine"``: Cosine similarity under a continuous inner product.
      For nonnegative curves, values lie in [0, 1], with 1 meaning identical up
      to overall scaling.
    * ``"js"``: Jensen--Shannon distance computed on segment-mass probability
      vectors. With normalized curves, values lie in [0, 1], with 0 meaning
      identical and 1 meaning maximally different under this metric.
    * ``"hellinger"``: Hellinger distance on segment-mass probability vectors
      (in [0, 1]).
    * ``"tv"``: Total variation distance on segment-mass probability vectors
      (in [0, 1]).

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin distributions evaluated on ``z``.
        method: Pairwise metric to compute.
        unit: Output units. If ``"percent"``, values are multiplied by 100.
        normalize: Whether to normalize curves before comparison.
        rtol: Relative tolerance for the normalization check.
        atol: Absolute tolerance for the normalization check.
        decimal_places: Rounding precision for output values.

    Returns:
        Nested mapping ``mat[i][j]`` giving the pairwise value between bins
        ``i`` and ``j``.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    if len(bins) == 0:
        return {}

    method_l = _validate_method(method)
    requires_norm, uses_masses = _SPECS[method_l]

    if uses_masses:
        bin_indices, masses = _prepare_mass_inputs(
            z,
            bins,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )

        pair_value: Callable[[int, int], float]
        if method_l == "js":
            pair_value = pair_js(masses)
        elif method_l == "hellinger":
            pair_value = pair_hellinger(masses)
        elif method_l == "tv":
            pair_value = pair_tv(masses)
        else:
            raise ValueError(f"method {method_l!r} is not supported for segment-mass metrics.")
    else:
        z_m, bin_indices, curves = _prepare_curve_inputs(
            z,
            bins,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )

        if method_l == "min":
            pair_value = pair_min(z_m, curves)
        else:
            pair_value = pair_cosine(z_m, curves)

    mat = fill_symmetric(bin_indices, pair_value)
    out = apply_unit(mat, unit)

    if decimal_places is None:
        return out

    return round_floats(out, decimal_places=decimal_places)


def between_bin_overlap(
    z: Any,
    bins_a: Mapping[int, Any],
    bins_b: Mapping[int, Any],
    *,
    method: str = "min",
    unit: MetricUnit = "fraction",
    normalize: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Computes a rectangular pairwise comparison matrix between two bin sets.

    This function compares all bin distributions from one tomographic sample
    against all bin distributions from another tomographic sample, assuming both
    are evaluated on a shared redshift grid. The output is generally rectangular
    rather than symmetric, since the two samples can contain different bin
    indices and different numbers of bins.

    Supported methods:

    * ``"min"``: Integral of the pointwise minimum of the two curves.
      If curves are normalized, values lie in [0, 1].
    * ``"cosine"``: Cosine similarity under a continuous inner product.
      For nonnegative curves, values lie in [0, 1], with 1 meaning identical up
      to overall scaling.
    * ``"js"``: Jensen--Shannon distance computed on segment-mass probability
      vectors. With normalized curves, values lie in [0, 1], with 0 meaning
      identical and larger values meaning more distinct distributions.
    * ``"hellinger"``: Hellinger distance on segment-mass probability vectors
      (in [0, 1]).
    * ``"tv"``: Total variation distance on segment-mass probability vectors
      (in [0, 1]).

    Args:
        z: One-dimensional redshift grid shared by both bin sets.
        bins_a: Mapping from first-sample bin index to bin distributions
            evaluated on ``z``.
        bins_b: Mapping from second-sample bin index to bin distributions
            evaluated on ``z``.
        method: Pairwise metric to compute.
        unit: Output units. If ``"percent"``, values are multiplied by 100.
        normalize: Whether to normalize curves before comparison.
        rtol: Relative tolerance for the normalization check.
        atol: Absolute tolerance for the normalization check.
        decimal_places: Rounding precision for output values.

    Returns:
        Nested mapping ``mat[i][j]`` giving the pairwise value between
        first-sample bin ``i`` and second-sample bin ``j``.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    if len(bins_a) == 0 or len(bins_b) == 0:
        return {}

    method_l = _validate_method(method)
    requires_norm, uses_masses = _SPECS[method_l]

    if uses_masses:
        row_indices, masses_a = _prepare_mass_inputs(
            z,
            bins_a,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )
        col_indices, masses_b = _prepare_mass_inputs(
            z,
            bins_b,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )

        pair_value: Callable[[int, int], float]
        if method_l == "js":
            pair_value = _pair_js_between(masses_a, masses_b)
        elif method_l == "hellinger":
            pair_value = _pair_hellinger_between(masses_a, masses_b)
        elif method_l == "tv":
            pair_value = _pair_tv_between(masses_a, masses_b)
        else:
            raise ValueError(f"method {method_l!r} is not supported for segment-mass metrics.")
    else:
        z_a, row_indices, curves_a = _prepare_curve_inputs(
            z,
            bins_a,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )
        z_b, col_indices, curves_b = _prepare_curve_inputs(
            z,
            bins_b,
            normalize=normalize,
            requires_norm=requires_norm,
            rtol=rtol,
            atol=atol,
        )

        _validate_same_grid(z_a, z_b)

        if method_l == "min":
            pair_value = _pair_min_between(z_a, curves_a, curves_b)
        else:
            pair_value = _pair_cosine_between(z_a, curves_a, curves_b)

    mat = _rectangular_from_pair_value(row_indices, col_indices, pair_value)
    out = apply_unit(mat, unit)

    if decimal_places is None:
        return out

    return round_floats(out, decimal_places=decimal_places)


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
    """Returns bin-index correlations passing a threshold in a chosen pairwise metric.

    This is a convenience wrapper around :func:`bin_overlap`. It computes the
    pairwise matrix and returns unique off-diagonal correlations ``(i, j)`` with
    ``i < j`` that pass the requested threshold.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin distributions evaluated on ``z``.
        threshold: Threshold to apply in the units specified by ``unit``.
        unit: Units used for both the overlap calculation and the threshold.
            Accepted values are ``"fraction"`` and ``"percent"``.
        method: Pairwise metric passed to :func:`bin_overlap`.
        direction: Whether to select values >= threshold (``"high"``) or
            <= threshold (``"low"``).
        normalize: Passed to :func:`bin_overlap`.
        rtol: Relative tolerance for normalization check (if needed).
        atol: Absolute tolerance for normalization check (if needed).
        decimal_places: Rounding precision for output values.

    Returns:
        List of (i, j, value) tuples with i < j, sorted by decreasing value for
        ``direction="high"`` and increasing value for ``direction="low"``.

    Raises:
        ValueError: If ``direction`` is not ``"high"`` or ``"low"``.
    """
    if direction not in {"high", "low"}:
        raise ValueError('direction must be "high" or "low".')

    values = bin_overlap(
        z,
        bins,
        method=method,
        unit=unit,
        normalize=normalize,
        rtol=rtol,
        atol=atol,
        decimal_places=None,
    )

    indices = sorted(int(k) for k in values.keys())
    out: list[tuple[int, int, float]] = []

    if direction == "high":
        for a, i in enumerate(indices):
            for j in indices[a + 1 :]:
                v = float(values[i][j])
                if v >= threshold:
                    out.append((i, j, v))
        out.sort(key=lambda t: t[2], reverse=True)
    else:
        for a, i in enumerate(indices):
            for j in indices[a + 1 :]:
                v = float(values[i][j])
                if v <= threshold:
                    out.append((i, j, v))
        out.sort(key=lambda t: t[2])

    if decimal_places is None:
        return out

    return [(i, j, float(np.round(v, decimal_places))) for (i, j, v) in out]


def between_overlap_pairs(
    z: Any,
    bins_a: Mapping[int, Any],
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
    """Returns between-sample bin pairs passing a threshold in a chosen metric.

    This is a convenience wrapper around :func:`between_bin_overlap`. It computes
    the rectangular pairwise matrix between two tomographic samples and returns
    all bin correlations ``(i, j)`` that pass the requested threshold.

    Args:
        z: One-dimensional redshift grid shared by both bin sets.
        bins_a: Mapping from first-sample bin index to bin distributions
            evaluated on ``z``.
        bins_b: Mapping from second-sample bin index to bin distributions
            evaluated on ``z``.
        threshold: Threshold to apply in the units specified by ``unit``.
        unit: Units used for both the metric calculation and the threshold.
            Accepted values are ``"fraction"`` and ``"percent"``.
        method: Pairwise metric passed to :func:`between_bin_overlap`.
        direction: Whether to select values >= threshold (``"high"``) or
            <= threshold (``"low"``).
        normalize: Passed to :func:`between_bin_overlap`.
        rtol: Relative tolerance for normalization check (if needed).
        atol: Absolute tolerance for normalization check (if needed).
        decimal_places: Rounding precision for output values.

    Returns:
        List of ``(i, j, value)`` tuples, where ``i`` is a first-sample bin
        index and ``j`` is a second-sample bin index. Results are sorted by
        decreasing value for ``direction="high"`` and increasing value for
        ``direction="low"``.

    Raises:
        ValueError: If ``direction`` is not ``"high"`` or ``"low"``.
    """
    if direction not in {"high", "low"}:
        raise ValueError('direction must be "high" or "low".')

    values = between_bin_overlap(
        z,
        bins_a,
        bins_b,
        method=method,
        unit=unit,
        normalize=normalize,
        rtol=rtol,
        atol=atol,
        decimal_places=None,
    )

    row_indices = sorted(int(k) for k in values.keys())
    out: list[tuple[int, int, float]] = []

    if direction == "high":
        for i in row_indices:
            for j, v in values[i].items():
                val = float(v)
                if val >= threshold:
                    out.append((i, int(j), val))
        out.sort(key=lambda t: t[2], reverse=True)
    else:
        for i in row_indices:
            for j, v in values[i].items():
                val = float(v)
                if val <= threshold:
                    out.append((i, int(j), val))
        out.sort(key=lambda t: t[2])

    if decimal_places is None:
        return out

    return [(i, j, float(np.round(v, decimal_places))) for (i, j, v) in out]


def leakage_matrix(
    z: Any,
    bins: Mapping[int, Any],
    bin_edges: Mapping[int, tuple[float, float]] | Sequence[float] | np.ndarray,
    *,
    unit: MetricUnit = "fraction",
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Computes a leakage/confusion matrix between bins based on nominal edges.

    The leakage matrix ``leak[i][j]`` gives the fraction of the total mass
    in bin ``i`` that lies within the edges of bin ``j``. The diagonal entries
    therefore give the completeness of each bin with respect to its nominal
    edges, while the off-diagonal entries give the contamination from other
    bins.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin distributions evaluated on ``z``.
        bin_edges: Either a mapping from bin index to (low, high) edges, or
            a sequence/array of edges where bin ``i`` has edges
            ``(bin_edges[i], bin_edges[i+1])``.
        unit: Output units. If ``"percent"``, values are multiplied by 100.
        decimal_places: Rounding precision for output values.

    Returns:
        Nested mapping ``leak[i][j]`` giving the fraction of mass in bin ``i``
        that lies within the edges of bin ``j``.

    Raises:
        ValueError: If a bin has non-positive total mass.
        ValueError: If bin edges are invalid (hi <= lo).
        ValueError: If ``unit`` is not supported.
    """
    if unit not in {"percent", "fraction"}:
        raise ValueError('unit must be "fraction" or "percent".')

    z_arr = np.asarray(z, dtype=float)
    if len(bins) == 0:
        return {}

    bin_indices = sorted(int(k) for k in bins.keys())
    edges_map = normalize_edges(bin_indices, bin_edges)

    curves: dict[int, np.ndarray] = {}
    for i in bin_indices:
        _, nz_arr = validate_axis_and_weights(z_arr, bins[i])
        curves[i] = nz_arr.astype(float, copy=False)

    leak: dict[int, dict[int, float]] = {i: {} for i in bin_indices}

    for i in bin_indices:
        total = float(np.trapezoid(curves[i], x=z_arr))
        if total <= 0.0:
            raise ValueError(f"bin {i} has non-positive total mass: {total}.")

        for j in bin_indices:
            lo, hi = edges_map[j]
            if not (hi > lo):
                raise ValueError(f"bin_edges[{j}] must satisfy hi > lo (got {lo}, {hi}).")

            mask = (z_arr >= lo) & (z_arr <= hi)
            if int(mask.sum()) < 2:
                frac = 0.0
            else:
                inside = float(np.trapezoid(curves[i][mask], x=z_arr[mask]))
                frac = inside / total

            leak[i][j] = float(frac)

    out = apply_unit(leak, unit)

    if decimal_places is None:
        return out

    return round_floats(out, decimal_places=decimal_places)


def between_interval_mass_matrix(
    z: Any,
    bins: Mapping[int, Any],
    target_edges: Mapping[int, tuple[float, float]] | Sequence[float] | np.ndarray,
    *,
    unit: MetricUnit = "fraction",
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Computes a rectangular interval-mass matrix against target bin edges.

    The interval-mass matrix ``mass[i][j]`` gives the fraction of the total mass
    in input bin ``i`` that lies within target interval ``j``. This is the
    between-sample analogue of a leakage matrix and is useful, for example,
    when asking how much of a source bin falls inside a lens-bin interval.

    Args:
        z: One-dimensional redshift grid shared by all input bins.
        bins: Mapping from input bin index to bin distributions evaluated on ``z``.
        target_edges: Either a mapping from target bin index to ``(low, high)``
            edges, or a sequence/array of edges where target bin ``j`` has edges
            ``(target_edges[j], target_edges[j+1])``.
        unit: Output units. If ``"percent"``, values are multiplied by 100.
        decimal_places: Rounding precision for output values.

    Returns:
        Nested mapping ``mass[i][j]`` giving the fraction of mass in input bin
        ``i`` that lies within target interval ``j``.

    Raises:
        ValueError: If a bin has non-positive total mass.
        ValueError: If target edges are invalid (hi <= lo).
        ValueError: If ``unit`` is not supported.
    """
    if unit not in {"percent", "fraction"}:
        raise ValueError('unit must be "fraction" or "percent".')

    z_arr = np.asarray(z, dtype=float)
    if len(bins) == 0:
        return {}

    input_indices = sorted(int(k) for k in bins.keys())

    if isinstance(target_edges, Mapping):
        target_indices = sorted(int(k) for k in target_edges.keys())
    else:
        target_edges_arr = np.asarray(target_edges, dtype=float)
        if target_edges_arr.ndim != 1 or len(target_edges_arr) < 2:
            raise ValueError("target_edges must define at least one interval.")
        target_indices = list(range(len(target_edges_arr) - 1))

    edges_map = normalize_edges(target_indices, target_edges)

    curves: dict[int, np.ndarray] = {}
    for i in input_indices:
        _, nz_arr = validate_axis_and_weights(z_arr, bins[i])
        curves[i] = nz_arr.astype(float, copy=False)

    out: dict[int, dict[int, float]] = {i: {} for i in input_indices}

    for i in input_indices:
        total = float(np.trapezoid(curves[i], x=z_arr))
        if total <= 0.0:
            raise ValueError(f"bin {i} has non-positive total mass: {total}.")

        for j in target_indices:
            lo, hi = edges_map[j]
            if not (hi > lo):
                raise ValueError(f"target_edges[{j}] must satisfy hi > lo (got {lo}, {hi}).")

            mask = (z_arr >= lo) & (z_arr <= hi)
            if int(mask.sum()) < 2:
                frac = 0.0
            else:
                inside = float(np.trapezoid(curves[i][mask], x=z_arr[mask]))
                frac = inside / total

            out[i][j] = float(frac)

    out = apply_unit(out, unit)

    if decimal_places is None:
        return out

    return round_floats(out, decimal_places=decimal_places)


def pearson_matrix(
    z: Any,
    bins: Mapping[int, Any],
    *,
    normalize: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Computes a trapezoid-weighted Pearson correlation matrix between bin curves.

    The Pearson correlation between two curves ``f(z)`` and ``g(z)`` is defined as

        corr(f, g) = cov(f, g) / (std(f) * std(g))

    where the covariance and standard deviations are computed using trapezoid
    integration weights over the redshift grid.

    Note: if ``normalize=True``, the comparison is in terms of shape correlations,
    since all curves are normalized to unit integral before computing the
    correlation. If ``normalize=False``, the correlation reflects both shape
    and amplitude similarities.

    Args:
        z: One-dimensional redshift grid shared by all bins.
        bins: Mapping from bin index to bin distributions evaluated on ``z``.
        normalize: Control normalization behavior. If ``True``, all bins
            are normalized before computing correlations.
        rtol: Relative tolerance for the normalization check.
        atol: Absolute tolerance for the normalization check.
        decimal_places: Rounding precision for output values.

    Returns:
        Nested mapping ``corr[i][j]`` giving the Pearson correlation between
        bins ``i`` and ``j``.

    Raises:
        ValueError: If a bin has non-positive integral when normalization is
            checked or performed.
    """
    if len(bins) == 0:
        return {}

    z_m, bin_indices, curves = _prepare_curve_inputs(
        z,
        bins,
        normalize=normalize,
        requires_norm=normalize,
        rtol=rtol,
        atol=atol,
    )

    w = trapz_weights(z_m)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise ValueError("Non-positive integration weights; check z grid.")

    mean: dict[int, float] = {}
    for i in bin_indices:
        mean[i] = float(np.sum(w * curves[i]) / wsum)

    std: dict[int, float] = {}
    for i in bin_indices:
        xi = curves[i] - mean[i]
        var = float(np.sum(w * xi * xi) / wsum)
        std[i] = float(np.sqrt(max(var, 0.0)))

    corr: dict[int, dict[int, float]] = {i: {} for i in bin_indices}

    for i in bin_indices:
        for j in bin_indices:
            if j < i:
                continue

            si = std[i]
            sj = std[j]
            if si == 0.0 or sj == 0.0:
                val = 0.0
            else:
                xi = curves[i] - mean[i]
                xj = curves[j] - mean[j]
                cov = float(np.sum(w * xi * xj) / wsum)
                val = float(cov / (si * sj))

            corr[i][j] = val
            corr[j][i] = val

    if decimal_places is None:
        return corr
    return round_floats(corr, decimal_places=decimal_places)


def between_pearson_matrix(
    z: Any,
    bins_a: Mapping[int, Any],
    bins_b: Mapping[int, Any],
    *,
    normalize: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    decimal_places: int | None = 2,
) -> dict[int, dict[int, float]]:
    """Computes a rectangular trapezoid-weighted Pearson matrix between two bin sets.

    The Pearson correlation between two curves ``f(z)`` and ``g(z)`` is defined as

        corr(f, g) = cov(f, g) / (std(f) * std(g))

    where the covariance and standard deviations are computed using trapezoid
    integration weights over the redshift grid.

    Unlike :func:`pearson_matrix`, this function compares two different
    tomographic samples and therefore returns a rectangular matrix
    ``corr[i][j]``, where ``i`` is from the first sample and ``j`` is from
    the second sample.

    Note: if ``normalize=True``, the comparison is in terms of shape correlations,
    since all curves are normalized to unit integral before computing the
    correlation. If ``normalize=False``, the correlation reflects both shape
    and amplitude similarities.

    Args:
        z: One-dimensional redshift grid shared by both bin sets.
        bins_a: Mapping from first-sample bin index to bin distributions
            evaluated on ``z``.
        bins_b: Mapping from second-sample bin index to bin distributions
            evaluated on ``z``.
        normalize: Whether to normalize curves before computing correlations.
        rtol: Relative tolerance for the normalization check.
        atol: Absolute tolerance for the normalization check.
        decimal_places: Rounding precision for output values.

    Returns:
        Nested mapping ``corr[i][j]`` giving the Pearson correlation between
        first-sample bin ``i`` and second-sample bin ``j``.

    Raises:
        ValueError: If either bin set contains a bin with non-positive integral
            when normalization is checked or performed.
        ValueError: If the two bin sets are not evaluated on the same z grid.
    """
    if len(bins_a) == 0 or len(bins_b) == 0:
        return {}

    z_a, row_indices, curves_a = _prepare_curve_inputs(
        z,
        bins_a,
        normalize=normalize,
        requires_norm=normalize,
        rtol=rtol,
        atol=atol,
    )
    z_b, col_indices, curves_b = _prepare_curve_inputs(
        z,
        bins_b,
        normalize=normalize,
        requires_norm=normalize,
        rtol=rtol,
        atol=atol,
    )

    _validate_same_grid(z_a, z_b)

    w = trapz_weights(z_a)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise ValueError("Non-positive integration weights; check z grid.")

    mean_a: dict[int, float] = {}
    for i in row_indices:
        mean_a[i] = float(np.sum(w * curves_a[i]) / wsum)

    mean_b: dict[int, float] = {}
    for j in col_indices:
        mean_b[j] = float(np.sum(w * curves_b[j]) / wsum)

    std_a: dict[int, float] = {}
    for i in row_indices:
        xi = curves_a[i] - mean_a[i]
        var = float(np.sum(w * xi * xi) / wsum)
        std_a[i] = float(np.sqrt(max(var, 0.0)))

    std_b: dict[int, float] = {}
    for j in col_indices:
        xj = curves_b[j] - mean_b[j]
        var = float(np.sum(w * xj * xj) / wsum)
        std_b[j] = float(np.sqrt(max(var, 0.0)))

    corr: dict[int, dict[int, float]] = {i: {} for i in row_indices}

    for i in row_indices:
        for j in col_indices:
            si = std_a[i]
            sj = std_b[j]
            if si == 0.0 or sj == 0.0:
                val = 0.0
            else:
                xi = curves_a[i] - mean_a[i]
                xj = curves_b[j] - mean_b[j]
                cov = float(np.sum(w * xi * xj) / wsum)
                val = float(cov / (si * sj))

            corr[i][j] = val

    if decimal_places is None:
        return corr
    return round_floats(corr, decimal_places=decimal_places)
