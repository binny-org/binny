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
    "overlap_pairs",
    "leakage_matrix",
    "pearson_matrix",
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

    This function compares all pairs of bin distributions evaluated on a shared
    redshift grid and returns a symmetric matrix of values.

    Supported methods:

    * ``"min"``: Integral of the pointwise minimum of the two curves.
      If curves are normalized, values lie in [0, 1] and the diagonal is 1.
    * ``"cosine"``: Cosine similarity under a continuous inner product.
      For nonnegative curves, values lie in [0, 1], with 1 meaning identical up
      to overall scaling.
    * ``"js"``: Jensen–Shannon distance computed on segment-mass probability
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
        normalize: Wheather to normalize curves before comparison.
        rtol: Relative tolerance for the normalization check.
        atol: Absolute tolerance for the normalization check.
        decimal_places: Rounding precision for output values.

    Returns:
        Nested mapping ``mat[i][j]`` giving the pairwise value between bins
        ``i`` and ``j``.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    z_arr = np.asarray(z, dtype=float)
    if len(bins) == 0:
        return {}

    method_l = method.lower()
    if method_l not in _SPECS:
        raise ValueError('method must be "min", "cosine", "js", "hellinger", or "tv".')

    requires_norm, uses_masses = _SPECS[method_l]
    bin_indices = sorted(int(k) for k in bins.keys())

    curve_norm: Literal["none", "normalize"] = "none"
    if requires_norm and normalize:
        curve_norm = "normalize"

    if uses_masses:
        _, masses = prepare_metric_inputs(
            z_arr,
            bins,
            mode="segments_prob",
            curve_norm=curve_norm,
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
        z_m, curves = prepare_metric_inputs(
            z_arr,
            bins,
            mode="curves",
            curve_norm=curve_norm,
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
    """Returns bin-index pairs passing a threshold in a chosen pairwise metric.

    This is a convenience wrapper around :func:`bin_overlap`. It computes the
    pairwise matrix and returns unique off-diagonal pairs ``(i, j)`` with
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
        return out

    for a, i in enumerate(indices):
        for j in indices[a + 1 :]:
            v = float(values[i][j])
            if v <= threshold:
                out.append((i, j, v))
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
            are normalized before computing correlations, raising an error
            if any already look normalized. If ``False``, bins that do not
            look normalized are normalized with a warning.
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
    z_arr = np.asarray(z, dtype=float)
    if len(bins) == 0:
        return {}

    bin_indices = sorted(int(k) for k in bins.keys())

    curve_norm: Literal["none", "normalize"] = "normalize" if normalize else "none"

    z_m, curves = prepare_metric_inputs(
        z_arr,
        bins,
        mode="curves",
        curve_norm=curve_norm,
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
