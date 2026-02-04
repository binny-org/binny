"""Pairwise distance/similarity metrics for curves or segment-mass vectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Literal, TypeAlias

import numpy as np

from binny.utils.types import FloatArray1D
from binny.utils.validators import (
    validate_axis_and_weights,
    validate_probability_vector,
    validate_same_shape,
)

PrepMode: TypeAlias = Literal["curves", "segments_prob"]
NormMode: TypeAlias = Literal["none", "normalize", "check"]
MetricUnit: TypeAlias = Literal["fraction", "percent"]

__all__ = [
    "pair_min",
    "pair_cosine",
    "pair_js",
    "pair_hellinger",
    "pair_tv",
    "fill_symmetric",
    "segment_mass_probs",
    "apply_unit",
    "prepare_metric_inputs",
    "mass_per_segment",
]


def _pair_min_kernel(
    z: FloatArray1D,
    curves: Mapping[int, FloatArray1D],
    i: int,
    j: int,
) -> float:
    """Computes overlap as the integral of the pointwise minimum.

    This overlap score integrates ``min(p_i(z), p_j(z))`` over a shared grid
    using the trapezoid rule. It is commonly used to quantify how strongly
    two nonnegative distributions overlap (e.g., tomographic ``n_i(z)``
    curves), with larger values indicating more shared support.

    Args:
        z: 1D grid of nodes used for trapezoid integration.
        curves: Mapping from bin id to curve values evaluated on ``z``.
        i: First bin id.
        j: Second bin id.

    Returns:
        The overlap integral for bins ``i`` and ``j``.
    """
    return float(np.trapezoid(np.minimum(curves[i], curves[j]), x=z))


def _pair_cosine_kernel(
    z: FloatArray1D,
    curves: Mapping[int, FloatArray1D],
    norms: Mapping[int, float],
    i: int,
    j: int,
) -> float:
    """Computes cosine similarity under a trapezoid inner product.

    This similarity treats curves as functions on a shared grid and computes a
    cosine-like similarity using trapezoid integration to define the inner
    product and L2 norms, with L2 normrs being the square root of the
    integral of the squared curve. This is useful for comparing curve shapes
    while reducing sensitivity to overall scale; values near 1 indicate similar
    shapes, and values near 0 indicate near-orthogonality under the chosen
    inner product.

    Args:
        z: 1D grid of nodes used for trapezoid integration.
        curves: Mapping from bin id to curve values evaluated on ``z``.
        norms: Mapping from bin id to precomputed L2 norms under the trapezoid
            inner product.
        i: First bin id.
        j: Second bin id.

    Returns:
        Cosine similarity for bins ``i`` and ``j``. If either curve has zero
        norm under the trapezoid inner product, the similarity is defined to be
        zero.
    """
    denom = float(norms[i]) * float(norms[j])
    if denom == 0.0:
        return 0.0
    num = float(np.trapezoid(curves[i] * curves[j], x=z))
    return float(num / denom)


def _kl_base2(a: FloatArray1D, b: FloatArray1D) -> float:
    """Kullback–Leibler divergence D_KL(a || b) with base-2 logarithm.

    This computes the KL divergence from ``a`` to ``b`` using base-2 logarithms.
    It is a useful measure of how much information is lost when ``b`` is used
    to approximate ``a``. The function raises a ValueError if ``b`` has  zeros
    where ``a`` is positive, as the KL divergence is undefined in that case.

    Args:
        a: First probability vector.
        b: Second probability vector.

    Returns:
        The KL divergence D_KL(a || b) using base-2 logarithms.

    Raises:
        ValueError: If ``b`` has zeros where ``a`` is positive.
    """
    mask = a > 0.0
    if not np.all(b[mask] > 0.0):
        raise ValueError("KL divergence is undefined when b has zeros where a is positive.")
    return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))


def _pair_js_kernel(masses: Mapping[int, FloatArray1D], i: int, j: int) -> float:
    """Computes Jensen–Shannon distance between probability vectors.

    This distance compares two discrete probability vectors (e.g., per-segment
    mass probabilities) using Jensen–Shannon divergence and returns its
    square root. With base-2 logarithms, the resulting distance is bounded
    in ``[0, 1]`` and is symmetric.

    Args:
        masses: Mapping from bin id to 1D probability vectors.
        i: First bin id.
        j: Second bin id.

    Returns:
        Jensen–Shannon distance for bins ``i`` and ``j``.
    """
    a = validate_probability_vector(masses[i], name=f"masses[{i}]")
    b = validate_probability_vector(masses[j], name=f"masses[{j}]")
    validate_same_shape(a, b, name_a=f"masses[{i}]", name_b=f"masses[{j}]")

    m = 0.5 * (a + b)
    js_div = 0.5 * _kl_base2(a, m) + 0.5 * _kl_base2(b, m)
    return float(np.sqrt(max(js_div, 0.0)))  # in [0, 1] for log base 2


def _pair_hellinger_kernel(masses: Mapping[int, FloatArray1D], i: int, j: int) -> float:
    """Computes Hellinger distance between probability vectors.

    Hellinger distance is a bounded, symmetric distance on discrete probability
    vectors. It is often used for stable comparisons of distributions
    represented on a fixed set of bins or segments.

    Args:
        masses: Mapping from bin id to 1D probability vectors.
        i: First bin id.
        j: Second bin id.

    Returns:
        Hellinger distance for bins ``i`` and ``j``.
    """
    a = validate_probability_vector(masses[i], name=f"masses[{i}]")
    b = validate_probability_vector(masses[j], name=f"masses[{j}]")
    validate_same_shape(a, b, name_a=f"masses[{i}]", name_b=f"masses[{j}]")

    diff = np.sqrt(a) - np.sqrt(b)
    h2 = 0.5 * float(np.sum(diff * diff))
    return float(np.sqrt(max(h2, 0.0)))  # in [0, 1]


def _pair_tv_kernel(masses: Mapping[int, FloatArray1D], i: int, j: int) -> float:
    """Computes total variation distance between probability vectors.

    Total variation distance is half the L1 distance between two discrete
    probability vectors. L1 refers to the sum of absolute differences.
    For valid probability vectors, it is bounded in ``[0, 1]`` and gives an
    interpretable notion of distributional difference.

    Args:
        masses: Mapping from bin id to 1D probability vectors.
        i: First bin id.
        j: Second bin id.

    Returns:
        Total variation distance for bins ``i`` and ``j``.
    """
    a = validate_probability_vector(masses[i], name=f"masses[{i}]")
    b = validate_probability_vector(masses[j], name=f"masses[{j}]")
    validate_same_shape(a, b, name_a=f"masses[{i}]", name_b=f"masses[{j}]")
    return float(0.5 * np.sum(np.abs(a - b)))  # in [0, 1]


def pair_min(
    z_arr: FloatArray1D,
    p: Mapping[int, FloatArray1D],
) -> Callable[[int, int], float]:
    """Computes overlap as the integral of the pointwise minimum.

    This overlap score integrates ``min(p_i(z), p_j(z))`` over a shared grid
    using the trapezoid rule. It is commonly used to quantify how strongly
    two nonnegative  distributions overlap (e.g., tomographic ``n_i(z)``
    bins), with larger values indicating more shared support.

    This function validates and caches the input curves once, and returns
    a callable suitable for repeated pairwise evaluation.

    Args:
        z_arr: 1D grid of nodes used for trapezoid integration.
        p: Mapping from bin id to curve values evaluated on ``z_arr``.

    Returns:
        A function ``f(i, j)`` that returns the overlap integral for bins
        ``i`` and ``j``.

    Raises:
        ValueError: If ``z_arr`` is not a valid strictly increasing 1D grid,
        or any curve is invalid on that grid.
        KeyError: If ``i`` or ``j`` is not present in ``p`` when evaluating
            the callable.
    """
    z, curves = prepare_metric_inputs(z_arr, p, mode="curves")
    return partial(_pair_min_kernel, z, curves)


def pair_cosine(z_arr: FloatArray1D, p: Mapping[int, FloatArray1D]) -> Callable[[int, int], float]:
    """Computes cosine similarity under a trapezoid inner product.

    This similarity treats curves as functions on a shared grid and computes
    a cosine-like similarity using trapezoid integration to define the inner
    product and L2 norms. It is useful for comparing curve shapes while
    reducing sensitivity to overall scale; values near 1 indicate similar
    shapes, and values near 0 indicate near-orthogonality under the chosen
    inner product.

    This function validates and caches the input curves once, precomputes
    per-curve norms, and returns a callable suitable for repeated pairwise
    evaluation.

    Args:
        z_arr: 1D grid of nodes used for trapezoid integration.
        p: Mapping from bin id to curve values evaluated on ``z_arr``.

    Returns:
        A function ``f(i, j)`` that returns cosine similarity for bins
        ``i`` and ``j``. If either curve has zero norm under the trapezoid
        inner product, the similarity is defined to be 0.

    Raises:
        ValueError: If ``z_arr`` is not a valid strictly increasing 1D grid,
            or any curve is invalid on that grid.
        KeyError: If ``i`` or ``j`` is not present in ``p`` when evaluating
            the callable.
    """
    z, curves = prepare_metric_inputs(z_arr, p, mode="curves")

    norms: dict[int, float] = {}
    for i, c in curves.items():
        n2 = float(np.trapezoid(c * c, x=z))
        norms[int(i)] = float(np.sqrt(max(n2, 0.0)))

    return partial(_pair_cosine_kernel, z, curves, norms)


def pair_js(masses: Mapping[int, FloatArray1D]) -> Callable[[int, int], float]:
    """Computes Jensen–Shannon distance between probability vectors.

    This distance compares two discrete probability vectors (e.g., per-segment
    mass probabilities) using Jensen–Shannon divergence and returns its square
    root. With base-2 logarithms, the resulting distance is bounded in
    ``[0, 1]`` and is symmetric. The returned callable validates inputs on
    each evaluation (shape/probability checks) using
    :func:`binny.utils.validators.validate_probability_vector`.

    Args:
        masses: Mapping from bin id to 1D probability vectors.

    Returns:
        A function ``f(i, j)`` that returns Jensen–Shannon distance for
        bins ``i`` and ``j``.

    Raises:
        KeyError: If ``i`` or ``j`` is not present in ``masses`` when
            evaluating the callable.
        ValueError: If either vector is not a valid probability vector,
            or shapes differ.
    """
    return partial(_pair_js_kernel, masses)


def pair_hellinger(
    masses: Mapping[int, FloatArray1D],
) -> Callable[[int, int], float]:
    """Computes Hellinger distance between probability vectors.

    Hellinger distance is a bounded, symmetric distance on discrete probability
    vectors. It is often used for stable comparisons of distributions
    represented on a fixed set of bins or segments. The returned callable
    validates inputs on each evaluation (shape/probability checks) using
    :func:`binny.utils.validators.validate_probability_vector`.

    Args:
        masses: Mapping from bin id to 1D probability vectors.

    Returns:
        A function ``f(i, j)`` that returns Hellinger distance for bins
        ``i`` and ``j``.

    Raises:
        KeyError: If ``i`` or ``j`` is not present in ``masses`` when
            evaluating the callable.
        ValueError: If either vector is not a valid probability vector,
            or shapes differ.
    """
    return partial(_pair_hellinger_kernel, masses)


def pair_tv(masses: Mapping[int, FloatArray1D]) -> Callable[[int, int], float]:
    """Computes total variation distance between probability vectors.

    Total variation distance is half the L1 distance between two discrete
    probability vectors. For valid probability vectors, it is bounded in
    ``[0, 1]`` and gives an  interpretable notion of distributional difference.
    The returned callable validates inputs on each evaluation
    (shape/probability checks) using
    :func:`binny.utils.validators.validate_probability_vector`.

    Args:
        masses: Mapping from bin id to 1D probability vectors.

    Returns:
        A function ``f(i, j)`` that returns total variation distance for
            bins ``i`` and ``j``.

    Raises:
        KeyError: If ``i`` or ``j`` is not present in ``masses`` when
            evaluating the callable.
        ValueError: If either vector is not a valid probability vector,
            or shapes differ.
    """
    return partial(_pair_tv_kernel, masses)


def fill_symmetric(
    bin_indices: list[int],
    pair_value: Callable[[int, int], float],
) -> dict[int, dict[int, float]]:
    """Returns a symmetric nested-dict matrix from a pairwise value function.

    This helper evaluates a pairwise metric on a set of bin indices and stores
    the results in a symmetric nested dictionary. It computes the upper
    triangle (including the diagonal) and mirrors values to fill the lower
    triangle.

    Args:
        bin_indices: Bin ids to include in the output matrix.
        pair_value: Callable returning the metric value for a pair of bin ids.

    Returns:
        A nested dictionary ``out[i][j]`` containing the metric for each pair
        of bin ids.

    Raises:
        Exception: Propagates any exception raised by ``pair_value``
            during evaluation.
    """
    out: dict[int, dict[int, float]] = {i: {} for i in bin_indices}

    for i in bin_indices:
        for j in bin_indices:
            if j < i:
                continue
            v = float(pair_value(i, j))
            out[i][j] = v
            out[j][i] = v

    return out


def segment_mass_probs(
    z_arr: FloatArray1D,
    p: Mapping[int, FloatArray1D],
) -> dict[int, FloatArray1D]:
    """Returns per-segment mass probability vectors derived from sampled curves.

    This converts each curve into trapezoid masses per segment using
    :func:`binny.utils.normalization.mass_per_segment`, then normalizes the
    segment masses to a probability vector. The output is suitable for
    discrete probability-vector metrics such as Jensen–Shannon, Hellinger,
    and total variation distances.

    Args:
        z_arr: 1D grid of nodes used to define the trapezoid segments.
        p: Mapping from bin id to curve values evaluated on ``z_arr``.

    Returns:
        Mapping from bin id to 1D ``float64`` probability vectors over segments.

    Raises:
        ValueError: If ``z_arr`` or any curve is invalid.
        ValueError: If any curve yields non-positive total segment mass.
    """
    z, curves = prepare_metric_inputs(z_arr, p, mode="curves")

    masses: dict[int, FloatArray1D] = {}
    for i, curve in curves.items():
        m = mass_per_segment(z, curve)
        s = float(np.sum(m))
        if s <= 0.0:
            raise ValueError(f"bin {i} has non-positive mass on segments.")
        masses[int(i)] = (m / s).astype(np.float64, copy=False)
    return masses


def apply_unit(
    mat: dict[int, dict[int, float]],
    unit: MetricUnit,
) -> dict[int, dict[int, float]]:
    """Returns a unit-converted copy of a nested-dict metric matrix.

    This helper converts matrices expressed as fractions to percentages
    when requested, while preserving the nested-dict structure.

    Args:
        mat: Nested dictionary ``mat[i][j]`` of metric values.
        unit: Output unit, either ``"fraction"`` or ``"percent"``.

    Returns:
        A nested dictionary in the requested unit.

    Raises:
        ValueError: If ``unit`` is not ``"fraction"`` or ``"percent"``.
    """
    if unit == "fraction":
        return mat
    if unit == "percent":
        return {i: {j: 100.0 * v for j, v in row.items()} for i, row in mat.items()}
    raise ValueError('unit must be "fraction" or "percent".')


def prepare_metric_inputs(
    z_arr: FloatArray1D,
    p: Mapping[int, FloatArray1D],
    *,
    mode: PrepMode,
    curve_norm: NormMode = "none",
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> tuple[FloatArray1D, dict[int, FloatArray1D]]:
    """Prepares inputs for pairwise metrics (validate once; optionally normalize).

    This is a convenience wrapper that standardizes the common boilerplate for
    pairwise curve metrics:

    - Validates ``z_arr`` and each curve in ``p`` using
        :func:`validate_axis_and_weights`.
    - Optionally normalizes curves to unit trapezoid integral or
        checks they already are.
    - Optionally converts curves to per-segment probability vectors
      (segment masses normalized to sum to 1), suitable for discrete
      probability metrics.

    Args:
        z_arr: 1D strictly increasing grid of nodes.
        p: Mapping from id to curve values evaluated on ``z_arr``.
        mode: Output mode:
            - ``"curves"``: return validated (and possibly normalized) node curves.
            - ``"segments_prob"``: return per-segment mass *probability* vectors.
        curve_norm: How to treat curve normalization before any conversion:
            - ``"none"``: no normalization checks beyond basic validation.
            - ``"normalize"``: divide each curve by its trapezoid integral.
            - ``"check"``: require each curve integrates to 1 within tolerance.
        rtol: Relative tolerance for the unit-integral check when
            ``curve_norm="check"``.
        atol: Absolute tolerance for the unit-integral check when
            ``curve_norm="check"``.

    Returns:
        ``(z_arr, out)`` where ``z_arr`` is float64 and ``out`` maps ids to arrays:
        - For ``mode="curves"``: arrays have length ``len(z_arr)``.
        - For ``mode="segments_prob"``: arrays have length ``len(z_arr) - 1``
            and sum to 1.

    Raises:
        ValueError: If ``z_arr`` or any curve fails validation, if a curve has
            non-positive trapezoid integral (needed for normalize/check),
            if a check fails, or if a curve yields non-positive total
            segment mass in ``"segments_prob"`` mode.
    """
    z_arr = np.asarray(z_arr, dtype=float)

    curves: dict[int, FloatArray1D] = {}
    for idx, curve in p.items():
        _, c = validate_axis_and_weights(z_arr, curve)
        area = float(np.trapezoid(c, x=z_arr))

        if curve_norm == "normalize":
            if area <= 0.0:
                raise ValueError(f"bin {idx} has non-positive integral: {area}.")
            c = (c / area).astype(np.float64, copy=False)

        elif curve_norm == "check":
            if area <= 0.0:
                raise ValueError(f"bin {idx} has non-positive integral: {area}.")
            if not np.isclose(area, 1.0, rtol=rtol, atol=atol):
                raise ValueError(
                    f"bin {idx} does not appear normalized (integral={area}). "
                    "Set curve_norm='normalize' or curve_norm='none'."
                )
            c = c.astype(np.float64, copy=False)

        else:  # "none"
            c = c.astype(np.float64, copy=False)

        curves[int(idx)] = c

    if mode == "curves":
        return z_arr.astype(np.float64, copy=False), curves

    if mode == "segments_prob":
        probs: dict[int, FloatArray1D] = {}
        for idx, c in curves.items():
            m = mass_per_segment(z_arr, c)
            s = float(np.sum(m))
            if s <= 0.0:
                raise ValueError(f"bin {idx} has non-positive mass on segments.")
            probs[idx] = (m / s).astype(np.float64, copy=False)
        return z_arr.astype(np.float64, copy=False), probs

    raise ValueError('mode must be "curves" or "segments_prob".')


def mass_per_segment(
    z_arr: FloatArray1D,
    p_arr: FloatArray1D,
) -> FloatArray1D:
    """Returns trapezoid masses per grid segment for a curve sampled at nodes.

    This converts node values into per-interval masses using the trapezoid rule,
    which is useful for building cumulative masses, rebinning, or diagnostics that
    operate on segment contributions rather than node values.

    Args:
        z_arr: 1D array of grid nodes.
        p_arr: 1D array of curve values at the nodes.

    Returns:
        A ``float64`` array of length ``len(z_arr) - 1`` containing trapezoid masses
        for each adjacent node interval.

    Raises:
        ValueError: If inputs are not 1D arrays of the same length.
        ValueError: If ``z_arr`` is not strictly increasing.
    """
    z_arr = np.asarray(z_arr, dtype=float)
    p_arr = np.asarray(p_arr, dtype=float)

    if z_arr.ndim != 1 or p_arr.ndim != 1 or z_arr.size != p_arr.size:
        raise ValueError("z_arr and p_arr must be 1D arrays of the same length.")

    if not np.all(np.diff(z_arr) > 0):
        raise ValueError("z_arr must be strictly increasing.")

    dz = np.diff(z_arr)
    mass = 0.5 * (p_arr[:-1] + p_arr[1:]) * dz
    return mass.astype(np.float64, copy=False)
