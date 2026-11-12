"""Unit tests for mixed bin-edge invariances and validation."""

from __future__ import annotations

import numpy as np
import pytest

from binny.axes.mixed_edges import mixed_edges
from binny.binning.edges import mixed_edges_from_segments


def _assert_edges_ok(
    edges: np.ndarray,
    n_bins: int,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
) -> None:
    """Assert that edges are finite, increasing, and correctly bounded."""
    edges = np.asarray(edges, dtype=float)

    assert edges.ndim == 1
    assert edges.size == n_bins + 1
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0)

    if x_min is not None:
        assert edges[0] == pytest.approx(x_min, rel=0.0, abs=1e-12)

    if x_max is not None:
        assert edges[-1] == pytest.approx(x_max, rel=0.0, abs=1e-12)


def _grid_linear(
    n: int = 2001,
    x_min: float = 0.0,
    x_max: float = 2.0,
) -> np.ndarray:
    """Generate a uniformly sampled coordinate grid."""
    return np.linspace(x_min, x_max, n)


def _gaussian(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Generate an unnormalized Gaussian density."""
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _weights_pos(x: np.ndarray) -> np.ndarray:
    """Generate a strictly positive nonuniform density."""
    return 1.0 + 0.2 * _gaussian(
        x,
        mu=0.9 * (x.min() + x.max()),
        sig=0.3 * (x.max() - x.min()),
    )


def _chi_of_z(z: np.ndarray) -> np.ndarray:
    """Generate a strictly increasing toy comoving distance."""
    return z + 0.15 * z**2


def _segment(
    method: str,
    n_bins: int,
    lo: float,
    hi: float,
    *,
    params: dict | None = None,
) -> dict:
    """Construct a declarative mixed-segment specification."""
    return {
        "scheme": method,
        "n_bins": n_bins,
        "x_min": lo,
        "x_max": hi,
        "params": {} if params is None else params,
    }


def _resolve(
    segments: list[dict],
    *,
    axis: np.ndarray | None = None,
    density: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Resolve mixed segments using the public scheme-resolution layer."""
    if axis is None:
        axis = _grid_linear(
            x_min=min(segment["x_min"] for segment in segments),
            x_max=max(segment["x_max"] for segment in segments),
        )

    if density is None:
        density = np.ones_like(axis)

    return mixed_edges_from_segments(
        segments,
        axis=axis,
        density=density,
        **kwargs,
    )


def test_mixed_edges_method_alias_invariance() -> None:
    """Tests that equivalent scheme aliases produce identical boundaries."""
    segments_a = [_segment("eq_dist", 6, 0.0, 2.0)]
    segments_b = [_segment("equidistant", 6, 0.0, 2.0)]

    edges_a = _resolve(segments_a)
    edges_b = _resolve(segments_b)

    _assert_edges_ok(edges_a, 6, x_min=0.0, x_max=2.0)
    assert np.allclose(edges_a, edges_b, rtol=0.0, atol=1e-12)


def test_mixed_edges_is_deterministic() -> None:
    """Tests that identical inputs produce identical mixed boundaries."""
    segments = [
        _segment("equidistant", 3, 0.0, 1.0),
        _segment("equidistant", 2, 1.0, 2.0),
    ]

    edges_a = _resolve(segments)
    edges_b = _resolve(segments)

    assert np.array_equal(edges_a, edges_b)


def test_mixed_edges_segment_param_overrides_global_fallback() -> None:
    """Tests that segment-specific population weights override global weights."""
    axis = _grid_linear()
    global_weights = _weights_pos(axis)
    segment_weights = _gaussian(axis, 0.6, 0.2)

    segments = [
        _segment(
            "equal_number",
            6,
            0.0,
            2.0,
            params={
                "equal_number_axis": axis,
                "equal_number_weights": segment_weights,
            },
        )
    ]

    edges_override = _resolve(
        segments,
        axis=axis,
        density=global_weights,
        equal_number_axis=axis,
        equal_number_weights=global_weights,
    )

    edges_expected = _resolve(
        [_segment("equal_number", 6, 0.0, 2.0)],
        axis=axis,
        density=global_weights,
        equal_number_axis=axis,
        equal_number_weights=segment_weights,
    )

    edges_global = _resolve(
        [_segment("equal_number", 6, 0.0, 2.0)],
        axis=axis,
        density=global_weights,
        equal_number_axis=axis,
        equal_number_weights=global_weights,
    )

    assert np.allclose(edges_override, edges_expected, rtol=0.0, atol=5e-12)
    assert not np.allclose(edges_override, edges_global, rtol=0.0, atol=1e-4)


def test_mixed_edges_concatenation_boundary_dedup() -> None:
    """Tests that adjacent segments retain only one shared boundary."""
    segments = [
        _segment("equidistant", 3, 0.0, 1.0),
        _segment("equidistant", 2, 1.0, 2.0),
    ]

    edges = _resolve(segments)

    _assert_edges_ok(edges, 5, x_min=0.0, x_max=2.0)
    assert np.sum(np.isclose(edges, 1.0, rtol=0.0, atol=1e-12)) == 1


def test_mixed_edges_translation_invariance_for_interval_methods() -> None:
    """Tests that translating intervals translates equidistant boundaries."""
    shift = 3.7

    segments = [
        _segment("equidistant", 4, 0.0, 2.0),
        _segment("equidistant", 3, 2.0, 5.0),
    ]

    shifted = [
        _segment("equidistant", 4, shift, 2.0 + shift),
        _segment("equidistant", 3, 2.0 + shift, 5.0 + shift),
    ]

    edges = _resolve(segments)
    edges_shifted = _resolve(shifted)

    assert np.allclose(edges_shifted, edges + shift, rtol=0.0, atol=1e-12)


def test_mixed_edges_scale_invariance_for_interval_methods() -> None:
    """Tests that scaling intervals scales equidistant boundaries."""
    scale = 2.5

    segments = [
        _segment("equidistant", 4, 0.0, 2.0),
        _segment("equidistant", 3, 2.0, 5.0),
    ]

    scaled = [
        _segment("equidistant", 4, 0.0, 2.0 * scale),
        _segment("equidistant", 3, 2.0 * scale, 5.0 * scale),
    ]

    edges = _resolve(segments)
    edges_scaled = _resolve(scaled)

    assert np.allclose(edges_scaled, scale * edges, rtol=0.0, atol=1e-12)


def test_mixed_edges_equal_number_invariant_under_weight_scaling() -> None:
    """Tests that scaling population weights preserves equal-number boundaries."""
    axis = _grid_linear()
    weights = _gaussian(axis, 0.9, 0.25)
    segments = [_segment("equal_number", 6, 0.0, 2.0)]

    edges_a = _resolve(segments, axis=axis, density=weights)
    edges_b = _resolve(segments, axis=axis, density=7.3 * weights)

    _assert_edges_ok(edges_a, 6, x_min=0.0, x_max=2.0)
    assert np.allclose(edges_a, edges_b, rtol=0.0, atol=5e-12)


def test_mixed_edges_equal_information_invariant_under_density_scaling() -> None:
    """Tests that scaling information density preserves equal-information edges."""
    axis = _grid_linear()
    information = _gaussian(axis, 1.0, 0.35)
    segments = [_segment("equal_information", 5, 0.0, 2.0)]

    edges_a = _resolve(
        segments,
        axis=axis,
        info_density=information,
    )
    edges_b = _resolve(
        segments,
        axis=axis,
        info_density=11.0 * information,
    )

    _assert_edges_ok(edges_a, 5, x_min=0.0, x_max=2.0)
    assert np.allclose(edges_a, edges_b, rtol=0.0, atol=5e-12)


def test_mixed_edges_equal_number_invariant_under_joint_x_weight_scaling() -> None:
    """Tests that rescaling coordinates and density rescales equal-number edges."""
    axis = _grid_linear()
    weights = _gaussian(axis, 0.9, 0.25)

    scale = 2.0
    scaled_axis = axis * scale
    scaled_weights = _gaussian(
        scaled_axis,
        0.9 * scale,
        0.25 * scale,
    )

    segments = [_segment("equal_number", 6, 0.0, 2.0)]
    scaled_segments = [_segment("equal_number", 6, 0.0, 2.0 * scale)]

    edges = _resolve(
        segments,
        axis=axis,
        density=weights,
    )
    scaled_edges = _resolve(
        scaled_segments,
        axis=scaled_axis,
        density=scaled_weights,
    )

    assert np.allclose(scaled_edges, scale * edges, rtol=0.0, atol=5e-12)


def test_mixed_edges_equidistant_chi_invariant_under_linear_chi_rescaling() -> None:
    """Tests that affine rescaling of comoving distance preserves chi-spaced edges."""
    redshift = _grid_linear()
    chi = _chi_of_z(redshift)

    scale = 3.1
    offset = 10.0

    # Equal spacing in chi is obtained by interpolating uniformly
    # spaced comoving-distance boundaries back to redshift.
    def chi_edges(distance: np.ndarray) -> np.ndarray:
        """Construct redshift edges uniformly spaced in comoving distance."""
        chi_boundaries = np.linspace(distance[0], distance[-1], 8)
        return np.interp(chi_boundaries, distance, redshift)

    edges_a = mixed_edges([chi_edges(chi)], total_n_bins=7)
    edges_b = mixed_edges(
        [chi_edges(scale * chi + offset)],
        total_n_bins=7,
    )

    _assert_edges_ok(edges_a, 7, x_min=0.0, x_max=2.0)
    assert np.allclose(edges_a, edges_b, rtol=0.0, atol=5e-12)


def test_mixed_edges_requires_matching_segment_boundaries() -> None:
    """Tests that mismatched adjacent segment boundaries raise an error."""
    segments = [
        np.linspace(0.0, 1.0, 4),
        np.linspace(1.1, 2.0, 3),
    ]

    with pytest.raises(ValueError, match="previous segment ends"):
        mixed_edges(segments, total_n_bins=5)


def test_mixed_edges_rejects_nonincreasing_edges_from_method() -> None:
    """Tests that nonincreasing segment boundaries are rejected."""
    segments = [np.array([1.0, 1.0, 1.0, 1.0])]

    with pytest.raises(ValueError, match="strictly increasing"):
        mixed_edges(segments, total_n_bins=3)


def test_mixed_edges_rejects_unknown_method() -> None:
    """Tests that the scheme resolver rejects unregistered binning methods."""
    segments = [
        _segment("this_is_not_a_method", 3, 0.0, 1.0),
    ]

    with pytest.raises(ValueError, match="Unknown|Unsupported"):
        _resolve(segments)


def test_mixed_edges_missing_required_global_or_segment_inputs_raises() -> None:
    """Tests that incomplete independent equal-number inputs raise an error."""
    axis = _grid_linear()
    segments = [_segment("equal_number", 3, 0.0, 2.0)]

    with pytest.raises(
        ValueError,
        match="Provide both equal_number_axis and equal_number_weights",
    ):
        _resolve(
            segments,
            axis=axis,
            equal_number_axis=axis,
        )


def test_mixed_edges_total_n_bins_validation() -> None:
    """Tests that the concatenator validates the expected number of bins."""
    segments = [
        np.linspace(0.0, 1.0, 3),
        np.linspace(1.0, 2.0, 3),
    ]

    with pytest.raises(ValueError, match="Expected 5 bins, got 4"):
        mixed_edges(segments, total_n_bins=5)
