"""Unit tests for binny.axes.mixed_edges invariances and validation."""

from __future__ import annotations

import numpy as np
import pytest

from binny.axes.mixed_edges import mixed_edges


def _assert_edges_ok(
    edges: np.ndarray,
    n_bins: int,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
) -> None:
    """Assert that edges array is well-formed and matches optional boundary checks."""
    edges = np.asarray(edges, dtype=float)
    assert edges.ndim == 1
    assert edges.size == n_bins + 1
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0)

    if x_min is not None:
        assert edges[0] == pytest.approx(float(x_min), rel=0.0, abs=1e-12)
    if x_max is not None:
        assert edges[-1] == pytest.approx(float(x_max), rel=0.0, abs=1e-12)


def _grid_linear(n: int = 2001, x_min: float = 0.0, x_max: float = 2.0) -> np.ndarray:
    """Generates a linear grid for testing."""
    return np.linspace(x_min, x_max, n)


def _grid_log(n: int = 2001, x_min: float = 1e-4, x_max: float = 2.0) -> np.ndarray:
    """Generates a logarithmic grid for testing."""
    return np.geomspace(float(x_min), float(x_max), n)


def _gaussian(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Generates Gaussian values for testing."""
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _weights_pos(x: np.ndarray) -> np.ndarray:
    """Generates a toy positive weight function for testing."""
    w = 1.0 + 0.2 * _gaussian(
        x, mu=0.9 * (x.min() + x.max()), sig=0.3 * (x.max() - x.min())
    )
    return np.asarray(w, dtype=float)


def _chi_of_z(z: np.ndarray) -> np.ndarray:
    """Generates a toy comoving distance function for testing."""
    return z + 0.15 * z**2


def _two_segment_specs(
    method0: str, method1: str, *, n0: int = 3, n1: int = 4
) -> list[dict]:
    """Generates a two-segment binning specification for testing."""
    return [
        {"method": method0, "n_bins": n0, "params": {}},
        {"method": method1, "n_bins": n1, "params": {}},
    ]


def test_mixed_edges_method_alias_invariance() -> None:
    """Tests that method aliases produce identical edges."""
    segs_a = [{"method": "eq", "n_bins": 6, "params": {"x_min": 0.0, "x_max": 2.0}}]
    segs_b = [
        {"method": "equidistant", "n_bins": 6, "params": {"x_min": 0.0, "x_max": 2.0}}
    ]

    ea = mixed_edges(segs_a, total_n_bins=6)
    eb = mixed_edges(segs_b, total_n_bins=6)

    assert np.allclose(ea, eb, rtol=0.0, atol=1e-12)


def test_mixed_edges_is_deterministic() -> None:
    """Tests that repeated calls with the same inputs yield identical edges."""
    segs = [
        {"method": "equidistant", "n_bins": 3, "params": {"x_min": 0.0, "x_max": 1.0}},
        {"method": "equidistant", "n_bins": 2, "params": {"x_min": 1.0, "x_max": 2.0}},
    ]
    e1 = mixed_edges(segs, total_n_bins=5)
    e2 = mixed_edges(segs, total_n_bins=5)
    assert np.array_equal(e1, e2)


def test_mixed_edges_segment_param_overrides_global_fallback() -> None:
    """Tests that segment-specific params override global inputs."""
    x = _grid_linear()
    w_global = _weights_pos(x)
    w_seg = 7.3 * w_global

    segments = [
        {
            "method": "equal_number",
            "n_bins": 6,
            "params": {"weights": w_seg},
        }
    ]

    e_with_override = mixed_edges(segments, x=x, weights=w_global, total_n_bins=6)

    # Equivalent to calling with the overridden weights as globals
    e_expected = mixed_edges(
        [{"method": "equal_number", "n_bins": 6, "params": {}}],
        x=x,
        weights=w_seg,
        total_n_bins=6,
    )

    assert np.allclose(e_with_override, e_expected, rtol=0.0, atol=5e-12)


def test_mixed_edges_concatenation_boundary_dedup() -> None:
    """Tests that shared boundaries between segments appear only once."""
    segments = [
        {"method": "equidistant", "n_bins": 3, "params": {"x_min": 0.0, "x_max": 1.0}},
        {"method": "equidistant", "n_bins": 2, "params": {"x_min": 1.0, "x_max": 2.0}},
    ]

    e = mixed_edges(segments, total_n_bins=5)

    _assert_edges_ok(e, 5, x_min=0.0, x_max=2.0)
    assert np.sum(np.isclose(e, 1.0, rtol=0.0, atol=1e-12)) == 1


def test_mixed_edges_translation_invariance_for_interval_methods() -> None:
    """Tests that translating the input interval translates the edges."""
    shift = 3.7
    segments0 = [
        {"method": "equidistant", "n_bins": 4, "params": {"x_min": 0.0, "x_max": 2.0}},
        {"method": "equidistant", "n_bins": 3, "params": {"x_min": 2.0, "x_max": 5.0}},
    ]
    segments_s = [
        {
            "method": "equidistant",
            "n_bins": 4,
            "params": {"x_min": 0.0 + shift, "x_max": 2.0 + shift},
        },
        {
            "method": "equidistant",
            "n_bins": 3,
            "params": {"x_min": 2.0 + shift, "x_max": 5.0 + shift},
        },
    ]

    e0 = mixed_edges(segments0, total_n_bins=7)
    e_s = mixed_edges(segments_s, total_n_bins=7)

    assert np.allclose(e_s, e0 + shift, rtol=0.0, atol=1e-12)


def test_mixed_edges_scale_invariance_for_interval_methods() -> None:
    """Tests that scaling the input interval scales the edges."""
    scale = 2.5
    segments0 = [
        {"method": "equidistant", "n_bins": 4, "params": {"x_min": 0.0, "x_max": 2.0}},
        {"method": "equidistant", "n_bins": 3, "params": {"x_min": 2.0, "x_max": 5.0}},
    ]
    segments_s = [
        {
            "method": "equidistant",
            "n_bins": 4,
            "params": {"x_min": 0.0 * scale, "x_max": 2.0 * scale},
        },
        {
            "method": "equidistant",
            "n_bins": 3,
            "params": {"x_min": 2.0 * scale, "x_max": 5.0 * scale},
        },
    ]

    e0 = mixed_edges(segments0, total_n_bins=7)
    e_s = mixed_edges(segments_s, total_n_bins=7)

    assert np.allclose(e_s, e0 * scale, rtol=0.0, atol=1e-12)


def test_mixed_edges_equal_number_invariant_under_weight_scaling() -> None:
    """Tests that scaling the weights does not change equal-number edges."""
    x = _grid_linear()
    w = _gaussian(x, 0.9, 0.25)

    e1 = mixed_edges(
        [{"method": "equal_number", "n_bins": 6, "params": {}}],
        x=x,
        weights=w,
        total_n_bins=6,
    )
    e2 = mixed_edges(
        [{"method": "equal_number", "n_bins": 6, "params": {}}],
        x=x,
        weights=7.3 * w,
        total_n_bins=6,
    )

    assert np.allclose(e1, e2, rtol=0.0, atol=5e-12)


def test_mixed_edges_equal_information_invariant_under_density_scaling() -> None:
    """Tests that scaling the information density does not change
    equal-information edges."""
    x = _grid_linear()
    info = _gaussian(x, 1.0, 0.35)

    e1 = mixed_edges(
        [{"method": "equal_information", "n_bins": 5, "params": {}}],
        x=x,
        info_density=info,
        total_n_bins=5,
    )
    e2 = mixed_edges(
        [{"method": "equal_information", "n_bins": 5, "params": {}}],
        x=x,
        info_density=11.0 * info,
        total_n_bins=5,
    )

    assert np.allclose(e1, e2, rtol=0.0, atol=5e-12)


def test_mixed_edges_equal_number_invariant_under_joint_x_weight_scaling() -> None:
    """Tests that scaling both x and weights scales equal-number edges."""
    x = _grid_linear()
    w = _gaussian(x, 0.9, 0.25)

    scale = 2.0
    xs = x * scale
    ws = _gaussian(xs, 0.9 * scale, 0.25 * scale)

    e = mixed_edges(
        [{"method": "equal_number", "n_bins": 6, "params": {}}],
        x=x,
        weights=w,
        total_n_bins=6,
    )
    es = mixed_edges(
        [{"method": "equal_number", "n_bins": 6, "params": {}}],
        x=xs,
        weights=ws,
        total_n_bins=6,
    )

    assert np.allclose(es, e * scale, rtol=0.0, atol=5e-12)


def test_mixed_edges_equidistant_chi_invariant_under_linear_chi_rescaling() -> None:
    """Tests that linear rescaling of chi does not change equidistant_chi edges."""
    z = _grid_linear()
    chi = _chi_of_z(z)

    a = 3.1
    b = 10.0

    e1 = mixed_edges(
        [{"method": "equidistant_chi", "n_bins": 7, "params": {}}],
        z=z,
        chi=chi,
        total_n_bins=7,
    )
    e2 = mixed_edges(
        [{"method": "equidistant_chi", "n_bins": 7, "params": {}}],
        z=z,
        chi=a * chi + b,
        total_n_bins=7,
    )

    assert np.allclose(e1, e2, rtol=0.0, atol=5e-12)


def test_mixed_edges_requires_matching_segment_boundaries() -> None:
    """Tests that mismatched segment boundaries raise an error."""
    segments = [
        {"method": "equidistant", "n_bins": 3, "params": {"x_min": 0.0, "x_max": 1.0}},
        {"method": "equidistant", "n_bins": 2, "params": {"x_min": 1.1, "x_max": 2.0}},
    ]

    with pytest.raises(ValueError, match="does not match previous right edge"):
        mixed_edges(segments, total_n_bins=5)


def test_mixed_edges_rejects_nonincreasing_edges_from_method() -> None:
    """Tests that non-increasing edges from a method raise an error."""
    segments = [
        {"method": "equidistant", "n_bins": 3, "params": {"x_min": 1.0, "x_max": 1.0}}
    ]
    with pytest.raises(ValueError):
        mixed_edges(segments, total_n_bins=3)


def test_mixed_edges_rejects_unknown_method() -> None:
    """Tests that an unknown method raises an error."""
    segments = [
        {
            "method": "this_is_not_a_method",
            "n_bins": 3,
            "params": {"x_min": 0.0, "x_max": 1.0},
        }
    ]
    with pytest.raises(ValueError, match="Unknown binning method"):
        mixed_edges(segments, total_n_bins=3)


def test_mixed_edges_missing_required_global_or_segment_inputs_raises() -> None:
    """Tests that missing required inputs raise an error."""
    segments = [{"method": "equal_number", "n_bins": 3, "params": {}}]
    with pytest.raises(
        ValueError, match="requires .* in params or as a global argument"
    ):
        mixed_edges(segments, total_n_bins=3)


def test_mixed_edges_total_n_bins_validation() -> None:
    """Tests that total_n_bins validation works correctly."""
    segments = [
        {"method": "equidistant", "n_bins": 2, "params": {"x_min": 0.0, "x_max": 1.0}},
        {"method": "equidistant", "n_bins": 2, "params": {"x_min": 1.0, "x_max": 2.0}},
    ]

    with pytest.raises(ValueError):
        mixed_edges(segments, total_n_bins=5)
