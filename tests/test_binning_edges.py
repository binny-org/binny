"""Unit tests for tomographic bin-edge construction and validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from binny.binning.edges import (
    mixed_edges_from_segments,
    resolve_bin_edges,
    segments_to_axes,
    validate_bin_edges,
)


def _toy_grid() -> np.ndarray:
    """Create a toy redshift grid."""
    return np.linspace(0.0, 2.0, 2001)


def _gaussian(z: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Create a Gaussian distribution."""
    return np.exp(-0.5 * ((z - mu) / sig) ** 2)


def test_photoz_segments_to_axes_converts_keys_and_types() -> None:
    """Tests that segment specifications use normalized keys and types."""
    segments: list[dict[str, Any]] = [
        {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 0.8},
        {"scheme": "equal_number", "n_bins": 3, "z_min": 0.8, "z_max": 2.0},
    ]

    out = segments_to_axes(segments)

    assert len(out) == 2
    assert out[0] == {
        "method": "equidistant",
        "n_bins": 2,
        "range": (0.0, 0.8),
        "params": {},
    }
    assert out[1] == {
        "method": "equal_number",
        "n_bins": 3,
        "range": (0.8, 2.0),
        "params": {},
    }


def test_photoz_segments_to_axes_missing_key_raises() -> None:
    """Tests that incomplete segment specifications are rejected."""
    segments = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0}]

    with pytest.raises(ValueError, match="requires x_min and x_max"):
        segments_to_axes(segments)


def test_mixed_edges_requires_paired_photoz_inputs() -> None:
    """Tests that selection axes and densities must be supplied together."""
    z = np.linspace(0.0, 2.0, 100)
    nz = np.exp(-z)

    segments = [
        {
            "scheme": "equal_number",
            "n_bins": 2,
            "z_min": 0.0,
            "z_max": 2.0,
        }
    ]

    with pytest.raises(
        ValueError,
        match="Provide both selection_axis and selection_density",
    ):
        mixed_edges_from_segments(
            segments,
            axis=z,
            density=nz,
            selection_axis=z,
            selection_density=None,
        )


def test_mixed_edges_uses_parent_axis() -> None:
    """Tests that mixed binning defaults to the parent redshift axis."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)
    segments = [{"scheme": "equidistant", "n_bins": 5, "z_min": 0.0, "z_max": 2.0}]

    edges = mixed_edges_from_segments(
        segments,
        axis=z,
        density=nz,
        selection_axis=None,
        selection_density=None,
    )

    np.testing.assert_allclose(edges, np.linspace(0.0, 2.0, 6))


def test_mixed_edges_uses_photoz_axis() -> None:
    """Tests that photo-z selection densities determine equal-number edges."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    selection_axis = np.linspace(0.0, 2.0, 801)
    selection_density = _gaussian(selection_axis, 0.6, 0.2)

    segments = [{"scheme": "equal_number", "n_bins": 4, "z_min": 0.0, "z_max": 2.0}]

    edges_photoz = mixed_edges_from_segments(
        segments,
        axis=z,
        density=nz,
        selection_axis=selection_axis,
        selection_density=selection_density,
    )

    edges_parent = mixed_edges_from_segments(
        segments,
        axis=z,
        density=nz,
        selection_axis=None,
        selection_density=None,
    )

    assert len(edges_photoz) == 5
    assert np.all(np.diff(edges_photoz) > 0.0)
    assert not np.allclose(edges_photoz, edges_parent)


def test_mixed_edges_total_bins_matches_segments() -> None:
    """Tests that mixed segments produce the requested total bin count."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    segments = [
        {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 1.0},
        {"scheme": "equidistant", "n_bins": 3, "z_min": 1.0, "z_max": 2.0},
    ]

    edges = mixed_edges_from_segments(
        segments,
        axis=z,
        density=nz,
        selection_axis=None,
        selection_density=None,
    )

    assert len(edges) == 6
    assert np.all(np.diff(edges) > 0.0)


def test_mixed_edges_missing_segment_key_raises() -> None:
    """Tests that mixed segments require complete interval boundaries."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)
    segments = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0}]

    with pytest.raises(ValueError, match="requires x_min and x_max"):
        mixed_edges_from_segments(
            segments,
            axis=z,
            density=nz,
            selection_axis=None,
            selection_density=None,
        )


@pytest.mark.parametrize(
    "edges",
    [
        [0.0, 0.5, 1.0],
        [0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ],
)
def test_validate_bin_edges_accepts_valid(edges: list[float]) -> None:
    """Tests that valid bin boundaries are accepted as float64 arrays."""
    result = validate_bin_edges(edges)

    np.testing.assert_array_equal(result, edges)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([0.0], "at least two"),
        ([[0.0, 1.0]], "must be 1D"),
        ([0.0, 0.5, 0.5, 1.0], "strictly increasing"),
        ([0.0, 1.0, 0.5], "strictly increasing"),
        ([0.0, np.nan, 1.0], "finite"),
        ([0.0, np.inf], "finite"),
    ],
)
def test_validate_bin_edges_rejects_invalid(
    edges: Any,
    message: str,
) -> None:
    """Tests that invalid bin boundaries raise informative errors."""
    with pytest.raises(ValueError, match=message):
        validate_bin_edges(edges)


def test_validate_bin_edges_require_within() -> None:
    """Tests that optional interval restrictions are enforced."""
    np.testing.assert_allclose(
        validate_bin_edges([0.2, 0.5, 0.8], require_within=(0.0, 1.0)),
        [0.2, 0.5, 0.8],
    )

    with pytest.raises(ValueError, match="must lie within"):
        validate_bin_edges([0.0, 0.5, 1.2], require_within=(0.0, 1.0))


def test_resolve_bin_edges_explicit() -> None:
    """Tests that explicitly supplied bin boundaries are preserved."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"edges": [0.2, 0.5, 1.2]},
    )

    np.testing.assert_allclose(edges, [0.2, 0.5, 1.2])


def test_resolve_bin_edges_equidistant() -> None:
    """Tests that equidistant binning generates uniformly spaced edges."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"scheme": "equidistant", "n_bins": 4},
    )

    np.testing.assert_allclose(edges, np.linspace(0.0, 2.0, 5))


def test_resolve_bin_edges_respects_range() -> None:
    """Tests that bin-range endpoints are not snapped to the input grid."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={
            "scheme": "equidistant",
            "n_bins": 4,
            "range": (0.23, 1.37),
        },
    )

    np.testing.assert_allclose(edges, np.linspace(0.23, 1.37, 5))


def test_resolve_bin_edges_equal_number() -> None:
    """Tests that uniform densities produce equidistant equal-number edges."""
    z = _toy_grid()
    nz = np.ones_like(z)

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"scheme": "equal_number", "n_bins": 4},
    )

    np.testing.assert_allclose(edges, np.linspace(0.0, 2.0, 5), atol=1e-3)


def test_resolve_bin_edges_mixed() -> None:
    """Tests that mixed segments are constructed through the edge resolver."""
    z = _toy_grid()
    nz = np.ones_like(z)

    segments = [
        {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 1.0},
        {"scheme": "equidistant", "n_bins": 2, "z_min": 1.0, "z_max": 2.0},
    ]

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"segments": segments},
    )

    np.testing.assert_allclose(edges, [0.0, 0.5, 1.0, 1.5, 2.0])


@pytest.mark.parametrize(
    "bins",
    [
        {"edges": [0.0, 1.0], "scheme": "equidistant", "n_bins": 2},
        {"edges": [0.0, 1.0], "segments": []},
        {
            "scheme": "equidistant",
            "n_bins": 2,
            "segments": [],
        },
    ],
)
def test_resolve_bin_edges_rejects_conflicting_inputs(
    bins: dict[str, Any],
) -> None:
    """Tests that conflicting bin specification modes are rejected."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="Specify exactly one"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins=bins,
        )


def test_resolve_bin_edges_requires_scheme() -> None:
    """Tests that bin specifications must select exactly one mode."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="Specify exactly one"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={},
        )


def test_resolve_bin_edges_rejects_unknown_scheme() -> None:
    """Tests that unsupported binning schemes are rejected."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="Unsupported binning_scheme"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"scheme": "unknown", "n_bins": 4},
        )


def test_resolve_bin_edges_generalized() -> None:
    """Tests that generalized alpha-beta binning works through the resolver."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    dchi_dz = 1.0 / np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)
    chi = np.zeros_like(z)
    chi[1:] = np.cumsum(0.5 * (dchi_dz[1:] + dchi_dz[:-1]) * np.diff(z))

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={
            "scheme": "generalized",
            "n_bins": 5,
            "params": {
                "alpha": 0.5,
                "beta": 0.5,
                "comoving_distance": chi,
            },
        },
    )

    assert edges.shape == (6,)
    assert np.all(np.diff(edges) > 0)


def test_resolve_bin_edges_generalized_requires_n_bins() -> None:
    """Tests that generalized binning requires a valid bin count."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    with pytest.raises(ValueError, match="n_bins"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={
                "scheme": "generalized",
                "n_bins": None,
            },
        )


@pytest.mark.parametrize(
    ("scheme", "expected"),
    [
        ("equidistant", np.linspace(0.2, 1.8, 5)),
        ("equal_number", np.linspace(0.2, 1.8, 5)),
    ],
)
def test_resolve_bin_edges_range_applies_to_schemes(
    scheme: str,
    expected: np.ndarray,
) -> None:
    """Tests that interval restrictions apply to multiple binning schemes."""
    z = _toy_grid()
    nz = np.ones_like(z)

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={
            "scheme": scheme,
            "n_bins": 4,
            "range": (0.2, 1.8),
        },
    )

    np.testing.assert_allclose(edges, expected, atol=1e-3)


def test_resolve_bin_edges_rejects_unknown_params() -> None:
    """Tests that unsupported scheme parameters are rejected."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="Unsupported parameters"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={
                "scheme": "equidistant",
                "n_bins": 4,
                "params": {"alpha": 0.5},
            },
        )


def test_resolve_bin_edges_rejects_range_outside_grid() -> None:
    """Tests that binning ranges outside the sampled grid are rejected."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="within the binning grid"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={
                "scheme": "equidistant",
                "n_bins": 4,
                "range": (0.2, 2.5),
            },
        )


def test_generalized_nonzero_beta_via_bins() -> None:
    """Tests that generalized binning supports nonzero beta and custom ranges."""
    z = np.linspace(0.0, 3.0, 501)
    nz = z**2 * np.exp(-z / 0.5)
    chi = 3000.0 * np.log1p(z)

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={
            "scheme": "generalized",
            "n_bins": 5,
            "range": [0.2, 2.5],
            "params": {
                "alpha": 0.5,
                "beta": 1.0,
                "comoving_distance": chi,
            },
        },
    )

    assert edges.shape == (6,)
    assert np.all(np.diff(edges) > 0)
    assert np.isclose(edges[0], 0.2)
    assert np.isclose(edges[-1], 2.5)


def test_generalized_derivative_methods_through_resolver() -> None:
    """Tests that analytic and numerical derivatives give consistent edges."""
    axis = np.linspace(0.0, 2.0, 101)
    density = np.exp(-axis)
    chi = 1000.0 * axis + 100.0 * axis**2

    def calculate(method: str) -> np.ndarray:
        return resolve_bin_edges(
            axis=axis,
            density=density,
            bins={
                "scheme": "generalized",
                "n_bins": 4,
                "params": {
                    "alpha": 1.0,
                    "beta": 1.0,
                    "comoving_distance": chi,
                    "derivative_method": method,
                },
            },
        )

    analytic = calculate("analytic")
    numerical = calculate("numerical")

    np.testing.assert_allclose(analytic, numerical, rtol=1e-3, atol=1e-3)
