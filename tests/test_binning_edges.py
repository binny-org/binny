"""Unit tests for tomographic bin-edge construction and validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from binny.binning.edges import (
    mixed_edges_from_segments,
    photoz_segments_to_axes,
    resolve_bin_edges,
    validate_bin_edges,
)


def _toy_grid() -> np.ndarray:
    """Create a toy redshift grid."""
    return np.linspace(0.0, 2.0, 2001)


def _gaussian(z: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Create a Gaussian distribution."""
    return np.exp(-0.5 * ((z - mu) / sig) ** 2)


def test_photoz_segments_to_axes_converts_keys_and_types() -> None:
    """Test conversion of segment specifications."""
    segments: list[dict[str, Any]] = [
        {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 0.8},
        {"scheme": "equal_number", "n_bins": 3, "z_min": 0.8, "z_max": 2.0},
    ]

    out = photoz_segments_to_axes(segments)

    assert len(out) == 2
    assert out[0] == {
        "method": "equidistant",
        "n_bins": 2,
        "params": {"x_min": 0.0, "x_max": 0.8},
    }
    assert out[1] == {
        "method": "equal_number",
        "n_bins": 3,
        "params": {"x_min": 0.8, "x_max": 2.0},
    }


def test_photoz_segments_to_axes_missing_key_raises() -> None:
    """Test rejection of incomplete segment specifications."""
    segments = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0}]

    with pytest.raises(ValueError, match="Missing key in mixed bin spec"):
        photoz_segments_to_axes(segments)


def test_mixed_edges_requires_paired_photoz_inputs():
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

    with pytest.raises(ValueError, match="Provide both z_ph and nz_ph"):
        mixed_edges_from_segments(
            segments,
            axis=z,
            density=nz,
            z_ph=z,
            nz_ph=None,
        )


def test_mixed_edges_uses_parent_axis() -> None:
    """Test default parent-axis selection."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)
    segments = [{"scheme": "equidistant", "n_bins": 5, "z_min": 0.0, "z_max": 2.0}]

    edges = mixed_edges_from_segments(
        segments,
        axis=z,
        density=nz,
        z_ph=None,
        nz_ph=None,
    )

    np.testing.assert_allclose(edges, np.linspace(0.0, 2.0, 6))


def test_mixed_edges_uses_photoz_axis() -> None:
    """Test that photo-z weights determine equal-number edges."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    z_ph = np.linspace(0.0, 2.0, 801)
    nz_ph = _gaussian(z_ph, 0.6, 0.2)

    segments = [{"scheme": "equal_number", "n_bins": 4, "z_min": 0.0, "z_max": 2.0}]

    edges_photoz = mixed_edges_from_segments(
        segments,
        axis=z,
        density=nz,
        z_ph=z_ph,
        nz_ph=nz_ph,
    )

    edges_parent = mixed_edges_from_segments(
        segments,
        axis=z,
        density=nz,
        z_ph=None,
        nz_ph=None,
    )

    assert len(edges_photoz) == 5
    assert np.all(np.diff(edges_photoz) > 0.0)
    assert not np.allclose(edges_photoz, edges_parent)


def test_mixed_edges_total_bins_matches_segments() -> None:
    """Test total number of bins across multiple segments."""
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
        z_ph=None,
        nz_ph=None,
    )

    assert len(edges) == 6
    assert np.all(np.diff(edges) > 0.0)


def test_mixed_edges_missing_segment_key_raises() -> None:
    """Test rejection of incomplete mixed segments."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)
    segments = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0}]

    with pytest.raises(ValueError, match="Missing key in mixed bin spec"):
        mixed_edges_from_segments(
            segments,
            axis=z,
            density=nz,
            z_ph=None,
            nz_ph=None,
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
    """Test valid edge arrays."""
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
    """Test rejection of invalid edges."""
    with pytest.raises(ValueError, match=message):
        validate_bin_edges(edges)


def test_validate_bin_edges_require_within() -> None:
    """Test optional interval restriction."""
    np.testing.assert_allclose(
        validate_bin_edges([0.2, 0.5, 0.8], require_within=(0.0, 1.0)),
        [0.2, 0.5, 0.8],
    )

    with pytest.raises(ValueError, match="must lie within"):
        validate_bin_edges([0.0, 0.5, 1.2], require_within=(0.0, 1.0))


def test_resolve_bin_edges_explicit() -> None:
    """Test explicitly supplied edges."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"edges": [0.2, 0.5, 1.2]},
    )

    np.testing.assert_allclose(edges, [0.2, 0.5, 1.2])


def test_resolve_bin_edges_equidistant() -> None:
    """Test equally spaced edges."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"scheme": "equidistant", "n_bins": 4},
    )

    np.testing.assert_allclose(edges, np.linspace(0.0, 2.0, 5))


def test_resolve_bin_edges_respects_range() -> None:
    """Test that bin range endpoints are not snapped to the grid."""
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
    """Test equal-number edges for a uniform distribution."""
    z = _toy_grid()
    nz = np.ones_like(z)

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"scheme": "equal_number", "n_bins": 4},
    )

    np.testing.assert_allclose(edges, np.linspace(0.0, 2.0, 5), atol=1e-3)


def test_resolve_bin_edges_mixed() -> None:
    """Test mixed-segment edge construction through the resolver."""
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
    """Test that multiple bin specification modes are rejected."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="Specify exactly one"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins=bins,
        )


def test_resolve_bin_edges_requires_scheme() -> None:
    """Test rejection of a bin specification without a mode."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="Specify exactly one"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={},
        )


def test_resolve_bin_edges_rejects_unknown_scheme() -> None:
    """Test unsupported binning schemes."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="Unsupported binning_scheme"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"scheme": "unknown", "n_bins": 4},
        )


def test_resolve_bin_edges_generalized() -> None:
    """Test generalized alpha-beta binning through the resolver."""
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
    """Test that generalized binning requires a bin count."""
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
    """Test range restriction for multiple generating schemes."""
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
    """Test rejection of unsupported scheme parameters."""
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
    """Test rejection of ranges outside the sampled grid."""
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


def test_generalized_nonzero_beta_via_bins():
    import numpy as np

    from binny.binning.edges import resolve_bin_edges

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
