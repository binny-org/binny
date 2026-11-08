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


def test_mixed_edges_requires_paired_photoz_inputs() -> None:
    """Require both photo-z axis and weights."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)
    segments = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 2.0}]

    with pytest.raises(ValueError, match="Provide both z_ph and nz_ph"):
        mixed_edges_from_segments(
            segments,
            z_axis=z,
            nz_axis=nz,
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
        z_axis=z,
        nz_axis=nz,
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
        z_axis=z,
        nz_axis=nz,
        z_ph=z_ph,
        nz_ph=nz_ph,
    )

    edges_parent = mixed_edges_from_segments(
        segments,
        z_axis=z,
        nz_axis=nz,
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
        z_axis=z,
        nz_axis=nz,
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
            z_axis=z,
            nz_axis=nz,
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
        z_axis=z,
        nz_axis=nz,
        bin_edges=[0.2, 0.5, 1.2],
        binning_scheme=None,
        n_bins=None,
    )

    np.testing.assert_allclose(edges, [0.2, 0.5, 1.2])


def test_resolve_bin_edges_equidistant() -> None:
    """Test equally spaced edges."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    edges = resolve_bin_edges(
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme="equidistant",
        n_bins=4,
    )

    np.testing.assert_allclose(edges, np.linspace(0.0, 2.0, 5))


def test_resolve_bin_edges_respects_range() -> None:
    """Test that a specified bin range is not snapped to the grid."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    edges = resolve_bin_edges(
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme="equidistant",
        n_bins=4,
        bin_range=(0.23, 1.37),
    )

    np.testing.assert_allclose(edges, np.linspace(0.23, 1.37, 5))


def test_resolve_bin_edges_equal_number() -> None:
    """Test equal-number edges for a uniform distribution."""
    z = _toy_grid()
    nz = np.ones_like(z)

    edges = resolve_bin_edges(
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme="equal_number",
        n_bins=4,
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
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme=segments,
        n_bins=None,
    )

    np.testing.assert_allclose(edges, [0.0, 0.5, 1.0, 1.5, 2.0])


def test_resolve_bin_edges_rejects_conflicting_inputs() -> None:
    """Test mutually exclusive edge specifications."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="Provide either bin_edges"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=[0.0, 1.0],
            binning_scheme="equidistant",
            n_bins=2,
        )


def test_resolve_bin_edges_requires_scheme() -> None:
    """Test missing edge specification."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="must provide binning_scheme"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme=None,
            n_bins=None,
        )


def test_resolve_bin_edges_rejects_unknown_scheme() -> None:
    """Test unsupported binning schemes."""
    z = _toy_grid()
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match="Unsupported binning_scheme"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme="unknown",
            n_bins=4,
        )
