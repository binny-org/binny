"""Unit tests for ``mixed_edges_from_segments`` helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from binny.nz_tomo.nz_tomo_utils import (
    extract_bin_edges_from_meta,
    mixed_edges_from_segments,
    photoz_segments_to_axes,
    resolve_bin_edges_for_leakage,
    resolve_n_bins_for_builder,
)


def _toy_grid() -> np.ndarray:
    """Creates a toy redshift grid for testing."""
    return np.linspace(0.0, 2.0, 2001)


def _gaussian(z: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Creates a Gaussian curve on grid z with mean mu and std sig."""
    return np.exp(-0.5 * ((z - mu) / sig) ** 2)


def test_photoz_segments_to_axes_converts_keys_and_types() -> None:
    """Tests that _photoz_segments_to_axes converts segment keys and types."""
    segments: list[dict[str, Any]] = [
        {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 0.8},
        {"scheme": "equal_number", "n_bins": 3, "z_min": 0.8, "z_max": 2.0},
    ]

    out = photoz_segments_to_axes(segments)

    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["method"] == "equidistant"
    assert out[0]["n_bins"] == 2
    assert out[0]["params"] == {"x_min": 0.0, "x_max": 0.8}
    assert out[1]["method"] == "equal_number"
    assert out[1]["n_bins"] == 3
    assert out[1]["params"] == {"x_min": 0.8, "x_max": 2.0}


def test_photoz_segments_to_axes_missing_key_raises() -> None:
    """Tests that _photoz_segments_to_axes raises on missing required keys."""
    segments: list[dict[str, Any]] = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0}]

    with pytest.raises(ValueError, match="Missing key in mixed bin spec"):
        photoz_segments_to_axes(segments)


def test_mixed_edges_from_segments_requires_both_photoz_inputs_or_none() -> None:
    """Tests that mixed_edges_from_segments enforces paired z_ph/nz_ph."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)
    segments = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 2.0}]

    msg = "Provide both z_ph and nz_ph, or neither"
    with pytest.raises(ValueError, match=msg):
        mixed_edges_from_segments(
            segments,
            z_axis=z,
            nz_axis=nz,
            z_ph=z,
            nz_ph=None,
        )


def test_mixed_edges_from_segments_uses_axis_proxy_when_photoz_missing() -> None:
    """Tests that mixed_edges_from_segments uses (z_axis, nz_axis) by default."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)
    segments = [
        {"scheme": "equidistant", "n_bins": 5, "z_min": 0.0, "z_max": 2.0},
    ]

    edges = mixed_edges_from_segments(
        segments,
        z_axis=z,
        nz_axis=nz,
        z_ph=None,
        nz_ph=None,
    )

    assert isinstance(edges, np.ndarray)
    assert edges.ndim == 1
    assert len(edges) == 6
    assert np.all(np.diff(edges) > 0.0)


def test_mixed_edges_from_segments_uses_photoz_axis_when_provided() -> None:
    """Tests that mixed_edges_from_segments uses (z_ph, nz_ph) when provided."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.3)

    z_ph = np.linspace(0.0, 2.0, 801)
    nz_ph = _gaussian(z_ph, 0.9, 0.35)

    segments = [
        {"scheme": "equidistant", "n_bins": 5, "z_min": 0.0, "z_max": 2.0},
    ]

    edges = mixed_edges_from_segments(
        segments,
        z_axis=z,
        nz_axis=nz,
        z_ph=z_ph,
        nz_ph=nz_ph,
    )

    assert isinstance(edges, np.ndarray)
    assert edges.ndim == 1
    assert len(edges) == 6
    assert np.all(np.diff(edges) > 0.0)


def test_mixed_edges_from_segments_total_bins_matches_sum_of_segments() -> None:
    """Tests that mixed_edges_from_segments returns total_n_bins + 1 edges."""
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

    assert len(edges) == (2 + 3 + 1)


def test_mixed_edges_from_segments_missing_segment_key_raises() -> None:
    """Tests that mixed_edges_from_segments raises on missing segment keys."""
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


def test_extract_bin_edges_from_meta_returns_none_if_missing():
    """Tests that extract_bin_edges_from_meta returns None when bin_edges absent."""
    assert extract_bin_edges_from_meta({}) is None
    assert extract_bin_edges_from_meta({"other": 123}) is None


def test_extract_bin_edges_from_meta_returns_float64_array():
    """Tests that extract_bin_edges_from_meta returns float64 array when present."""
    meta = {"bin_edges": [0.0, 0.5, 1.0]}
    out = extract_bin_edges_from_meta(meta)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_resolve_n_bins_for_builder_prefers_bin_edges():
    """Tests that resolve_n_bins_for_builder returns None when bin_edges provided."""
    assert resolve_n_bins_for_builder(bin_edges=[0.0, 1.0], n_bins=5) is None


def test_resolve_n_bins_for_builder_returns_n_bins_when_no_edges():
    """Tests that resolve_n_bins_for_builder passes through n_bins if no bin_edges."""
    assert resolve_n_bins_for_builder(bin_edges=None, n_bins=5) == 5
    assert resolve_n_bins_for_builder(bin_edges=None, n_bins=None) is None


def test_resolve_bin_edges_for_leakage_uses_explicit_edges():
    """Tests that resolve_bin_edges_for_leakage uses explicit bin_edges when provided."""
    out = resolve_bin_edges_for_leakage(bin_edges=[0.0, 0.4, 1.0], cached_bin_edges=None)
    assert out.dtype == np.float64
    assert np.allclose(out, [0.0, 0.4, 1.0])


def test_resolve_bin_edges_for_leakage_uses_cached_when_explicit_missing():
    """Tests that resolve_bin_edges_for_leakage uses cached edges when explicit missing."""
    out = resolve_bin_edges_for_leakage(bin_edges=None, cached_bin_edges=[0.0, 0.5, 1.0])
    assert out.dtype == np.float64
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_resolve_bin_edges_for_leakage_raises_if_neither_available():
    """Tests that resolve_bin_edges_for_leakage raises if no explicit or cached edges."""
    with pytest.raises(ValueError, match=r"bin_edges is required"):
        resolve_bin_edges_for_leakage(bin_edges=None, cached_bin_edges=None)
