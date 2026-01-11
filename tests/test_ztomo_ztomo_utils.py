"""Unit tests for ztomo_utils.py"""

from __future__ import annotations

import numpy as np
import pytest

from binny.ztomo.ztomo_utils import mixed_edges


def _make_axes():
    """Creates z and nz axes for testing."""
    z = np.linspace(0.0, 3.0, 301)
    nz = np.exp(-z) * (1.0 + z)  # positive weights
    return z, nz


def test_mixed_edges_equidistant_two_segments_concatenates_and_dedupes_boundary():
    """Tests that mixed_edges returns expected edges for a simple case
    with two segments."""
    z, nz = _make_axes()

    segments = [
        {"z_min": 0.0, "z_max": 1.0, "scheme": "equidistant", "n_bins": 2},
        {"z_min": 1.0, "z_max": 3.0, "scheme": "eq", "n_bins": 4},
    ]

    edges = mixed_edges(
        segments,
        z_axis=z,
        nz_axis=nz,
        z_ph=None,
        nz_ph=None,
        normalize_input=True,
        norm_method="trapezoid",
    )

    # total bins = 2 + 4 => edges length = 7
    assert edges.shape == (7,)
    assert np.isfinite(edges).all()
    assert np.all(np.diff(edges) > 0)

    # boundary appears exactly once
    assert np.sum(np.isclose(edges, 1.0, atol=1e-12, rtol=0.0)) == 1

    # endpoints match
    assert edges[0] == pytest.approx(0.0, abs=0.0)
    assert edges[-1] == pytest.approx(3.0, abs=0.0)


def test_mixed_edges_equal_number_proxy_uses_z_nz_when_zph_none():
    """Tests that mixed_edges with equal-number segment uses z and nz
    when z_ph and nz_ph are not provided."""
    z, nz = _make_axes()

    segments = [
        {"z_min": 0.0, "z_max": 1.5, "scheme": "equal_number", "n_bins": 3},
        {"z_min": 1.5, "z_max": 3.0, "scheme": "equidistant", "n_bins": 3},
    ]

    edges = mixed_edges(
        segments,
        z_axis=z,
        nz_axis=nz,
        z_ph=None,
        nz_ph=None,
        normalize_input=True,
        norm_method="trapezoid",
    )

    assert edges.shape == (3 + 3 + 1,)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == pytest.approx(0.0, abs=0.0)
    assert edges[-1] == pytest.approx(3.0, abs=0.0)

    # forced segment endpoints present exactly
    assert np.isclose(edges[0], 0.0, atol=0.0, rtol=0.0)
    assert np.isclose(edges[3], 1.5, atol=1e-12, rtol=0.0)
    assert np.isclose(edges[-1], 3.0, atol=0.0, rtol=0.0)


def test_mixed_edges_equal_number_with_explicit_zph_nzph_changes_result():
    """Tests that mixed_edges with equal-number segment uses z_ph and nz_ph
    when provided, changing the resulting edges."""
    z, nz = _make_axes()

    # Create an observed-z axis that is different from z
    # to ensure code path is used.
    z_ph = np.linspace(0.0, 3.0, 301)
    # Put more weight at high z_ph than nz(z) does (skews quantiles)
    nz_ph = (z_ph + 0.1) ** 2

    segments = [
        {"z_min": 0.0, "z_max": 3.0, "scheme": "equal_number", "n_bins": 6},
    ]

    edges_proxy = mixed_edges(
        segments,
        z_axis=z,
        nz_axis=nz,
        z_ph=None,
        nz_ph=None,
        normalize_input=True,
        norm_method="trapezoid",
    )
    edges_obs = mixed_edges(
        segments,
        z_axis=z,
        nz_axis=nz,
        z_ph=z_ph,
        nz_ph=nz_ph,
        normalize_input=True,
        norm_method="trapezoid",
    )

    assert edges_proxy.shape == edges_obs.shape == (7,)
    assert np.all(np.diff(edges_proxy) > 0)
    assert np.all(np.diff(edges_obs) > 0)
    assert not np.allclose(edges_proxy, edges_obs, rtol=0.0, atol=1e-10)


def test_mixed_edges_raises_if_segments_empty():
    """Tests that mixed_edges raises ValueError if segments is empty."""
    z, nz = _make_axes()
    with pytest.raises(ValueError, match="non-empty"):
        mixed_edges(
            [],
            z_axis=z,
            nz_axis=nz,
            z_ph=None,
            nz_ph=None,
            normalize_input=True,
            norm_method="trapezoid",
        )


def test_mixed_edges_raises_if_only_one_of_zph_nzph_provided():
    """Tests that mixed_edges raises ValueError if
    only one of z_ph and nz_ph is provided."""
    z, nz = _make_axes()
    segments = [{"z_min": 0.0, "z_max": 1.0, "scheme": "equidistant", "n_bins": 2}]

    with pytest.raises(ValueError, match="Provide both z_ph and nz_ph"):
        mixed_edges(
            segments,
            z_axis=z,
            nz_axis=nz,
            z_ph=z,
            nz_ph=None,
            normalize_input=True,
            norm_method="trapezoid",
        )

    with pytest.raises(ValueError, match="Provide both z_ph and nz_ph"):
        mixed_edges(
            segments,
            z_axis=z,
            nz_axis=nz,
            z_ph=None,
            nz_ph=nz,
            normalize_input=True,
            norm_method="trapezoid",
        )


def test_mixed_edges_raises_on_missing_required_segment_key():
    """Tests that mixed_edges raises ValueError if
    a segment is missing a required key."""
    z, nz = _make_axes()
    segments = [{"z_min": 0.0, "z_max": 1.0, "scheme": "equidistant"}]

    with pytest.raises(ValueError, match="Missing key.*n_bins"):
        mixed_edges(
            segments,
            z_axis=z,
            nz_axis=nz,
            z_ph=None,
            nz_ph=None,
            normalize_input=True,
            norm_method="trapezoid",
        )


def test_mixed_edges_raises_when_segment_not_contiguous():
    """Tests that mixed_edges raises ValueError if a segment is not contiguous."""
    z, nz = _make_axes()

    segments = [
        {"z_min": 0.0, "z_max": 1.0, "scheme": "equidistant", "n_bins": 2},
        {"z_min": 1.1, "z_max": 2.0, "scheme": "equidistant", "n_bins": 2},
    ]

    with pytest.raises(ValueError, match="contiguous"):
        mixed_edges(
            segments,
            z_axis=z,
            nz_axis=nz,
            z_ph=None,
            nz_ph=None,
            normalize_input=True,
            norm_method="trapezoid",
        )


def test_mixed_edges_raises_when_segments_overlap_or_decrease():
    """Tests that mixed_edges raises ValueError if segments overlap or decrease."""
    z, nz = _make_axes()

    segments = [
        {"z_min": 0.0, "z_max": 2.0, "scheme": "equidistant", "n_bins": 2},
        {"z_min": 1.5, "z_max": 3.0, "scheme": "equidistant", "n_bins": 2},
    ]

    with pytest.raises(ValueError, match="non-overlapping"):
        mixed_edges(
            segments,
            z_axis=z,
            nz_axis=nz,
            z_ph=None,
            nz_ph=None,
            normalize_input=True,
            norm_method="trapezoid",
        )


def test_mixed_edges_raises_on_invalid_segment_bounds():
    """Tests that mixed_edges raises ValueError if a segment has invalid bounds."""
    z, nz = _make_axes()

    segments_bad = [{"z_min": 1.0, "z_max": 1.0, "scheme": "equidistant", "n_bins": 2}]
    with pytest.raises(ValueError, match="z_max > z_min"):
        mixed_edges(
            segments_bad,
            z_axis=z,
            nz_axis=nz,
            z_ph=None,
            nz_ph=None,
            normalize_input=True,
            norm_method="trapezoid",
        )

    segments_nan = [
        {"z_min": np.nan, "z_max": 1.0, "scheme": "equidistant", "n_bins": 2}
    ]
    with pytest.raises(ValueError, match="finite"):
        mixed_edges(
            segments_nan,
            z_axis=z,
            nz_axis=nz,
            z_ph=None,
            nz_ph=None,
            normalize_input=True,
            norm_method="trapezoid",
        )


def test_mixed_edges_raises_on_unsupported_scheme():
    """Tests that mixed_edges raises ValueError if
    a segment has an unsupported scheme."""
    z, nz = _make_axes()
    segments = [{"z_min": 0.0, "z_max": 1.0, "scheme": "log", "n_bins": 3}]

    with pytest.raises(ValueError, match="Unsupported segment scheme"):
        mixed_edges(
            segments,
            z_axis=z,
            nz_axis=nz,
            z_ph=None,
            nz_ph=None,
            normalize_input=True,
            norm_method="trapezoid",
        )


def test_mixed_edges_equal_number_segment_raises_if_too_few_points():
    """Tests that mixed_edges raises ValueError if a segment has too few points."""
    # Make a very coarse axis so the segment slice has <2 points.
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    nz = np.array([1.0, 1.0, 1.0], dtype=float)

    segments = [{"z_min": 0.1, "z_max": 0.2, "scheme": "equal_number", "n_bins": 2}]

    with pytest.raises(ValueError, match="too few points"):
        mixed_edges(
            segments,
            z_axis=z,
            nz_axis=nz,
            z_ph=None,
            nz_ph=None,
            normalize_input=True,
            norm_method="trapezoid",
        )
